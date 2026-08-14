#!/usr/bin/env python3
"""Safely assign IDs to a growing daisy chain of XL330 servos."""

from __future__ import annotations

import argparse
import sys

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:
    sys.exit("dynamixel_sdk is missing. Run: python -m pip install dynamixel-sdk")


ADDR_ID = 7
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

XL330_MODELS = {1190: "XL330-M077-T", 1200: "XL330-M288-T"}
MIN_VOLTAGE = 3.7
MAX_VOLTAGE = 6.0


def read_register(packet: PacketHandler, port: PortHandler, motor_id: int, address: int, size: int) -> int:
    if size == 1:
        value, result, _error = packet.read1ByteTxRx(port, motor_id, address)
    elif size == 2:
        value, result, _error = packet.read2ByteTxRx(port, motor_id, address)
    else:
        raise ValueError(f"Unsupported register size: {size}")

    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    return value


def write_byte(packet: PacketHandler, port: PortHandler, motor_id: int, address: int, value: int) -> None:
    result, error = packet.write1ByteTxRx(port, motor_id, address, value)
    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error:
        raise RuntimeError(packet.getRxPacketError(error))


def scan(packet: PacketHandler, port: PortHandler) -> dict[int, list[int]]:
    found, result = packet.broadcastPing(port)
    if result != COMM_SUCCESS:
        return {}
    return found


def hardware_error_names(value: int) -> str:
    names = []
    for bit, name in (
        (0, "input voltage"),
        (2, "overheating"),
        (3, "motor encoder"),
        (4, "electrical shock/circuit"),
        (5, "overload"),
    ):
        if value & (1 << bit):
            names.append(name)
    return ", ".join(names) or f"unknown value {value}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign IDs while adding one factory-ID-1 XL330 at a time. "
            "Use descending IDs so an existing ID 1 never collides with the next new motor."
        )
    )
    parser.add_argument("--port", required=True, help="U2D2 device path")
    parser.add_argument(
        "--ids",
        required=True,
        nargs="+",
        type=int,
        help="IDs in connection order, normally descending (example: 8 7 6 5 4 3 2 1)",
    )
    parser.add_argument("--baudrate", type=int, default=57_600)
    parser.add_argument("--source-id", type=int, default=1, help="Factory ID of each newly added motor")
    args = parser.parse_args()

    if len(args.ids) != len(set(args.ids)):
        parser.error("--ids must be unique")
    if any(motor_id < 0 or motor_id > 252 for motor_id in args.ids):
        parser.error("IDs must be between 0 and 252")
    if args.source_id in args.ids and args.ids[-1] != args.source_id:
        parser.error(f"source ID {args.source_id} must be last to prevent an ID collision")
    return args


def main() -> int:
    args = parse_args()
    port = PortHandler(args.port)
    packet = PacketHandler(2.0)

    if not port.openPort():
        sys.exit(f"Could not open {args.port}")
    if not port.setBaudRate(args.baudrate):
        port.closePort()
        sys.exit(f"Could not set host baud rate to {args.baudrate}")

    print("This script sends no position or velocity commands.")
    print(f"Planned IDs: {args.ids}")

    try:
        for target_id in args.ids:
            input(
                f"\nPOWER OFF, add exactly one new ID-{args.source_id} motor, "
                f"then power on and press Enter to assign ID {target_id}: "
            )

            found = scan(packet, port)
            found_ids = sorted(found)
            print(f"Detected IDs: {found_ids or 'none'}")
            if args.source_id not in found:
                raise RuntimeError(
                    f"No motor at source ID {args.source_id}. Power off and check power, TTL wiring, and ID collisions."
                )
            if target_id != args.source_id and target_id in found:
                raise RuntimeError(f"Target ID {target_id} already exists; refusing to create a duplicate")

            model, result, _error = packet.ping(port, args.source_id)
            if result != COMM_SUCCESS:
                raise RuntimeError(packet.getTxRxResult(result))
            if model not in XL330_MODELS:
                raise RuntimeError(f"Model {model} is not a supported XL330; refusing to use XL330 limits")

            voltage = read_register(packet, port, args.source_id, ADDR_PRESENT_VOLTAGE, 2) / 10.0
            temperature = read_register(packet, port, args.source_id, ADDR_PRESENT_TEMPERATURE, 1)
            torque = read_register(packet, port, args.source_id, ADDR_TORQUE_ENABLE, 1)
            hardware_error = read_register(packet, port, args.source_id, ADDR_HARDWARE_ERROR, 1)
            print(
                f"Source: {XL330_MODELS[model]}, {voltage:.1f} V, {temperature} C, "
                f"torque={'ON' if torque else 'off'}, hardware_error={hardware_error}"
            )

            if not MIN_VOLTAGE <= voltage <= MAX_VOLTAGE:
                raise RuntimeError(
                    f"UNSAFE XL330 VOLTAGE: {voltage:.1f} V; required {MIN_VOLTAGE:.1f}-{MAX_VOLTAGE:.1f} V"
                )
            if hardware_error:
                raise RuntimeError(f"Hardware error: {hardware_error_names(hardware_error)}")
            if torque:
                print("Disabling torque before the EEPROM ID write")
                write_byte(packet, port, args.source_id, ADDR_TORQUE_ENABLE, 0)
                if read_register(packet, port, args.source_id, ADDR_TORQUE_ENABLE, 1) != 0:
                    raise RuntimeError("Torque did not disable")

            if target_id == args.source_id:
                print(f"Leaving final motor at ID {target_id}")
            else:
                write_byte(packet, port, args.source_id, ADDR_ID, target_id)
                model_after, result, error = packet.ping(port, target_id)
                if result != COMM_SUCCESS or error or model_after != model:
                    raise RuntimeError(f"ID {target_id} verification failed")
                print(f"Assigned and verified ID {target_id}")

            print("POWER OFF before connecting the next motor.")
    finally:
        port.closePort()

    print(f"\nComplete. Final detected IDs after the next power-on should be: {sorted(args.ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
