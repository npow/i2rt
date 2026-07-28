#!/usr/bin/env python3
"""Briefly probe YAM motor status without starting a robot or a rollout.

For each configured motor this script sends only the DaMiao enable command,
records its feedback, and immediately disables it again.  It never sends a
joint-position target, writes calibration/zero registers, or opens cameras.

The motor interface may transactionally clear the documented ``0xD`` CAN
watchdog state so that a powered-but-idle arm can report its health.  All
other faults, especially calibration faults, remain fail-closed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config


@dataclass
class Result:
    channel: str
    motor_id: int
    motor_type: str
    status: str
    detail: str


def configured_motors() -> List[tuple[int, str]]:
    arm = _load_arm_config(ArmType.YAM)
    gripper_motor_type = GripperType.LINEAR_4310.get_motor_type(ArmType.YAM)
    return [(int(motor_id), str(motor_type)) for motor_id, motor_type in arm.motor_list] + [
        (0x07, gripper_motor_type)
    ]


def probe_channel(channel: str) -> List[Result]:
    interface = DMSingleMotorCanInterface(
        channel=channel,
        bustype="socketcan",
        control_mode=ControlMode.MIT,
        name=f"yam-preflight-{channel}",
    )
    results: List[Result] = []
    try:
        for motor_id, motor_type in configured_motors():
            enabled = False
            try:
                feedback = interface.motor_on(motor_id, motor_type, max_retry=2)
                enabled = True
                # Feed one explicit zero-torque command at the motor's live
                # position before disabling it.  This never requests motion
                # and avoids leaving a just-enabled motor with an unspecified
                # command state during the brief probe window.
                feedback = interface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=feedback.position,
                    vel=0.0,
                    kp=0.0,
                    kd=0.0,
                    torque=0.0,
                    max_retry=1,
                )
                results.append(
                    Result(
                        channel=channel,
                        motor_id=motor_id,
                        motor_type=motor_type,
                        status=feedback.error_message,
                        detail=(
                            f"MOS={feedback.temperature_mos:.1f}C "
                            f"rotor={feedback.temperature_rotor:.1f}C"
                        ),
                    )
                )
            except Exception as exc:  # keep probing the other independently-disabled motors
                results.append(Result(channel, motor_id, motor_type, "ERROR", str(exc)))
            finally:
                # An enabled motor is disabled immediately.  Do not leave the
                # preflight process holding an arm in motor mode.
                if enabled:
                    try:
                        interface.motor_off(motor_id)
                    except Exception as exc:  # report cleanup failure explicitly
                        results.append(Result(channel, motor_id, motor_type, "OFF_ERROR", str(exc)))
    finally:
        interface.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("channels", nargs="*", default=["can_left", "can_right"])
    args = parser.parse_args()

    all_results: List[Result] = []
    for channel in args.channels:
        print(f"\n=== {channel}: enable/read/disable each motor ===")
        try:
            all_results.extend(probe_channel(channel))
        except Exception as exc:
            all_results.append(Result(channel, -1, "n/a", "CHANNEL_ERROR", str(exc)))

    for result in all_results:
        print(
            f"{result.channel:>10} motor {result.motor_id}: "
            f"{result.status:<42} {result.detail}"
        )

    return 0 if all(result.status == "normal" for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
