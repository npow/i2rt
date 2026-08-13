#!/usr/bin/env python3
"""Briefly probe YAM motor status without starting a robot or a rollout.

For each configured motor this script sends only the DaMiao enable command,
records its feedback, and immediately disables it again.  It never sends a
joint-position target, writes calibration/zero registers, or opens cameras.

The motor interface may transactionally clear the documented ``0xD`` CAN
watchdog state so that a powered-but-idle arm can report its health.  A
latched output-shaft calibration status is also cleared automatically (the
driver's startup default): at most once per motor, requiring a fresh normal
enable reply and a position that is continuous across the clear.  Pass
``--no-clear-latched-calibration`` to keep calibration faults fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import tyro

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config


@dataclass
class Result:
    channel: str
    motor_id: int
    motor_type: str
    status: str
    detail: str


def configured_motors() -> list[tuple[int, str]]:
    arm = _load_arm_config(ArmType.YAM)
    gripper_motor_type = GripperType.LINEAR_4310.get_motor_type(ArmType.YAM)
    return [(int(motor_id), str(motor_type)) for motor_id, motor_type in arm.motor_list] + [
        (0x07, gripper_motor_type)
    ]


def probe_channel(channel: str, *, clear_latched_calibration: bool = True) -> list[Result]:
    interface = DMSingleMotorCanInterface(
        channel=channel,
        bustype="socketcan",
        control_mode=ControlMode.MIT,
        name=f"yam-preflight-{channel}",
    )
    results: list[Result] = []
    try:
        for motor_id, motor_type in configured_motors():
            enabled = False
            try:
                feedback = interface.motor_on(
                    motor_id,
                    motor_type,
                    max_retry=2,
                    clear_latched_calibration=clear_latched_calibration,
                )
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


def main(
    channels: Annotated[list[str] | None, tyro.conf.Positional] = None,
    clear_latched_calibration: bool = True,
) -> int:
    """Probe YAM motors without commanding motion.

    Args:
        channels: SocketCAN channels to probe.
        clear_latched_calibration: Clear each output-shaft calibration status
            at most once, then require a fresh normal enable reply with a
            position continuous across the clear (the driver's default).
    """
    if channels is None:
        channels = ["can_left", "can_right"]

    all_results: list[Result] = []
    for channel in channels:
        print(f"\n=== {channel}: enable/read/disable each motor ===")
        try:
            all_results.extend(
                probe_channel(
                    channel,
                    clear_latched_calibration=clear_latched_calibration,
                )
            )
        except Exception as exc:
            all_results.append(Result(channel, -1, "n/a", "CHANNEL_ERROR", str(exc)))

    for result in all_results:
        print(
            f"{result.channel:>10} motor {result.motor_id}: "
            f"{result.status:<42} {result.detail}"
        )

    return 0 if all(result.status == "normal" for result in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main))
