import logging
import xml.etree.ElementTree as ET
from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import (
    ArmType,
    GripperType,
    _load_arm_config,
    combine_arm_and_gripper_xml,
)

logger = logging.getLogger(__name__)


# These are motor-joint endpoints measured at the physical end stops.  The
# left gripper's useful stroke crosses a 2*pi branch, so do not pre-normalize
# either value.  JointMapper intentionally accepts either ordering: command
# 0 maps to the first (closed) endpoint and command 1 maps to the second
# (open) endpoint.
_LEFT_LINEAR_4310_RAW_LIMITS = np.array([6.325055, 1.234264], dtype=float)
# Confirmed against the physical right gripper: raw ~=6.285 is fully closed.
# Policy convention remains 0=closed, 1=open, so this arm has the same raw
# endpoint ordering as the left one even though an earlier calibration label
# had these two values swapped.
_RIGHT_LINEAR_4310_RAW_LIMITS = np.array([6.322004, 1.235790], dtype=float)
_GRIPPER_BRANCH_TOLERANCE_RAD = 0.25


def _calibrated_gripper_limits(channel: str) -> Optional[np.ndarray]:
    """Return a copy of measured raw gripper endpoints for a CAN channel.

    These values live in the same motor coordinate used by ``DMChain`` before
    its software offsets.  Keeping the calibration in that stable coordinate
    makes it independent of which gripper endpoint happens to be present when
    the process starts.
    """
    limits = {
        "can_left": _LEFT_LINEAR_4310_RAW_LIMITS,
        "can_right": _RIGHT_LINEAR_4310_RAW_LIMITS,
    }.get(channel)
    return None if limits is None else limits.copy()


def _startup_wrap_offsets(
    positions: Sequence[float], *, gripper_index: Optional[int]
) -> np.ndarray:
    """Return arm-only +/-2pi startup offset corrections.

    Arm joints are represented near zero after startup, but a gripper's
    calibrated stroke may legitimately include positions above +pi.  Applying
    the arm convention to the gripper makes its calibration depend on whether
    it starts open or closed, so the gripper is deliberately excluded.
    """
    offsets = np.zeros(len(positions), dtype=float)
    for idx, position in enumerate(positions):
        if idx == gripper_index:
            continue
        if position < -np.pi:
            offsets[idx] = -2 * np.pi
        elif position > np.pi:
            offsets[idx] = 2 * np.pi
    return offsets


def _select_gripper_branch_turns(
    position: float,
    limits: Sequence[float],
    *,
    tolerance_rad: float = _GRIPPER_BRANCH_TOLERANCE_RAD,
) -> tuple[int, float]:
    """Choose the equivalent 2*pi motor-coordinate branch for a gripper.

    The DM feedback coordinate can restart on a neighbouring 2*pi branch.
    Select the branch nearest the calibrated stroke midpoint, then reject a
    position that is not plausibly inside the measured stroke.  The returned
    position is in the calibrated motor-joint frame; it is not a command.
    """
    lo, hi = sorted(float(value) for value in limits)
    midpoint = (lo + hi) / 2.0
    turns = int(np.rint((midpoint - position) / (2.0 * np.pi)))
    aligned_position = position + turns * 2.0 * np.pi
    if not lo - tolerance_rad <= aligned_position <= hi + tolerance_rad:
        raise RuntimeError(
            "Gripper feedback is outside its calibrated stroke after 2*pi branch alignment: "
            f"position={position:.4f}, aligned={aligned_position:.4f}, "
            f"limits=[{lo:.4f}, {hi:.4f}]"
        )
    return turns, aligned_position


def _align_gripper_startup_branch(
    motor_chain: DMChainCanInterface,
    gripper_index: int,
    limits: Sequence[float],
) -> tuple[int, float, float]:
    """Align one gripper's software offset with its calibrated 2*pi branch.

    ``absolute_positions`` must remain untouched because the driver updates it
    from live feedback.  Updating the software offset instead keeps both
    feedback and outgoing commands in the same continuous calibrated frame.
    """
    initial_position = float(motor_chain.read_states()[gripper_index].pos)
    turns, aligned_position = _select_gripper_branch_turns(initial_position, limits)
    if turns:
        motor_chain.motor_offset[gripper_index] -= (
            turns * 2.0 * np.pi * motor_chain.motor_direction[gripper_index]
        )
    return turns, initial_position, aligned_position


def _load_joint_limits_from_xml(*xml_paths: str) -> np.ndarray:
    """Parse joint limits (range attributes) from one or more XML files.

    Collects all ``<joint name="jointN" range="lo hi">`` elements across the
    given XML files.  Returns an (N, 2) array of [lower, upper] limits,
    ordered by joint name (joint1, joint2, ...).  Duplicate joint names are
    ignored (first occurrence wins).
    """
    seen: set[str] = set()
    joints: list[tuple[str, float, float]] = []
    for xml_path in xml_paths:
        logger.info(f"Loading joint limits from XML: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for joint_elem in root.iter("joint"):
            name = joint_elem.get("name", "")
            range_str = joint_elem.get("range")
            if range_str and name.startswith("joint") and name not in seen:
                lo, hi = (float(x) for x in range_str.split())
                joints.append((name, lo, hi))
                seen.add(name)

    limits = np.array([[lo, hi] for _, lo, hi in joints])
    logger.info(f"  joint limits ({len(joints)} joints):")
    for name, lo, hi in joints:
        logger.info(f"    {name}: [{lo:.5f}, {hi:.5f}]")
    return limits


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def _get_gripper_only_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.LINEAR_4310,
    sim: bool = False,
    enable_auto_recovery: bool = False,
) -> "Robot":
    """Create a gripper-only robot (no arm).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        gripper_type: Which gripper to load. Must not be NO_GRIPPER.
        sim: If True, return a SimRobot instead of connecting to real hardware.
        enable_auto_recovery: If True, the motor chain tries to clean+re-enable errored motors in its
            control loop instead of failing fast. Defaults to False (fail-fast).
    """
    if gripper_type == GripperType.NO_GRIPPER:
        raise ValueError("gripper_type cannot be NO_GRIPPER when arm_type is NO_ARM")

    xml_path = gripper_type.get_xml_path()
    # One motor drives the gripper; extra XML joints are coupled via equality constraints.
    n_dofs = 1

    nominal_arm = ArmType.YAM
    gripper_limits = gripper_type.get_gripper_limits(nominal_arm)
    gripper_needs_cal = gripper_type.get_gripper_needs_calibration(nominal_arm)

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        sim_gripper_limits = gripper_limits
        if sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        return SimRobot(
            xml_path=xml_path,
            n_dofs=n_dofs,
            gripper_index=0,
            gripper_limits=sim_gripper_limits,
        )

    # --- Real hardware path ---------------------------------------------------
    motor_type = gripper_type.get_motor_type(nominal_arm)
    gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd(nominal_arm)
    direction = gripper_type.get_motor_direction(nominal_arm)

    motor_chain = DMChainCanInterface(
        [[0x07, motor_type]],
        [0.0],
        [direction],
        channel,
        motor_chain_name="gripper_only",
        receive_mode=ReceiveMode.p16,
        start_thread=True,
        enable_auto_recovery=enable_auto_recovery,
    )

    return MotorChainRobot(
        motor_chain=motor_chain,
        xml_path=xml_path,
        use_gravity_comp=False,
        joint_limits=None,
        kp=np.array([gripper_kp]),
        kd=np.array([gripper_kd]),
        gripper_index=0,
        gripper_limits=gripper_limits,
        enable_gripper_calibration=gripper_needs_cal,
        gripper_type=gripper_type,
        arm_type=nominal_arm,
        zero_gravity_mode=False,
    )


def get_yam_robot(
    channel: str = "can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
    gravity_comp_factor: Optional[np.ndarray] = None,
    gripper_limits_override: Optional[np.ndarray] = None,
    gripper_kp: Optional[float] = None,
    gripper_kd: Optional[float] = None,
    sim: bool = False,
    joint_state_saver_factory: Optional[Callable[[], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
    enable_auto_recovery: bool = False,
    use_coulomb_friction: bool = False,
) -> "Robot":
    """Create a YAM-family robot (real or sim).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        arm_type: Which arm variant to use. Use ``ArmType.NO_ARM`` for gripper-only.
        gripper_type: Which gripper (or NO_GRIPPER / YAM_TEACHING_HANDLE).
        zero_gravity_mode: Start in gravity-compensation mode.
        ee_mass: Optional end-effector mass override (kg) for MuJoCo inertial.
        ee_inertia: Optional 10-element inertia override [ipos(3), quat(4), diaginertia(3)].
        gravity_comp_factor: Per-joint array (6 elements, arm joints only) multiplied against gravity torques.
            Overrides the arm-type default when provided.
        gripper_limits_override: Optional [closed, open] limits. If provided, skips calibration.
        gripper_kp: Optional gripper kp override. Defaults to gripper_type's default.
        gripper_kd: Optional gripper kd override. Defaults to gripper_type's default.
        sim: If True, return a SimRobot instead of connecting to real hardware.
        enable_auto_recovery: If True, the motor chain tries to clean+re-enable errored motors in its
            control loop instead of failing fast. Defaults to False (fail-fast).
        use_coulomb_friction: If True, add the per-joint Coulomb friction feedforward (from the arm
            config) during gravity compensation. Defaults to False. Only affects real hardware; ignored
            in sim mode (SimRobot has no friction feedforward).
    """
    # --- Gripper-only path (no arm) -------------------------------------------
    if arm_type == ArmType.NO_ARM:
        return _get_gripper_only_robot(
            channel=channel, gripper_type=gripper_type, sim=sim, enable_auto_recovery=enable_auto_recovery
        )

    with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
    with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE

    # Per-arm calibration measured on the stable USB-CAN interfaces. These are
    # raw, continuous motor-space endpoints [closed, open]; keeping them here
    # prevents startup from re-running the physical calibration.
    if with_gripper and gripper_limits_override is None:
        calibrated = _calibrated_gripper_limits(channel)
        if calibrated is not None:
            gripper_limits_override = calibrated

    hw = _load_arm_config(arm_type)
    effective_gravity_comp = hw.gravity_comp_factor if gravity_comp_factor is None else gravity_comp_factor
    if with_gripper:
        effective_gravity_comp = np.append(effective_gravity_comp, 1.0)

    model_path = combine_arm_and_gripper_xml(
        arm_type,
        gripper_type,
        ee_mass=ee_mass,
        ee_inertia=ee_inertia,
    )

    # Load limits for motor-driven joints only (arm joints + last wrist joint from gripper XML).
    all_joint_limits = _load_joint_limits_from_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())
    n_arm_joints = len(hw.motor_list)
    joint_limits = all_joint_limits[:n_arm_joints]
    joint_limits[:, 0] -= 0.15  # safety buffer
    joint_limits[:, 1] += 0.15

    # Build mutable lists from the frozen arm config, then extend for gripper.
    motor_list = [[can_id, mtype] for can_id, mtype in hw.motor_list]
    directions = list(hw.directions)
    kp = hw.kp.copy()
    kd = hw.kd.copy()
    grav_comp_kd = hw.grav_comp_kd.copy()
    coulomb_friction = hw.coulomb_friction.copy()
    motor_offsets = [0.0] * len(motor_list)

    if with_gripper:
        motor_type = gripper_type.get_motor_type(arm_type)
        default_kp, default_kd = gripper_type.get_motor_kp_kd(arm_type)
        _gripper_kp = gripper_kp if gripper_kp is not None else default_kp
        _gripper_kd = gripper_kd if gripper_kd is not None else default_kd
        logging.info(f"adding gripper motor type={motor_type}, kp={_gripper_kp}, kd={_gripper_kd}")
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        directions.append(gripper_type.get_motor_direction(arm_type))
        kp = np.append(kp, _gripper_kp)
        kd = np.append(kd, _gripper_kd)
        grav_comp_kd = np.append(grav_comp_kd, 0.0)
        coulomb_friction = np.append(coulomb_friction, 0.0)

    if gripper_limits_override is not None and with_gripper:
        gripper_limits = np.asarray(gripper_limits_override)
        gripper_needs_cal = False
    else:
        gripper_limits = gripper_type.get_gripper_limits(arm_type) if with_gripper else None
        gripper_needs_cal = gripper_type.get_gripper_needs_calibration(arm_type) if with_gripper else False

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        # In sim mode, grippers that need calibration have no limits yet — use [0, 1] default.
        sim_gripper_limits = gripper_limits
        if with_gripper and sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        sim_grav_comp = np.ones(len(motor_list))

        return SimRobot(
            xml_path=model_path,
            n_dofs=len(motor_list),
            joint_limits=joint_limits,
            gripper_index=n_arm_joints if with_gripper else None,
            gripper_limits=sim_gripper_limits,
            gravity_comp_factor=sim_grav_comp,
        )

    # --- Real hardware path ---------------------------------------------------

    # Single pass: create chain, read positions, fix wrap-around offsets in-place, then start thread.
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        # DMChain starts its command stream immediately after watchdog-safe
        # bring-up, before this function performs offset bookkeeping.
        start_thread=True,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
        enable_auto_recovery=enable_auto_recovery,
    )
    motor_states = motor_chain.read_states()
    logging.debug(f"motor_states: {motor_states}")

    positions = [state.pos for state in motor_states]
    logging.info(f"current_pos: {positions}")
    gripper_index = n_arm_joints if with_gripper else None
    startup_offsets = _startup_wrap_offsets(positions, gripper_index=gripper_index)
    for idx, offset in enumerate(startup_offsets):
        if offset < 0:
            logging.info(f"motor {idx} pos={positions[idx]:.3f}, offset -2π")
        elif offset > 0:
            logging.info(f"motor {idx} pos={positions[idx]:.3f}, offset +2π")
    motor_chain.motor_offset += startup_offsets

    if gripper_index is not None and gripper_limits is not None:
        turns, initial_gripper_position, aligned_gripper_position = _align_gripper_startup_branch(
            motor_chain,
            gripper_index,
            gripper_limits,
        )
        logging.info(
            "gripper %d startup branch: position=%.3f, turns=%+d, aligned=%.3f",
            gripper_index,
            initial_gripper_position,
            turns,
            aligned_gripper_position,
        )

    logging.info(f"adjusted motor_offsets: {motor_chain.motor_offset.tolist()}")

    # The command stream is already running; corrected offsets are now used
    # by subsequent reads and commands.
    logging.info(f"YAM initial motor_states: {motor_chain.read_states()}")

    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=effective_gravity_comp,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        grav_comp_kd=grav_comp_kd,
        coulomb_friction=coulomb_friction,
        use_coulomb_friction=use_coulomb_friction,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )

    if with_gripper:
        return get_robot(
            gripper_index=n_arm_joints,
            gripper_limits=gripper_limits,
            enable_gripper_calibration=gripper_needs_cal,
            gripper_type=gripper_type,
            arm_type=arm_type,
            limit_gripper_force=50.0,
        )
    return get_robot()
