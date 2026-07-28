"""Regression tests for calibrated linear-gripper coordinate frames."""

import numpy as np

from i2rt.robots.get_robot import (
    _calibrated_gripper_limits,
    _select_gripper_branch_turns,
    _startup_wrap_offsets,
)
from i2rt.robots.utils import JointMapper


def test_left_gripper_raw_endpoints_keep_policy_aperture_semantics() -> None:
    limits = _calibrated_gripper_limits("can_left")
    assert limits is not None
    np.testing.assert_allclose(limits, [6.325055, 1.234264])

    mapper = JointMapper({6: limits}, total_dofs=7)
    for normalized, raw in ((0.0, limits[0]), (0.25, 5.05235725), (0.5, 3.7796595), (0.75, 2.50696175), (1.0, limits[1])):
        command = np.zeros(7)
        command[6] = normalized
        np.testing.assert_allclose(mapper.to_robot_joint_pos_space(command)[6], raw, atol=1e-6)

        feedback = np.zeros(7)
        feedback[6] = raw
        np.testing.assert_allclose(mapper.to_command_joint_pos_space(feedback)[6], normalized, atol=1e-6)


def test_startup_wrap_keeps_gripper_in_its_continuous_raw_frame() -> None:
    offsets = _startup_wrap_offsets([-3.4, 3.5, 6.325055], gripper_index=2)
    np.testing.assert_allclose(offsets, [-2 * np.pi, 2 * np.pi, 0.0])


def test_gripper_startup_selects_the_calibrated_2pi_branch() -> None:
    """A feedback restart near zero must map to the left closed endpoint."""
    turns, aligned = _select_gripper_branch_turns(0.039864, [6.325055, 1.234264])

    assert turns == 1
    np.testing.assert_allclose(aligned, 6.323049, atol=1e-6)


def test_right_gripper_calibration_matches_measured_endpoints() -> None:
    limits = _calibrated_gripper_limits("can_right")
    assert limits is not None
    np.testing.assert_allclose(limits, [6.322004, 1.235790], atol=1e-6)

    mapper = JointMapper({6: limits}, total_dofs=7)
    command = np.zeros(7)
    command[6] = 0.0
    np.testing.assert_allclose(mapper.to_robot_joint_pos_space(command)[6], limits[0], atol=1e-6)
    command[6] = 1.0
    np.testing.assert_allclose(mapper.to_robot_joint_pos_space(command)[6], limits[1], atol=1e-6)
