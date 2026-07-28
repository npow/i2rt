"""Pure unit tests for the linear-gripper contact limiter."""

from unittest.mock import patch

import pytest

from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType


LEFT_CLOSED = 6.325055
LEFT_OPEN = 1.234264


def _raw_qpos(normalized_qpos: float) -> float:
    return LEFT_CLOSED + normalized_qpos * (LEFT_OPEN - LEFT_CLOSED)


def _state(
    *,
    current_normalized: float,
    target_normalized: float,
    effort: float,
    last_command_normalized: float | None = None,
) -> dict[str, float]:
    if last_command_normalized is None:
        last_command_normalized = current_normalized
    return {
        "target_qpos": _raw_qpos(target_normalized),
        "current_qpos": _raw_qpos(current_normalized),
        "current_qvel": 0.0,
        "current_eff": effort,
        "current_normalized_qpos": current_normalized,
        "target_normalized_qpos": target_normalized,
        "last_command_qpos": _raw_qpos(last_command_normalized),
    }


def _limiter() -> GripperForceLimiter:
    return GripperForceLimiter(
        max_force=50.0,
        gripper_type=GripperType.LINEAR_4310,
        arm_type=ArmType.YAM,
        kp=20.0,
    )


def test_open_endpoint_high_effort_releases_before_latching_close() -> None:
    """A manually/freshly opened gripper can begin a close despite open-stop effort."""
    limiter = _limiter()
    state = _state(current_normalized=0.984, target_normalized=0.228, effort=1.0)

    command = limiter.update(state)

    command_normalized = (command - LEFT_CLOSED) / (LEFT_OPEN - LEFT_CLOSED)
    # The endpoint gate never jumps directly to 0.228; it makes a 3%-stroke
    # retreat from 0.984, which starts closing without discarding the limiter.
    assert command_normalized == pytest.approx(0.954, abs=1e-6)
    assert limiter._is_clogged is False

    # Once feedback is off the open stop and the high effort is gone, the
    # original close target reaches the controller normally.
    off_stop = _state(current_normalized=0.949, target_normalized=0.228, effort=0.0)
    assert limiter.update(off_stop) == pytest.approx(_raw_qpos(0.228))
    assert limiter._is_clogged is False


def test_reversing_away_from_a_latched_direction_releases_the_limiter() -> None:
    """A contact latch must not survive a request to move away from that contact."""
    limiter = _limiter()
    blocked_close = _state(current_normalized=0.50, target_normalized=0.20, effort=1.0)
    limiter.update(blocked_close)
    assert limiter._is_clogged is True
    assert limiter._blocked_direction == 1.0

    retreat = _state(current_normalized=0.50, target_normalized=0.80, effort=1.0)
    assert limiter.update(retreat) == pytest.approx(_raw_qpos(0.80))
    assert limiter._is_clogged is False
    assert limiter._blocked_direction is None


@pytest.mark.parametrize("target_normalized", [0.50, 0.80])
def test_high_effort_does_not_latch_while_holding_or_opening(target_normalized: float) -> None:
    """Only a close request may create a new object-contact latch."""
    limiter = _limiter()
    state = _state(current_normalized=0.50, target_normalized=target_normalized, effort=1.0)

    assert limiter.update(state) == pytest.approx(_raw_qpos(target_normalized))
    assert limiter._is_clogged is False
    assert limiter._blocked_direction is None


def test_open_endpoint_release_is_time_bounded_if_feedback_does_not_move() -> None:
    """A stuck endpoint returns to ordinary force limiting after 250 ms."""
    limiter = _limiter()
    state = _state(current_normalized=0.984, target_normalized=0.228, effort=1.0)

    with patch("i2rt.robots.utils.time.monotonic", side_effect=[100.0, 100.251]):
        limiter.update(state)
        timed_out_command = limiter.update(state)

    assert limiter._is_clogged is True
    assert timed_out_command != pytest.approx(_raw_qpos(0.228))


def test_continued_closing_against_an_object_remains_force_limited() -> None:
    """The endpoint release does not disable normal object-contact protection."""
    limiter = _limiter()
    closing = _state(current_normalized=0.50, target_normalized=0.20, effort=1.0)

    first_command = limiter.update(closing)
    second_command = limiter.update(closing)

    assert limiter._is_clogged is True
    assert limiter._blocked_direction == 1.0
    assert first_command != pytest.approx(_raw_qpos(0.20))
    assert second_command != pytest.approx(_raw_qpos(0.20))
