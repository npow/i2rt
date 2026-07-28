from unittest.mock import patch

import can

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, DMSingleMotorCanInterface
from i2rt.motor_drivers.utils import FeedbackFrameInfo, MotorErrorCode, MotorType, ReceiveMode


class _FakeBus:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def test_request_keeps_reading_after_stale_other_motor_feedback():
    """A stale motor-2 reply must not consume motor-1's valid reply."""
    interface = object.__new__(CanInterface)
    interface.bus = _FakeBus()
    interface.channel = "fake"
    interface.name = "fake"
    interface.receive_mode = ReceiveMode.p16
    interface.use_buffered_reader = False
    frames = [
        can.Message(arbitration_id=18, data=bytes([0x12, 0, 0, 0, 0, 0, 0, 0]), is_extended_id=False),
        can.Message(arbitration_id=17, data=bytes([0x11, 0, 0, 0, 0, 0, 0, 0]), is_extended_id=False),
    ]
    interface._receive_message = lambda *args, **kwargs: frames.pop(0) if frames else None

    reply = interface._send_message_get_response(1, 1, [0] * 8, max_retry=1)

    assert reply.arbitration_id == 17
    assert reply.data[0] & 0x0F == 1


def test_startup_retries_disabled_without_clearing_a_fault():
    interface = object.__new__(DMSingleMotorCanInterface)
    commands = []

    def send(_motor_id, data):
        commands.append(data[-1])
        return object()

    interface._send_system_command = send
    statuses = iter([MotorErrorCode.disabled, MotorErrorCode.normal])

    def parse(*_args, **_kwargs):
        code = next(statuses)
        return FeedbackFrameInfo(1, hex(code), "test", 0.0, 0.0, 0.0, 25.0, 25.0)

    interface.parse_recv_message = parse
    interface.clean_error = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not clear disabled"))

    with patch("i2rt.motor_drivers.dm_driver.time.sleep"):
        interface.motor_on(1, MotorType.DM4310)

    assert commands == [0xFC, 0xFC]


def test_startup_recovers_watchdog_and_only_cold_latched_thermal_statuses():
    def run(statuses, *, temperature=25.0):
        interface = object.__new__(DMSingleMotorCanInterface)
        commands = []

        def send(_motor_id, data):
            commands.append(data[-1])
            return object()

        interface._send_system_command = send

        def parse(*args, **kwargs):
            code = statuses.pop(0)
            return FeedbackFrameInfo(
                1,
                hex(code),
                "test",
                0.0,
                0.0,
                0.0,
                temperature,
                temperature,
            )

        interface.parse_recv_message = parse
        try:
            with patch("i2rt.motor_drivers.dm_driver.time.sleep"):
                interface.motor_on(1, MotorType.DM4310)
                outcome = "enabled"
        except RuntimeError:
            outcome = "rejected"
        return outcome, commands

    # First FC reports 0xD.  The one 0xFB reply is consumed transactionally,
    # then the next FC sees enabled; no untracked clear replies remain.
    assert run([MotorErrorCode.loss_communication, MotorErrorCode.disabled, MotorErrorCode.normal]) == (
        "enabled",
        [0xFC, 0xFB, 0xFC],
    )
    # DaMiao's 0xFB Clear Error is valid for thermal faults.  A status at
    # ambient temperature is a historical/latching condition, so clear it
    # once and require a clean subsequent enable reply.
    assert run(
        [
            MotorErrorCode.motor_over_temperature,
            MotorErrorCode.disabled,
            MotorErrorCode.normal,
        ]
    ) == ("enabled", [0xFC, 0xFB, 0xFC])
    assert run(
        [
            MotorErrorCode.mosfet_over_temperature,
            MotorErrorCode.disabled,
            MotorErrorCode.normal,
        ]
    ) == ("enabled", [0xFC, 0xFB, 0xFC])
    # Do not clear a genuinely warm motor, or loop clearing a status that
    # returns after the one allowed fault reset.
    assert run([MotorErrorCode.motor_over_temperature], temperature=61.0) == (
        "rejected",
        [0xFC],
    )
    assert run(
        [
            MotorErrorCode.motor_over_temperature,
            MotorErrorCode.motor_over_temperature,
            MotorErrorCode.motor_over_temperature,
        ]
    ) == ("rejected", [0xFC, 0xFB, 0xFC])


def test_system_command_sends_once_after_drain_with_longer_reply_window():
    interface = object.__new__(DMSingleMotorCanInterface)
    drains = []
    sends = []
    reply = object()
    interface._drain_bus = lambda **kwargs: drains.append(kwargs) or 0
    interface._get_frame_id = lambda motor_id: motor_id

    def send(*args, **kwargs):
        sends.append((args, kwargs))
        return reply

    interface._send_message_get_response = send

    assert interface._send_system_command(1, [0xFF] * 7 + [0xFC]) is reply
    assert drains == [{"timeout_s": 0.003, "idle_count": 1}]
    assert sends == [
        ((1, 1, [0xFF] * 7 + [0xFC]), {"max_retry": 1, "response_timeout": 0.1})
    ]


def test_output_shaft_calibration_fault_is_named_and_never_auto_cleared():
    assert MotorErrorCode.get_error_message(MotorErrorCode.output_shaft_calibration) == (
        "output-shaft encoder calibration error"
    )

    chain = object.__new__(DMChainCanInterface)
    chain.motor_list = [(1, MotorType.DM4310)]

    class _Interface:
        def __init__(self):
            self.enabled = []

        def motor_on(self, motor_id, motor_type):
            self.enabled.append((motor_id, motor_type))

    chain.motor_interface = _Interface()
    feedback = [
        FeedbackFrameInfo(
            1,
            hex(MotorErrorCode.output_shaft_calibration),
            "output-shaft encoder calibration error",
            0.0,
            0.0,
            0.0,
            25.0,
            25.0,
        )
    ]

    assert chain._try_recover_motors(feedback) is False
    assert chain.motor_interface.enabled == []


def test_chain_startup_failure_releases_its_can_interface():
    """A failed constructor must not strand the SocketCAN ownership lock."""

    created = []

    class _FailingInterface:
        def __init__(self, *args, **kwargs):
            self.closed = False
            created.append(self)

        def _drain_bus(self, **kwargs):
            return 0

        def motor_on(self, *args, **kwargs):
            raise RuntimeError("expected startup failure")

        def close(self):
            self.closed = True

    with patch("i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface", _FailingInterface):
        try:
            DMChainCanInterface(
                motor_list=[(1, MotorType.DM4310)],
                motor_offset=[0.0],
                motor_direction=[1.0],
                channel="fake",
                start_thread=False,
            )
        except RuntimeError as exc:
            assert str(exc) == "expected startup failure"
        else:  # pragma: no cover - test assertion
            raise AssertionError("expected DMChainCanInterface startup to fail")

    assert len(created) == 1
    assert created[0].closed is True
