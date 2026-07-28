import logging
import os
import time
from typing import List, Optional

import can

try:
    import fcntl
except ImportError:  # pragma: no cover - SocketCAN is Linux-only in production.
    fcntl = None

from i2rt.motor_drivers.utils import ReceiveMode


class CanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        use_buffered_reader: bool = False,
    ):
        self.channel = channel
        # A CAN socket does not arbitrate ownership at the application level:
        # two viewers/controllers can both consume replies and send commands.
        # Refuse a second controller for the same SocketCAN interface.
        self._channel_lock_fd: Optional[int] = None
        if bustype == "socketcan" and fcntl is not None:
            lock_name = channel.replace("/", "_")
            lock_path = f"/tmp/i2rt-can-{lock_name}.lock"
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(lock_fd)
                raise RuntimeError(
                    f"CAN channel {channel!r} is already owned by another i2rt process; "
                    "stop the other controller/viewer before opening it."
                ) from None
            self._channel_lock_fd = lock_fd
        try:
            self.bus = can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)
        except Exception:
            if self._channel_lock_fd is not None:
                fcntl.flock(self._channel_lock_fd, fcntl.LOCK_UN)
                os.close(self._channel_lock_fd)
                self._channel_lock_fd = None
            raise
        self.busstate = self.bus.state
        self.name = name
        self.receive_mode = receive_mode
        self.use_buffered_reader = use_buffered_reader
        logging.info(f"Can interface {self.name} use_buffered_reader: {use_buffered_reader}")
        if use_buffered_reader:
            # Initialize BufferedReader for asynchronous message handling
            self.buffered_reader = can.BufferedReader()
            self.notifier = can.Notifier(self.bus, [self.buffered_reader])

    def close(self) -> None:
        """Shut down the CAN bus."""
        try:
            if self.use_buffered_reader:
                self.notifier.stop()
            self.bus.shutdown()
        finally:
            if self._channel_lock_fd is not None:
                fcntl.flock(self._channel_lock_fd, fcntl.LOCK_UN)
                os.close(self._channel_lock_fd)
                self._channel_lock_fd = None

    def _is_expected_feedback(self, response: can.Message, expected_id: int, motor_id: int) -> bool:
        """Return true only for a valid feedback frame from ``motor_id``.

        On a multi-motor CAN bus stale feedback and register replies are
        common.  Matching only arbitration ID is insufficient: the low nibble
        of feedback byte 0 is the motor ID in the DaMiao MIT protocol.
        """
        if response.is_error_frame or response.is_remote_frame:
            return False
        if response.arbitration_id != expected_id or len(response.data) != 8:
            return False
        if self.receive_mode == ReceiveMode.p16 and (response.data[0] & 0x0F) != motor_id:
            return False
        return True

    def _send_message_get_response(
        self,
        id: int,
        motor_id: int,
        data: List[int],
        max_retry: int = 5,
        expected_id: Optional[int] = None,
        response_timeout: float = 0.01,
    ) -> can.Message:
        """Send a message over the CAN bus.

        Args:
            id (int): The arbitration ID of the message.
            data (List[int]): The data payload of the message.

        Returns:
            can.Message: The message that was sent.
        """
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        if expected_id is None:
            expected_id = self.receive_mode.get_receive_id(motor_id)
        for _ in range(max_retry):
            try:
                self.bus.send(message)
                # Keep consuming frames until this transaction's deadline. Do
                # not discard the valid reply merely because a stale frame
                # arrived first.
                deadline = time.monotonic() + response_timeout
                while time.monotonic() < deadline:
                    response = self._receive_message(
                        motor_id,
                        timeout=min(0.001, max(0.0, deadline - time.monotonic())),
                        supress_warning=True,
                    )
                    if response is None:
                        continue
                    if self._is_expected_feedback(response, expected_id, motor_id):
                        return response
                    logging.debug(
                        "Ignoring non-matching CAN frame on %s while waiting for motor %s: "
                        "arb=0x%x data=%s",
                        self.channel,
                        motor_id,
                        response.arbitration_id,
                        bytes(response.data).hex(),
                    )
            except can.CanError as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {id} over can bus. Retrying..."
                    + "\033[0m"
                )
            time.sleep(0.001)
        raise AssertionError(
            f"fail to communicate with the motor {id} on {self.name} at can channel {self.bus.channel_info}"
        )

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message, or None if no message is received.
        """
        try:
            return self._receive_message(motor_id, timeout, supress_warning=True)
        except AssertionError:
            return None

    def _drain_bus(self, timeout_s: float = 0.05, idle_count: int = 10) -> int:
        """Drain pending CAN frames until the bus is idle.

        Loops `try_receive_message(timeout=0.001)` until either `idle_count`
        consecutive 1 ms reads return None or `timeout_s` wall-clock has
        elapsed. Used at init handovers (e.g. between encoder validation and
        motor bring-up) to flush stale frames that would otherwise be misread
        as the next motor's reply. Returns the number of frames consumed.
        """
        drained = 0
        idle = 0
        deadline = time.time() + timeout_s
        while time.time() < deadline and idle < idle_count:
            if self.try_receive_message(timeout=0.001) is None:
                idle += 1
            else:
                idle = 0
                drained += 1
        return drained

    def _receive_message(
        self, motor_id: Optional[int] = None, timeout: float = 0.009, supress_warning: bool = False
    ) -> Optional[can.Message]:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message.

        Raises:
            AssertionError: If no message is received within the timeout.
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.use_buffered_reader:
                message = self.buffered_reader.get_message(timeout=0.001)
            else:
                message = self.bus.recv(timeout=0.001)
            if message:
                return message
        if not supress_warning:
            logging.warning(
                "\033[91m"
                + f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
                + "\033[0m"
            )
