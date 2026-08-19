from __future__ import annotations

import socket
import struct

import pytest

import dca1000_bridge


class _FakeUdpSocket:
    def __init__(self) -> None:
        self.bound = None
        self.timeout = None
        self.closed = False
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.responses: list[tuple[bytes, tuple[str, int]]] = []

    def bind(self, address) -> None:
        self.bound = address

    def settimeout(self, timeout: float) -> None:
        self.timeout = float(timeout)

    def sendto(self, packet: bytes, address) -> int:
        self.sent.append((bytes(packet), address))
        command = struct.unpack("<HHHH", packet)[1]
        # Exercise the production loop that ignores asynchronous status.
        self.responses.append(
            (
                struct.pack(
                    "<HHHH",
                    dca1000_bridge.DCA1000_HEADER,
                    0x0A,
                    0,
                    dca1000_bridge.DCA1000_FOOTER,
                ),
                ("192.168.33.180", 4096),
            )
        )
        self.responses.append(
            (
                struct.pack(
                    "<HHHH",
                    dca1000_bridge.DCA1000_HEADER,
                    command,
                    0,
                    dca1000_bridge.DCA1000_FOOTER,
                ),
                ("192.168.33.180", 4096),
            )
        )
        return len(packet)

    def recvfrom(self, _size: int):
        if not self.responses:
            raise socket.timeout()
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def test_dca1000_command_packet_and_response_are_fixed_little_endian() -> None:
    packet = dca1000_bridge.build_dca1000_command(
        dca1000_bridge.DCA1000_CMD_START_RECORD
    )

    assert packet == bytes.fromhex("5aa505000000aaee")
    assert dca1000_bridge.parse_dca1000_response(packet) == (
        dca1000_bridge.DCA1000_CMD_START_RECORD,
        0,
    )


def test_direct_record_control_binds_static_port_and_handles_start_stop(monkeypatch) -> None:
    fake_socket = _FakeUdpSocket()
    monkeypatch.setattr(dca1000_bridge.socket, "socket", lambda *_args: fake_socket)
    control = dca1000_bridge.DCA1000UdpRecordControl(
        pc_ip="192.168.33.30",
        capture_card_ip="192.168.33.180",
        config_port=4096,
    )

    control.open()
    assert control.start_record() == 0
    assert control.stop_record() == 0
    control.close()

    assert fake_socket.bound == ("192.168.33.30", 4096)
    assert [struct.unpack("<HHHH", item[0])[1] for item in fake_socket.sent] == [
        dca1000_bridge.DCA1000_CMD_SYSTEM_ALIVENESS,
        dca1000_bridge.DCA1000_CMD_START_RECORD,
        dca1000_bridge.DCA1000_CMD_STOP_RECORD,
    ]
    assert all(item[1] == ("192.168.33.180", 4096) for item in fake_socket.sent)
    assert fake_socket.closed


def test_direct_record_control_rejects_nonzero_dca_status(monkeypatch) -> None:
    fake_socket = _FakeUdpSocket()
    monkeypatch.setattr(dca1000_bridge.socket, "socket", lambda *_args: fake_socket)
    control = dca1000_bridge.DCA1000UdpRecordControl(
        pc_ip="192.168.33.30",
        capture_card_ip="192.168.33.180",
        config_port=4096,
    )
    control.open()
    fake_socket.responses.clear()
    fake_socket.responses.append(
        (
            struct.pack(
                "<HHHH",
                dca1000_bridge.DCA1000_HEADER,
                dca1000_bridge.DCA1000_CMD_START_RECORD,
                7,
                dca1000_bridge.DCA1000_FOOTER,
            ),
            ("192.168.33.180", 4096),
        )
    )
    monkeypatch.setattr(fake_socket, "sendto", lambda packet, address: len(packet))

    with pytest.raises(dca1000_bridge.MmwaveStudioError, match="status 0x0007"):
        control.start_record()
    control.close()


def test_bridge_start_and_stop_use_direct_record_control(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class _Control:
        def start_record(self) -> int:
            events.append("start_record")
            return 0

        def stop_record(self) -> int:
            events.append("stop_record")
            return 0

    bridge = dca1000_bridge.MmwaveStudioBridge()
    bridge._hw_connected = True
    bridge._dca_record_control = _Control()  # type: ignore[assignment]
    monkeypatch.setattr(bridge, "start_frame", lambda: events.append("start_frame") or 0)
    monkeypatch.setattr(bridge, "stop_frame", lambda: events.append("stop_frame") or 0)

    bridge.start_streaming(tmp_path / "unused.bin", arm_delay_s=0.0)
    bridge.stop_streaming(stop_delay_s=0.0)

    assert events == ["start_record", "start_frame", "stop_frame", "stop_record"]
    assert not bridge.is_streaming


def test_bridge_still_stops_direct_dca_when_stop_frame_fails(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class _Control:
        def start_record(self) -> int:
            return 0

        def stop_record(self) -> int:
            events.append("stop_record")
            return 0

    bridge = dca1000_bridge.MmwaveStudioBridge()
    bridge._hw_connected = True
    bridge._streaming = True
    bridge._dca_record_control = _Control()  # type: ignore[assignment]

    def _failed_stop_frame() -> int:
        events.append("stop_frame")
        raise dca1000_bridge.MmwaveStudioError("RSTD stop failed")

    monkeypatch.setattr(bridge, "stop_frame", _failed_stop_frame)

    with pytest.raises(dca1000_bridge.MmwaveStudioError, match="RSTD stop failed"):
        bridge.stop_streaming(stop_delay_s=0.0)

    assert events == ["stop_frame", "stop_record"]
