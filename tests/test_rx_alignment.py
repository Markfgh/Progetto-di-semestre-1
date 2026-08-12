"""Regressioni pure per l'allineamento dello stream UDP DCA1000."""

from __future__ import annotations

import radar_app


def test_first_packet_byte_count_recovers_midframe_offset() -> None:
    assert radar_app._rx_frame_offset(0, 16) == 0
    assert radar_app._rx_frame_offset(37, 16) == 5


def test_gap_alignment_counts_every_crossed_frame_without_iterating() -> None:
    # Parte a byte 12, perde 32 byte: completa due frame corrotti e lascia
    # altri 12 byte corrotti nel terzo frame.
    assert radar_app._rx_gap_alignment(12, 32, 16) == (2, 12)
    assert radar_app._rx_gap_alignment(12, 4, 16) == (1, 0)
    assert radar_app._rx_gap_alignment(0, 16_000_005, 16) == (1_000_000, 5)


def test_rx_socket_receive_buffer_config_is_bounded_for_low_latency() -> None:
    mib = 1024 * 1024
    assert radar_app._resolve_rx_socket_rcvbuf_bytes({}) == (16 * mib, 16 * mib)
    assert radar_app._resolve_rx_socket_rcvbuf_bytes({"socket_rcvbuf_bytes": 8 * mib}) == (8 * mib, 8 * mib)
    assert radar_app._resolve_rx_socket_rcvbuf_bytes({"socket_rcvbuf_bytes": "bad"}) == (16 * mib, 16 * mib)
    assert radar_app._resolve_rx_socket_rcvbuf_bytes({"socket_rcvbuf_bytes": 0}) == (0, 1 * mib)
    assert radar_app._resolve_rx_socket_rcvbuf_bytes({"socket_rcvbuf_bytes": 128 * mib}) == (128 * mib, 64 * mib)
