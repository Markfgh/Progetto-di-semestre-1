from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import multiprocessing as mp
import pytest

import main_refactory
import offline_processing
from offline_processing import OfflineBPRuntime
from shutdown_utils import cleanup_processes, close_queues, process_is_alive


class FakeQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = int(maxsize)
        self.items = []
        self.closed = False
        self.joined = False

    def put_nowait(self, item) -> None:
        if self.closed:
            raise ValueError("queue is closed")
        if self.maxsize > 0 and len(self.items) >= self.maxsize:
            raise queue.Full
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True


class FakeEvent:
    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return bool(self._set)

    def wait(self, timeout=None) -> bool:
        _ = timeout
        return bool(self._set)


class FakeProcessBase:
    def __init__(self, *args, **kwargs) -> None:
        self.daemon = False
        self.started = False
        self.terminated = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return bool(self.started and not self.terminated)

    def join(self, timeout=None) -> None:
        _ = timeout

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True


def _runtime(tmp_path: Path) -> OfflineBPRuntime:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "Config.yaml"
    offline_cfg.write_text("{}", encoding="utf-8")
    fallback_cfg.write_text("{}", encoding="utf-8")
    return OfflineBPRuntime(
        offline_config_path=offline_cfg,
        fallback_capture_cfg=fallback_cfg,
        c_m_s=3.0e8,
        fs_hz=1.0,
        slope_hz_s=1.0,
        fc_hz=77.0e9,
        nfft_range=4,
        range_max_m=1.0,
        crossrange_max_m=1.0,
        image_h=2,
        image_w=2,
    )


def test_cleanup_helpers_are_idempotent() -> None:
    proc = FakeProcessBase()
    proc.start()
    q = FakeQueue()
    q.put_nowait("pending")

    cleanup_processes((proc,), graceful_timeout_s=0.01, terminate_timeout_s=0.2, close_handles=True)
    close_queues((q,))

    cleanup_processes((proc,), graceful_timeout_s=0.01, terminate_timeout_s=0.01, close_handles=True)
    close_queues((q,))

    assert not process_is_alive(proc)
    assert proc.terminated
    assert proc.closed
    assert q.closed
    assert q.joined
    with pytest.raises(ValueError):
        q.put_nowait("closed")


def test_logger_flushes_pending_buffer_on_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_refactory, "BYTES_PER_FRAME", 4)
    monkeypatch.setattr(main_refactory, "_build_capture_file_header", lambda pos_id: b"HDR")

    free_slots: queue.Queue[int] = queue.Queue()
    shm_frames = bytearray(b"ABCD")
    slot_state = mp.Array("i", [1])
    slot_ok = mp.Array("i", [1])
    slot_pos_id = mp.Array("i", [7])
    slot_usemask = mp.Array("i", [2])
    publish_lock = threading.Lock()

    cap_active = mp.Value("i", 0)
    cap_pos_id = mp.Value("i", 7)
    cap_id = mp.Value("I", 0)
    cap_saved = mp.Value("i", 0)
    log_bytes = mp.Value("L", 0)
    stop_evt = threading.Event()

    worker = threading.Thread(
        target=main_refactory.logger_worker,
        kwargs={
            "free_slots": free_slots,
            "shm_frames": shm_frames,
            "slot_state": slot_state,
            "slot_ok": slot_ok,
            "slot_pos_id": slot_pos_id,
            "slot_usemask": slot_usemask,
            "publish_lock": publish_lock,
            "cap_active": cap_active,
            "cap_pos_id": cap_pos_id,
            "cap_id": cap_id,
            "cap_saved": cap_saved,
            "log_bytes": log_bytes,
            "stop_evt": stop_evt,
            "out_dir_s": str(tmp_path),
            "frames_per_position": 2,
            "block_frames": 4,
        },
        daemon=True,
    )
    worker.start()

    with cap_active.get_lock():
        cap_active.value = 1

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        with cap_id.get_lock():
            cap_id.value = (int(cap_id.value) + 1) & 0xFFFFFFFF
        with cap_saved.get_lock():
            if int(cap_saved.value) == 1:
                break
        time.sleep(0.03)
    else:
        stop_evt.set()
        worker.join(timeout=1.0)
        pytest.fail("logger did not copy the test frame")

    stop_evt.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert (tmp_path / "capture_pos7.bin").read_bytes() == b"HDRABCD"
    assert list(free_slots.queue) == [0]


def test_offline_runtime_start_failure_cleans_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_processes: list[FakeProcessBase] = []
    closed_queues: list[FakeQueue] = []

    class FailingSecondProcess(FakeProcessBase):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            fake_processes.append(self)

        def start(self) -> None:
            if len([proc for proc in fake_processes if proc.started]) >= 1:
                raise RuntimeError("spawn boom")
            super().start()

    def recording_close_queues(queues_iter):
        queues = [q for q in queues_iter if q is not None]
        closed_queues.extend(queues)
        return close_queues(queues)

    monkeypatch.setattr(offline_processing.mp, "Queue", FakeQueue)
    monkeypatch.setattr(offline_processing.mp, "Event", FakeEvent)
    monkeypatch.setattr(offline_processing, "Process", FailingSecondProcess)
    monkeypatch.setattr(offline_processing, "close_queues", recording_close_queues)

    runtime = _runtime(tmp_path)

    with pytest.raises(RuntimeError, match="spawn boom"):
        runtime.start(timeout_s=2.0)

    assert len(fake_processes) == 2
    assert fake_processes[0].terminated
    assert all(proc.closed for proc in fake_processes)
    assert closed_queues
    assert all(q.closed and q.joined for q in closed_queues)
    assert runtime._reader_p is None
    assert runtime._dsp_p is None
    assert runtime._reader_to_dsp_q is None
    assert runtime._cmd_q is None
    assert runtime._status_q is None
    assert runtime._stop_evt is None
    runtime.stop()


def test_offline_runtime_start_timeout_closes_processes_and_queues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_processes: list[FakeProcessBase] = []
    closed_queues: list[FakeQueue] = []

    class AliveFakeProcess(FakeProcessBase):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            fake_processes.append(self)

    def recording_close_queues(queues_iter):
        queues = [q for q in queues_iter if q is not None]
        closed_queues.extend(queues)
        return close_queues(queues)

    monkeypatch.setattr(offline_processing.mp, "Queue", FakeQueue)
    monkeypatch.setattr(offline_processing.mp, "Event", FakeEvent)
    monkeypatch.setattr(offline_processing, "Process", AliveFakeProcess)
    monkeypatch.setattr(offline_processing, "close_queues", recording_close_queues)

    runtime = _runtime(tmp_path)
    with pytest.raises(RuntimeError, match="timeout start offline runtime"):
        runtime.start(timeout_s=0.01)

    assert fake_processes
    assert all(proc.terminated for proc in fake_processes)
    assert all(proc.closed for proc in fake_processes)
    assert closed_queues
    assert all(q.closed and q.joined for q in closed_queues)
    assert runtime._reader_p is None
    assert runtime._dsp_p is None
    assert runtime._reader_to_dsp_q is None
    runtime.stop()
