"""Regressioni sul rilascio robusto di processi, code e risorse condivise."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import multiprocessing as mp
import pytest

import radar_app
import offline_processing
from offline_processing import OfflineBPRuntime
from process_cleanup import cleanup_processes, close_queues, process_is_alive


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


def _capture_metadata_store() -> radar_app.CaptureMetadataStore:
    return radar_app.CaptureMetadataStore(
        buffer=mp.RawArray("B", radar_app.CAPTURE_METADATA_BUFFER_BYTES),
        byte_count=mp.Value("I", 0),
        session_id=mp.Value("I", 0),
        lock=mp.Lock(),
    )


def _publish_capture_metadata(
    store: radar_app.CaptureMetadataStore,
    *,
    session_id: int,
    position_id: int,
    position_mm: float,
    position_microsteps: int,
) -> None:
    metadata = radar_app.normalize_capture_metadata(
        position_id,
        {
            "position": position_id,
            "carriage_position_mm": position_mm,
            "carriage_microsteps": position_microsteps,
        },
    )
    radar_app.write_capture_metadata(store, session_id, metadata)


def _runtime(tmp_path: Path) -> OfflineBPRuntime:
    offline_cfg = tmp_path / "offline_config.yaml"
    fallback_cfg = tmp_path / "realtime_config.yaml"
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


def test_cleanup_diagnostics_count_exited_processes_and_survivors() -> None:
    proc = FakeProcessBase()
    proc.start()
    result = cleanup_processes((proc,), graceful_timeout_s=0.0, terminate_timeout_s=0.0)

    assert result == {
        "joined": 1,
        "join_attempts": 2,
        "terminated": 1,
        "survivors": 0,
        "closed": 1,
    }


def test_close_queues_does_not_block_on_stuck_feeder() -> None:
    release = threading.Event()

    class BlockingQueue(FakeQueue):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False

        def join_thread(self) -> None:
            release.wait(timeout=1.0)

        def cancel_join_thread(self) -> None:
            self.cancelled = True
            release.set()

    q = BlockingQueue()
    started = time.perf_counter()
    assert close_queues((q,), join_timeout_s=0.01) == 1
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5
    assert q.cancelled


def test_offline_runtime_fatal_error_is_sticky_across_later_status(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime._status_q = FakeQueue()
    runtime._ready = True
    runtime._status_q.put_nowait({"type": "error", "error": "fatal worker"})
    runtime._status_q.put_nowait({"type": "progress", "phase": "stale progress"})
    runtime._status_q.put_nowait({"type": "ready"})

    runtime._drain_status()

    assert runtime.last_error == "fatal worker"
    # A stale ready message must not make a failed runtime usable again.
    assert runtime.ready is False
    assert runtime.poll_frame() is None


def test_offline_runtime_rejects_started_state_with_dead_worker(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    reader = FakeProcessBase()
    dsp = FakeProcessBase()
    reader.start()
    dsp.start()
    dsp.terminate()
    runtime._started = True
    runtime._reader_p = reader
    runtime._dsp_p = dsp

    with pytest.raises(RuntimeError, match="non vitale: dsp"):
        runtime.start()

    assert runtime.last_error == "offline runtime non vitale: dsp"


def test_offline_runtime_poll_detects_worker_hard_exit(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    reader = FakeProcessBase()
    dsp = FakeProcessBase()
    reader.start()
    dsp.start()
    reader.terminate()
    runtime._started = True
    runtime._ready = True
    runtime._reader_p = reader
    runtime._dsp_p = dsp

    assert runtime.poll_frame() is None
    assert runtime.last_error == "offline runtime non vitale: reader"
    assert runtime.ready is False


def test_logger_flushes_pending_buffer_to_partial_file_on_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radar_app, "BYTES_PER_FRAME", 4)
    monkeypatch.setattr(radar_app, "_build_capture_file_header", lambda pos_id, **_kwargs: b"HDR")

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
    capture_metadata = _capture_metadata_store()
    cap_cancel_id = mp.Value("I", 0)
    cap_done_id = mp.Value("I", 0)
    cap_result = mp.Value("i", 0)
    log_bytes = mp.Value("L", 0)
    stop_evt = threading.Event()
    ready_evt = threading.Event()
    out_dir_shared = mp.Array("u", 1024, lock=True)
    radar_app._write_shared_text(out_dir_shared, str(tmp_path))

    worker = threading.Thread(
        target=radar_app.logger_worker,
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
            "capture_metadata": capture_metadata,
            "cap_cancel_id": cap_cancel_id,
            "cap_done_id": cap_done_id,
            "cap_result": cap_result,
            "log_bytes": log_bytes,
            "stop_evt": stop_evt,
            "out_dir_shared": out_dir_shared,
            "frames_per_position": 2,
            "block_frames": 4,
            "ready_evt": ready_evt,
        },
        daemon=True,
    )
    worker.start()
    assert ready_evt.wait(1.0)

    _publish_capture_metadata(
        capture_metadata,
        session_id=1,
        position_id=7,
        position_mm=12.5,
        position_microsteps=1000,
    )

    with cap_active.get_lock():
        cap_active.value = 1
    with cap_id.get_lock():
        cap_id.value = 1

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
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
    assert not (tmp_path / "capture_pos7.bin").exists()
    assert (tmp_path / "capture_pos7.bin.part").read_bytes() == b"HDRABCD"
    assert list(free_slots.queue) == [0]


def test_logger_completion_releases_surplus_capture_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame gia' taggati oltre il target non devono consumare il ring."""
    monkeypatch.setattr(radar_app, "BYTES_PER_FRAME", 4)
    monkeypatch.setattr(radar_app, "_build_capture_file_header", lambda pos_id, **_kwargs: b"HDR")

    free_slots: queue.Queue[int] = queue.Queue()
    shm_frames = bytearray(b"ABCDEFGH")
    slot_state = mp.Array("i", [1, 1])
    slot_ok = mp.Array("i", [1, 1])
    slot_pos_id = mp.Array("i", [7, 7])
    slot_usemask = mp.Array("i", [2, 2])
    publish_lock = threading.Lock()
    cap_active = mp.Value("i", 0)
    cap_pos_id = mp.Value("i", 7)
    cap_id = mp.Value("I", 0)
    cap_saved = mp.Value("i", 0)
    capture_metadata = _capture_metadata_store()
    cap_cancel_id = mp.Value("I", 0)
    cap_done_id = mp.Value("I", 0)
    cap_result = mp.Value("i", 0)
    log_bytes = mp.Value("L", 0)
    stop_evt = threading.Event()
    ready_evt = threading.Event()
    out_dir_shared = mp.Array("u", 1024, lock=True)
    radar_app._write_shared_text(out_dir_shared, str(tmp_path))

    worker = threading.Thread(
        target=radar_app.logger_worker,
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
            "capture_metadata": capture_metadata,
            "cap_cancel_id": cap_cancel_id,
            "cap_done_id": cap_done_id,
            "cap_result": cap_result,
            "log_bytes": log_bytes,
            "stop_evt": stop_evt,
            "out_dir_shared": out_dir_shared,
            "frames_per_position": 1,
            "block_frames": 4,
            "ready_evt": ready_evt,
        },
        daemon=True,
    )
    worker.start()
    assert ready_evt.wait(1.0)
    _publish_capture_metadata(
        capture_metadata,
        session_id=1,
        position_id=7,
        position_mm=1.0,
        position_microsteps=10,
    )
    with cap_active.get_lock():
        cap_active.value = 1
    with cap_id.get_lock():
        cap_id.value = 1

    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and int(cap_done_id.value) != 1:
        time.sleep(0.005)
    stop_evt.set()
    worker.join(timeout=1.0)

    assert int(cap_done_id.value) == 1
    assert int(cap_result.value) == 1
    assert int(cap_active.value) == 0
    assert list(slot_usemask) == [0, 0]
    assert list(slot_state) == [0, 0]
    released = list(free_slots.queue)
    assert len(released) == 2
    assert set(released) == {0, 1}


def test_logger_reports_flush_failure_as_capture_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(radar_app, "BYTES_PER_FRAME", 4)
    monkeypatch.setattr(radar_app, "_build_capture_file_header", lambda pos_id, **_kwargs: b"HDR")

    class FlushFailingFile:
        def write(self, payload) -> int:
            return len(payload)

        def flush(self) -> None:
            raise OSError("flush failed")

        def close(self) -> None:
            return None

    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: FlushFailingFile())

    free_slots: queue.Queue[int] = queue.Queue()
    slot_state = mp.Array("i", [1])
    slot_ok = mp.Array("i", [1])
    slot_pos_id = mp.Array("i", [9])
    slot_usemask = mp.Array("i", [2])
    cap_active = mp.Value("i", 0)
    cap_pos_id = mp.Value("i", 9)
    cap_id = mp.Value("I", 0)
    cap_saved = mp.Value("i", 0)
    capture_metadata = _capture_metadata_store()
    cap_cancel_id = mp.Value("I", 0)
    cap_done_id = mp.Value("I", 0)
    cap_result = mp.Value("i", 0)
    stop_evt = threading.Event()
    ready_evt = threading.Event()
    out_dir_shared = mp.Array("u", 1024, lock=True)
    radar_app._write_shared_text(out_dir_shared, str(tmp_path))

    worker = threading.Thread(
        target=radar_app.logger_worker,
        kwargs={
            "free_slots": free_slots,
            "shm_frames": bytearray(b"ABCD"),
            "slot_state": slot_state,
            "slot_ok": slot_ok,
            "slot_pos_id": slot_pos_id,
            "slot_usemask": slot_usemask,
            "publish_lock": threading.Lock(),
            "cap_active": cap_active,
            "cap_pos_id": cap_pos_id,
            "cap_id": cap_id,
            "cap_saved": cap_saved,
            "capture_metadata": capture_metadata,
            "cap_cancel_id": cap_cancel_id,
            "cap_done_id": cap_done_id,
            "cap_result": cap_result,
            "log_bytes": mp.Value("L", 0),
            "stop_evt": stop_evt,
            "out_dir_shared": out_dir_shared,
            "frames_per_position": 1,
            "block_frames": 4,
            "ready_evt": ready_evt,
        },
        daemon=True,
    )
    worker.start()
    assert ready_evt.wait(1.0)
    _publish_capture_metadata(
        capture_metadata,
        session_id=1,
        position_id=9,
        position_mm=2.0,
        position_microsteps=20,
    )
    cap_active.value = 1
    cap_id.value = 1
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline and int(cap_done_id.value) != 1:
        time.sleep(0.005)
    stop_evt.set()
    worker.join(timeout=1.0)

    assert int(cap_done_id.value) == 1
    assert int(cap_result.value) == -1


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


def test_offline_runtime_detects_reader_hard_exit_without_waiting_for_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_processes: list[FakeProcessBase] = []

    class ReaderDiesProcess(FakeProcessBase):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.process_index = len(fake_processes)
            fake_processes.append(self)

        def start(self) -> None:
            super().start()
            if self.process_index == 0:
                self.terminate()

    monkeypatch.setattr(offline_processing.mp, "Queue", FakeQueue)
    monkeypatch.setattr(offline_processing.mp, "Event", FakeEvent)
    monkeypatch.setattr(offline_processing, "Process", ReaderDiesProcess)

    runtime = _runtime(tmp_path)
    started_at = time.perf_counter()
    with pytest.raises(RuntimeError, match="terminato durante start: reader"):
        runtime.start(timeout_s=5.0)

    assert time.perf_counter() - started_at < 1.0
    assert runtime._reader_p is None
    assert runtime._dsp_p is None
