"""Benchmark riproducibili delle parti più costose della pipeline realtime.

Il file confronta implementazioni Python/Numba e misura le FFT su forme di
frame coerenti con la configurazione di cattura, senza usare hardware.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import scipy.fft as fft

import realtime_dsp


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _shape_cfg(cfg: dict[str, Any]) -> dict[str, int]:
    """Deriva forme coerenti per benchmark da una configurazione di cattura."""
    capture = cfg.get("capture", {}) or {}
    fft_cfg = cfg.get("fft", {}) or {}
    dsp_cfg = cfg.get("dsp", {}) or {}
    samples = int(capture.get("samples", 256))
    chirps = int(capture.get("chirps", 256))
    tx = int(capture.get("tx", 2))
    rx = int(capture.get("rx", 4))
    x_frames = int(capture.get("x_frames", 1))
    nfft_range = int(fft_cfg.get("nfft_range", 256))
    nfft_angle = int(fft_cfg.get("nfft_angle", 128))
    workers = int(dsp_cfg.get("fft_workers", fft_cfg.get("workers", 1)))
    # Nel TDM-MIMO i chirp sono alternati tra trasmettitori: la FFT Doppler vede
    # quindi i soli loop per TX, mentre l'array virtuale combina TX * RX.
    loops = max(1, chirps // max(1, tx))
    return {
        "samples": samples,
        "chirps": chirps,
        "loops": loops,
        "tx": tx,
        "rx": rx,
        "x_frames": x_frames,
        "nfft_range": nfft_range,
        "nfft_angle": nfft_angle,
        "workers": max(1, workers),
        "virtual_ant": max(1, tx * rx),
        "processing_bins": max(1, nfft_range // 2),
    }


def _median_ms(fn, *, repeats: int) -> float:
    """Misura più volte una funzione e usa la mediana contro jitter e outlier."""
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _candidate_mask(power: np.ndarray, threshold_map: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        power_db = realtime_dsp.compute_detection_power_db_map(power)
    return np.isfinite(power_db) & np.isfinite(threshold_map) & (power_db >= threshold_map)


def benchmark_ca_cfar(*, repeats: int = 10) -> None:
    """Confronta CA-CFAR Python e Numba verificandone anche l'equivalenza."""
    # Il seme fisso fa ricevere ai due backend esattamente le stesse mappe di potenza.
    rng = np.random.default_rng(9955)
    cases = [
        ("static_range_angle", rng.lognormal(2.0, 0.9, size=(128, 128)).astype(np.float32), (8, 2, 8, 2)),
        ("moving_range_doppler", rng.lognormal(2.0, 0.9, size=(128, 128)).astype(np.float32), (8, 2, 4, 1)),
    ]
    print("[BENCH] CA-CFAR Python vs optional Numba JIT")
    status = realtime_dsp.cfar_numba_runtime_status()
    if not status["available"]:
        print("[BENCH] Numba unavailable; only Python baseline will run.")
    else:
        # Il warmup separa compilazione JIT e verifica iniziale dai campioni temporali.
        realtime_dsp.configure_cfar_numba_runtime(
            realtime_dsp.CfarNumbaConfig(enabled=True, warmup_on_start=True, self_check_on_start=True),
            log=True,
        )

    for name, power, params in cases:
        train_r, guard_r, train_c, guard_c = params
        kwargs = dict(
            threshold_mode="ca_cfar",
            train_range_bins=train_r,
            guard_range_bins=guard_r,
            train_col_bins=train_c,
            guard_col_bins=guard_c,
            threshold_offset_db=12.0,
            min_power_db=8.0,
            os_cfar_rank=0,
        )

        py_map = realtime_dsp._compute_cfar_threshold_db_map_python(power, **kwargs)
        py_mask = _candidate_mask(power, py_map)
        py_ms = _median_ms(lambda: realtime_dsp._compute_cfar_threshold_db_map_python(power, **kwargs), repeats=repeats)
        print(f"[BENCH] {name} python_ms={py_ms:.3f} candidates={int(np.count_nonzero(py_mask))}")

        if realtime_dsp.cfar_numba_runtime_status()["enabled"]:
            jit_map = realtime_dsp.compute_cfar_threshold_db_map(power, **kwargs)
            jit_mask = _candidate_mask(power, jit_map)
            # Piccole differenze float sono ammesse nella mappa; la maschera deve invece
            # restare identica, per garantire che le decisioni CFAR non cambino.
            np.testing.assert_allclose(py_map, jit_map, rtol=2e-5, atol=2e-4, equal_nan=True)
            if not np.array_equal(py_mask, jit_mask):
                raise AssertionError(f"{name}: candidate_mask mismatch")
            jit_ms = _median_ms(lambda: realtime_dsp.compute_cfar_threshold_db_map(power, **kwargs), repeats=repeats)
            speedup = py_ms / max(jit_ms, 1e-9)
            print(f"[BENCH] {name} numba_ms={jit_ms:.3f} speedup={speedup:.2f}x")


def benchmark_fft(shape: dict[str, int], *, repeats: int = 10) -> None:
    """Misura le tre FFT con le forme effettive della pipeline radar."""
    rng = np.random.default_rng(1234)
    x_frames = shape["x_frames"]
    loops = shape["loops"]
    tx = shape["tx"]
    rx = shape["rx"]
    samples = shape["samples"]
    nfft_range = shape["nfft_range"]
    nfft_angle = shape["nfft_angle"]
    workers = shape["workers"]
    bins = shape["processing_bins"]
    virtual_ant = shape["virtual_ant"]

    # Le forme e gli assi riproducono la pipeline: range sull'asse sample (3),
    # Doppler sui loop (1) e angolo sulle antenne virtuali (3).
    # I buffer sono creati fuori dalla misura per cronometrare le FFT, non allocazione o RNG.
    range_in = (rng.standard_normal((x_frames, loops, tx, samples, rx)) + 1j * rng.standard_normal((x_frames, loops, tx, samples, rx))).astype(np.complex64)
    doppler_in = (rng.standard_normal((x_frames, loops, tx, bins, rx)) + 1j * rng.standard_normal((x_frames, loops, tx, bins, rx))).astype(np.complex64)
    angle_in = (rng.standard_normal((x_frames, loops, bins, virtual_ant)) + 1j * rng.standard_normal((x_frames, loops, bins, virtual_ant))).astype(np.complex64)

    print("[BENCH] FFT microbenchmarks using scipy.fft global backend")
    print(f"[BENCH] shape={shape}")
    # overwrite_x=False rende ogni ripetizione equivalente: l'input sintetico non viene
    # trasformato in-place dalla FFT precedente.
    range_ms = _median_ms(lambda: fft.fft(range_in, n=nfft_range, axis=3, workers=workers, overwrite_x=False), repeats=repeats)
    doppler_ms = _median_ms(lambda: fft.fft(doppler_in, n=loops, axis=1, workers=workers, overwrite_x=False), repeats=repeats)
    angle_ms = _median_ms(lambda: fft.ifft(angle_in, n=nfft_angle, axis=3, workers=workers, overwrite_x=False), repeats=repeats)
    print(f"[BENCH] range_fft_ms={range_ms:.3f} workers={workers}")
    print(f"[BENCH] doppler_fft_ms={doppler_ms:.3f} workers={workers}")
    print(f"[BENCH] angle_ifft_ms={angle_ms:.3f} workers={workers}")


def _geometry() -> realtime_dsp.VirtualArrayGeometry:
    return realtime_dsp.VirtualArrayGeometry(
        order_flat=np.arange(8, dtype=np.int32),
        phase_centers_lambda=np.arange(8, dtype=np.float32) * np.float32(0.25),
        identity_order=True,
        uniform_half_lambda=False,
        uniform_spacing_lambda=0.25,
        angle_axis_sign=1.0,
        angle_u_to_sin_scale=2.0,
    )


def _snapshot(angle_deg: float) -> np.ndarray:
    geometry = _geometry()
    u_target = np.float32(2.0 * np.sin(np.deg2rad(np.float32(angle_deg))))
    return np.exp((-1j * np.float32(2.0 * np.pi)) * geometry.phase_centers_lambda * u_target).astype(np.complex64)


def _dsp_cfg(samples: int, loops: int, workers: int) -> realtime_dsp.RealtimeDSPConfig:
    return realtime_dsp.RealtimeDSPConfig(
        c=1.0,
        fs=1.0,
        slope=0.15625,
        samples=samples,
        chirps=loops * 2,
        rx=4,
        tx=2,
        x_frames=1,
        bytes_per_frame=0,
        nfft_range=samples,
        nfft_angle=256,
        range_max_display=2.0,
        range_profile_count=8,
        virtual_ant=8,
        fft_workers=workers,
        debug_stats=False,
    )


def _run_synthetic_process_buffer(*, use_numba: bool, workers: int) -> list[realtime_dsp.Detection]:
    """Esegue un frame sintetico end-to-end per controllare che Numba non cambi l'esito."""
    samples = 32
    loops = 8
    n_frames = 1
    range_bin = 9
    angle_deg = 25.0
    dsp_cfg = _dsp_cfg(samples=samples, loops=loops, workers=workers)
    geometry = _geometry()
    angle_axis = realtime_dsp.build_angle_axis_deg(dsp_cfg.nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(dsp_cfg.virtual_ant, dsp_cfg.nfft_angle, geometry=geometry)

    sample_phase = np.exp(1j * 2.0 * np.pi * range_bin * np.arange(samples, dtype=np.float32) / samples)
    snapshot = _snapshot(angle_deg).reshape(2, 4)
    raw = (
        sample_phase.reshape(1, 1, 1, samples, 1)
        * snapshot.reshape(1, 1, 2, 1, 4)
        * np.complex64(30.0 + 0.0j)
    )
    # Inseriamo un solo bersaglio deterministico nel layout reale
    # [frame, loop, tx, sample, rx], così il confronto osserva anche range e angolo.
    raw = np.tile(raw, (n_frames, loops, 1, 1, 1)).astype(np.complex64, copy=False).reshape(-1)

    # Le due esecuzioni differiscono solo nel backend CFAR; configurazione, segnale e
    # buffer di pubblicazione rimangono identici per rendere significativo il confronto.
    if use_numba:
        realtime_dsp.configure_cfar_numba_runtime(
            realtime_dsp.CfarNumbaConfig(enabled=True, warmup_on_start=True, self_check_on_start=True),
            log=False,
        )
    else:
        realtime_dsp.configure_cfar_numba_runtime(realtime_dsp.CfarNumbaConfig(enabled=False), log=False)

    gui_h = 32
    gui_w = 32
    gui_heat_views = (
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
    )
    gui_profile_views = (
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
        np.full(dsp_cfg.range_profile_count * samples, -120.0, dtype=np.float32),
    )
    profiles_out = np.empty((dsp_cfg.range_profile_count, samples), dtype=np.float32)

    # process_buffer usa FFT in-place: la copia isola l'input del benchmark da tali
    # trasformazioni e conserva il segnale sintetico di riferimento.
    _, detections = realtime_dsp.process_buffer(
        raw.copy(),
        n_frames,
        np.ones((1, 1, 1, samples, 1), dtype=np.float32),
        np.ones((1, loops, 1, 1, 1), dtype=np.float32),
        np.ones((1, 1, 1, dsp_cfg.virtual_ant), dtype=np.float32),
        False,
        False,
        False,
        realtime_dsp.MeanSelection(enabled=False),
        realtime_dsp.PostRangeFftFilterConfig(),
        realtime_dsp.PostRangeFftFilterConfig(),
        realtime_dsp.PostRangeFftFilterConfig(),
        False,
        False,
        realtime_dsp.AngleProcessingConfig(mode="bartlett"),
        realtime_dsp.HeatmapEMAConfig(enabled=False),
        realtime_dsp.HeatmapSpatialFilterConfig(enabled=False),
        realtime_dsp.DisplayProjectionConfig(projection_mode="cartesian", projection_interp="nearest"),
        geometry,
        steering,
        angle_axis,
        None,
        2.0,
        1.5,
        None,
        None,
        realtime_dsp.DetectionConfigStatic(
            enabled=True,
            threshold_mode="ca_cfar",
            localmax_range_bins=1,
            localmax_angle_bins=3,
            min_power_db=-120.0,
            max_detections=4,
            cfar_train_range_bins=1,
            cfar_guard_range_bins=1,
            cfar_train_col_bins=2,
            cfar_guard_col_bins=1,
            cfar_threshold_db=6.0,
        ),
        realtime_dsp.DetectionConfigMoving(enabled=False),
        realtime_dsp.FusionConfig(enabled=True),
        realtime_dsp.BackgroundSubtractionState(),
        realtime_dsp.BackgroundSubtractionState(),
        realtime_dsp.BackgroundSubtractionState(),
        None,
        None,
        None,
        None,
        None,
        None,
        gui_h,
        gui_w,
        gui_heat_views,
        gui_profile_views,
        mp.Value("i", 0),
        mp.Value("i", 0),
        threading.Lock(),
        True,
        profiles_out,
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        mp.Value("d", 0.0),
        dsp_cfg,
        20,
        20,
    )
    return detections


def _assert_detection_lists_equivalent(a: list[realtime_dsp.Detection], b: list[realtime_dsp.Detection]) -> None:
    # Prima confrontiamo le decisioni discrete, poi le grandezze continue con una
    # tolleranza numerica: accelerare non deve spostare o aggiungere detection.
    if len(a) != len(b):
        raise AssertionError(f"detection count changed: {len(a)} != {len(b)}")
    for idx, (left, right) in enumerate(zip(a, b)):
        scalar_fields = ["range_bin", "angle_bin", "doppler_bin", "source"]
        for field in scalar_fields:
            if getattr(left, field) != getattr(right, field):
                raise AssertionError(f"detection {idx} field {field} changed: {getattr(left, field)} != {getattr(right, field)}")
        float_fields = ["range_m", "angle_deg", "x_m", "y_m", "power_db"]
        for field in float_fields:
            np.testing.assert_allclose(getattr(left, field), getattr(right, field), rtol=1e-5, atol=1e-4)


def benchmark_process_buffer(*, workers: int) -> None:
    """Confronta la pipeline completa prima/dopo l'accelerazione JIT."""
    status = realtime_dsp.cfar_numba_runtime_status()
    if not status["available"]:
        print("[BENCH] process_buffer end-to-end skipped: Numba unavailable.")
        return
    # Questo non misura il tempo end-to-end: verifica che il backend JIT conservi
    # l'esito della pipeline completa sul medesimo scenario sintetico.
    py_detections = _run_synthetic_process_buffer(use_numba=False, workers=workers)
    jit_detections = _run_synthetic_process_buffer(use_numba=True, workers=workers)
    _assert_detection_lists_equivalent(py_detections, jit_detections)
    print(f"[BENCH] process_buffer synthetic equivalence OK detections={len(py_detections)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual realtime DSP benchmarks.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("Config.yaml"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fft-workers", type=int, nargs="+", default=None, help="Optional FFT worker counts to sweep.")
    parser.add_argument("--skip-e2e", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    shape = _shape_cfg(cfg)
    benchmark_ca_cfar(repeats=args.repeats)
    worker_counts = args.fft_workers if args.fft_workers is not None else [shape["workers"]]
    for workers in worker_counts:
        shape_for_workers = dict(shape)
        shape_for_workers["workers"] = max(1, int(workers))
        benchmark_fft(shape_for_workers, repeats=args.repeats)
    if not args.skip_e2e:
        benchmark_process_buffer(workers=shape["workers"])


if __name__ == "__main__":
    main()
