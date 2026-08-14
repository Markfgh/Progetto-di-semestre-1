"""Benchmark riproducibili delle parti più costose della pipeline realtime.

Il file confronta implementazioni Python/Numba e misura le FFT su forme di
frame coerenti con la configurazione di cattura, senza usare hardware.
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import scipy.fft as fft

import realtime_dsp
from capture_file_inspector import read_capture_header


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


def benchmark_realtime_cfar_profiles(cfg: dict[str, Any], shape: dict[str, int], *, repeats: int = 10) -> None:
    """Measure CA and OS JIT on the configured static and moving map geometries."""
    status = realtime_dsp.cfar_numba_runtime_status()
    if not status.get("enabled") or not status.get("os_enabled"):
        print("[BENCH] realtime CFAR profiles skipped: CA/OS Numba backends are not both enabled.")
        return

    radar = cfg.get("radar", {}) or {}
    processing = cfg.get("processing", {}) or {}
    dr_m = (
        float(radar.get("c", 3e8))
        * float(radar.get("fs", 10e6))
        / (2.0 * float(radar.get("slope", 39.01e12)) * float(shape["nfft_range"]))
    )
    processing_bins = min(
        int(shape["nfft_range"]) // 2,
        max(1, int(np.floor(float(processing.get("range_max_m", 18.0)) / dr_m))),
    )
    static_cfg = cfg.get("detection_static", {}) or {}
    moving_cfg = cfg.get("detection_moving", {}) or {}
    rng = np.random.default_rng(20260812)
    profiles = [
        (
            "static_range_angle",
            rng.lognormal(2.0, 0.9, size=(processing_bins, int(shape["nfft_angle"]))).astype(np.float32),
            static_cfg,
            "cfar_train_col_bins",
            "cfar_guard_col_bins",
        ),
        (
            "moving_range_doppler",
            rng.lognormal(2.0, 0.9, size=(processing_bins, int(shape["loops"]))).astype(np.float32),
            moving_cfg,
            "cfar_train_col_bins",
            "cfar_guard_col_bins",
        ),
    ]
    print(f"[BENCH] configured realtime CFAR maps range_bins={processing_bins}")
    for name, power, block, train_col_key, guard_col_key in profiles:
        kwargs = dict(
            train_range_bins=int(block.get("cfar_train_range_bins", 8)),
            guard_range_bins=int(block.get("cfar_guard_range_bins", 2)),
            train_col_bins=int(block.get(train_col_key, 4)),
            guard_col_bins=int(block.get(guard_col_key, 1)),
            threshold_offset_db=float(block.get("cfar_threshold_db", 9.0)),
            min_power_db=float(block.get("min_power_db", 4.0)),
            os_cfar_rank=int(block.get("os_cfar_rank", 0)),
        )
        for mode in ("ca_cfar", "os_cfar"):
            fn = lambda mode=mode: realtime_dsp.compute_cfar_threshold_db_map(
                power,
                threshold_mode=mode,
                **kwargs,
            )
            fn()
            timings = []
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                fn()
                timings.append((time.perf_counter() - t0) * 1000.0)
            values = np.asarray(timings, dtype=np.float64)
            print(
                f"[BENCH] {name} {mode} shape={power.shape} "
                f"median_ms={float(np.median(values)):.3f} p95_ms={float(np.percentile(values, 95)):.3f}"
            )


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


def _pre_person_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Recreate the previous committed realtime tuning for a relative benchmark."""
    baseline = copy.deepcopy(cfg)
    dsp = baseline.setdefault("dsp", {})
    dsp["window_angle"] = "hanning"
    dsp["heatmap_ema"] = {"enabled": False, "alpha": 0.10}
    display_filters = dsp.setdefault("display_filters", {})
    display_filters.setdefault("slow_time", {})["mode"] = "mean_subtraction"
    display_filters["background_subtraction"] = {
        "enabled": True,
        "mode": "frozen",
        "alpha": 0.02,
        "init_frames": 40,
        "window_frames": 40,
        "clamp_positive_only": True,
    }
    static_filters = dsp.setdefault("detection_static_filters", {})
    static_filters["background_subtraction"] = {
        "enabled": True,
        "mode": "frozen",
        "alpha": 0.02,
        "init_frames": 40,
        "window_frames": 40,
        "clamp_positive_only": False,
    }
    baseline["detection_static"] = {
        "enabled": True,
        "threshold_mode": "relative",
        "threshold_db": -10,
        "localmax_range_bins": 1,
        "localmax_angle_bins": 1,
        "min_power_db": 6,
        "max_detections": 12,
        "cfar_train_range_bins": 8,
        "cfar_guard_range_bins": 2,
        "cfar_train_col_bins": 8,
        "cfar_guard_col_bins": 2,
        "cfar_threshold_db": 10,
        "os_cfar_rank": 0,
    }
    baseline["detection_moving"] = {
        "enabled": True,
        "threshold_mode": "ca_cfar",
        "threshold_db": -7,
        "localmax_range_bins": 2,
        "localmax_doppler_bins": 2,
        "zero_doppler_exclusion_bins": 0,
        "min_power_db": 6,
        "max_detections": 12,
        "cfar_train_range_bins": 8,
        "cfar_guard_range_bins": 2,
        "cfar_train_col_bins": 4,
        "cfar_guard_col_bins": 1,
        "cfar_threshold_db": 10,
        "os_cfar_rank": 0,
    }
    return baseline


def _capture_benchmark_runner(cfg: dict[str, Any], raw_frame: np.ndarray):
    capture = cfg.get("capture", {}) or {}
    radar = cfg.get("radar", {}) or {}
    fft_cfg = cfg.get("fft", {}) or {}
    display = cfg.get("display", {}) or {}
    samples = int(capture["samples"])
    chirps = int(capture["chirps"])
    tx = int(capture["tx"])
    rx = int(capture["rx"])
    loops = chirps // tx
    virtual_ant = tx * rx
    nfft_range = int(fft_cfg["nfft_range"])
    nfft_angle = int(fft_cfg["nfft_angle"])
    workers = int(fft_cfg.get("workers", 1))
    image_cfg = realtime_dsp.display_image_resolutions_from_yaml_dict(cfg).realtime
    gui_h = int(image_cfg.height)
    gui_w = int(image_cfg.width)
    fft_plot_h = nfft_range
    range_max_display = float(display.get("range_max", 15.0))
    range_max_processing = float(realtime_dsp.resolve_processing_range_max_m(cfg))
    dsp_cfg = realtime_dsp.RealtimeDSPConfig(
        c=float(radar["c"]),
        fs=float(radar["fs"]),
        slope=float(radar["slope"]),
        samples=samples,
        chirps=chirps,
        rx=rx,
        tx=tx,
        x_frames=1,
        bytes_per_frame=chirps * samples * rx * 4,
        nfft_range=nfft_range,
        nfft_angle=nfft_angle,
        range_max_display=range_max_display,
        range_profile_count=virtual_ant,
        virtual_ant=virtual_ant,
        fft_workers=workers,
        debug_stats=False,
        range_max_processing_m=range_max_processing,
        normalize_skip_range_bins=0,
        zero_after_range_fft_bins=0,
        range_angle_moving=realtime_dsp.range_angle_moving_from_yaml_dict(cfg),
        display_zoom=realtime_dsp.display_zoom_from_yaml_dict(cfg),
    )
    selection = realtime_dsp.selection_from_yaml_dict(cfg)
    w_range, w_doppler, w_angle = realtime_dsp.build_windows(selection, samples, loops, virtual_ant)
    static_filters, _ = realtime_dsp.sanitize_detection_static_post_range_fft_filters(
        realtime_dsp.detection_static_post_range_fft_filters_from_yaml_dict(cfg)
    )
    moving_filters, _ = realtime_dsp.sanitize_detection_moving_pre_doppler_filters(
        realtime_dsp.detection_moving_pre_doppler_filters_from_yaml_dict(cfg)
    )
    display_filters, _ = realtime_dsp.sanitize_display_post_range_fft_filters(
        realtime_dsp.display_post_range_fft_filters_from_yaml_dict(cfg)
    )
    angle_cfg = realtime_dsp.angle_processing_from_yaml_dict(cfg)
    projection_cfg = realtime_dsp.display_projection_from_yaml_dict(cfg)
    geometry, _ = realtime_dsp.build_virtual_array_geometry_from_yaml_dict(cfg, dsp_cfg)
    angle_axis = realtime_dsp.build_angle_axis_deg(nfft_angle, geometry=geometry)
    steering = realtime_dsp.build_angle_steering_matrix(virtual_ant, nfft_angle, geometry=geometry)
    display_x_max = float(display.get("crossrange_max", 7.5))
    dr_m = dsp_cfg.c * dsp_cfg.fs / (2.0 * dsp_cfg.slope * dsp_cfg.nfft_range)
    projection_lut = realtime_dsp.build_display_projection_lut(
        gui_h,
        gui_w,
        display_x_max,
        range_max_display,
        dr_m,
        angle_axis,
        projection_cfg.projection_mode,
        projection_cfg.projection_interp,
    )
    static_cfg = realtime_dsp.detection_static_from_yaml_dict(cfg)
    moving_cfg = realtime_dsp.detection_moving_from_yaml_dict(cfg)
    doppler_axis = realtime_dsp.build_doppler_axis_mps(
        cfg,
        dsp_cfg,
        loops,
        doppler_fft_shift=moving_cfg.doppler_fft_shift,
    )
    compensation = realtime_dsp.build_tdm_mimo_doppler_compensation_table(
        loops,
        tx,
        doppler_fft_shift=moving_cfg.doppler_fft_shift,
    )
    processing_max_bin = max(1, min(int(np.floor(range_max_processing / dr_m)), nfft_range // 2))
    display_max_bin = max(1, min(int(np.ceil(range_max_display / dr_m)), nfft_range // 2))

    bg_shape = (loops, tx, nfft_range, rx)
    static_bg = realtime_dsp.BackgroundSubtractionState(model=np.zeros(bg_shape, dtype=np.complex64))
    moving_bg = realtime_dsp.BackgroundSubtractionState()
    display_bg = realtime_dsp.BackgroundSubtractionState(model=np.zeros(bg_shape, dtype=np.complex64))
    heatmap_ema = None
    gui_heat_views = (
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
        np.full(gui_h * gui_w, -120.0, dtype=np.float32),
    )
    gui_profile_views = (
        np.full(virtual_ant * fft_plot_h, -120.0, dtype=np.float32),
        np.full(virtual_ant * fft_plot_h, -120.0, dtype=np.float32),
    )
    latest_idx = mp.Value("i", 0)
    latest_seq = mp.Value("i", 0)
    gui_lock = threading.Lock()
    stats = [mp.Value("d", 0.0) for _ in range(4)]
    profiles_out = np.empty((virtual_ant, fft_plot_h), dtype=np.float32)
    virtual_work = np.empty((1, loops, processing_max_bin, tx, rx), dtype=np.complex64)
    virtual_flat_work = np.empty((1, loops, processing_max_bin, virtual_ant), dtype=np.complex64)
    doppler_work = np.empty((1, loops, tx, processing_max_bin, rx), dtype=np.complex64)
    profiles_db_work = np.empty((tx, fft_plot_h, rx), dtype=np.float32)
    heatmap_db_work = np.empty((gui_h, gui_w), dtype=np.float32)

    def run_once() -> int:
        nonlocal heatmap_ema
        heatmap_ema, detections = realtime_dsp.process_buffer(
            raw_frame.copy(),
            1,
            w_range,
            w_doppler,
            w_angle,
            not realtime_dsp._window_is_identity(selection.window_range),
            not realtime_dsp._window_is_identity(selection.window_doppler),
            not realtime_dsp._window_is_identity(selection.window_angle),
            realtime_dsp.mean_before_range_fft_from_yaml_dict(cfg),
            static_filters,
            moving_filters,
            display_filters,
            bool(static_filters.loop_average_after_background.enabled),
            bool(display_filters.loop_average_after_background.enabled),
            angle_cfg,
            realtime_dsp.heatmap_ema_from_yaml_dict(cfg),
            realtime_dsp.heatmap_spatial_filter_from_yaml_dict(cfg),
            projection_cfg,
            geometry,
            steering,
            angle_axis,
            projection_lut,
            range_max_display,
            display_x_max,
            doppler_axis,
            compensation,
            static_cfg,
            moving_cfg,
            realtime_dsp.fusion_from_yaml_dict(cfg),
            static_bg,
            moving_bg,
            display_bg,
            heatmap_ema,
            virtual_work,
            virtual_flat_work,
            doppler_work,
            profiles_db_work,
            heatmap_db_work,
            gui_h,
            gui_w,
            gui_heat_views,
            gui_profile_views,
            latest_idx,
            latest_seq,
            gui_lock,
            True,
            profiles_out,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            dsp_cfg,
            processing_max_bin,
            display_max_bin,
        )
        return len(detections)

    return run_once


def benchmark_capture_frame(path: Path, cfg: dict[str, Any], *, repeats: int) -> None:
    header, data_offset = read_capture_header(path)
    capture = header["capture"]
    frame_i16 = int(capture["chirps"]) * int(capture["samples"]) * int(capture["rx"]) * 2
    packed = np.fromfile(path, dtype=np.int16, count=frame_i16, offset=int(data_offset))
    raw = np.empty(frame_i16 // 2, dtype=np.complex64)
    realtime_dsp._convert_rx4_iiiiqqqq_to_complex64(raw, packed)

    profiles = [("previous", _pre_person_profile(cfg)), ("person", cfg)]
    print(f"[BENCH] captured process_buffer frame={path}")
    for name, profile in profiles:
        run_once = _capture_benchmark_runner(profile, raw)
        for _ in range(4):
            run_once()
        timings = []
        detection_counts = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            detection_counts.append(run_once())
            timings.append((time.perf_counter() - t0) * 1000.0)
        values = np.asarray(timings, dtype=np.float64)
        print(
            f"[BENCH] captured {name} median_ms={float(np.median(values)):.3f} "
            f"p95_ms={float(np.percentile(values, 95)):.3f} "
            f"detections_median={float(np.median(detection_counts)):.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual realtime DSP benchmarks.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("realtime_config.yaml"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fft-workers", type=int, nargs="+", default=None, help="Optional FFT worker counts to sweep.")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--capture", type=Path, default=None, help="Optional rt_capture_v1 frame for relative end-to-end timing.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    shape = _shape_cfg(cfg)
    benchmark_ca_cfar(repeats=args.repeats)
    benchmark_realtime_cfar_profiles(cfg, shape, repeats=args.repeats)
    worker_counts = args.fft_workers if args.fft_workers is not None else [shape["workers"]]
    for workers in worker_counts:
        shape_for_workers = dict(shape)
        shape_for_workers["workers"] = max(1, int(workers))
        benchmark_fft(shape_for_workers, repeats=args.repeats)
    if not args.skip_e2e:
        benchmark_process_buffer(workers=shape["workers"])
    if args.capture is not None:
        benchmark_capture_frame(args.capture, cfg, repeats=args.repeats)


if __name__ == "__main__":
    main()
