# dsp_selection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any

import numpy as np


### ---------------------------- WINDOW SELECTION ----------------------------

WindowType = Literal["rectangular", "hanning", "hamming", "blackman"]

@dataclass(frozen=True)
class DspSelection:
    window_range: WindowType = "blackman"
    window_angle: WindowType = "hanning"

def _get_window_1d(win_type: str, size: int) -> np.ndarray:
    wt = win_type.lower()
    if wt == "rectangular":
        return np.ones(size, dtype=np.float32)
    if wt == "hanning":
        return np.hanning(size).astype(np.float32, copy=False)
    if wt == "hamming":
        return np.hamming(size).astype(np.float32, copy=False)
    if wt == "blackman":
        return np.blackman(size).astype(np.float32, copy=False)
    raise ValueError(f"Unknown window type: {win_type!r}. Use: rectangular|hanning|hamming|blackman")


def build_windows(selection: DspSelection, samples: int, virtual_ant: int):
    """
    Restituisce finestre già reshape() per il tuo broadcasting:
    - window_range: (1,1,1,SAMPLES,1)
    - window_angle: (1,1,1,VIRTUAL_ANT)
    """
    w_range = _get_window_1d(selection.window_range, samples).reshape(1, 1, 1, samples, 1)
    w_angle = _get_window_1d(selection.window_angle, virtual_ant).reshape(1, 1, 1, virtual_ant)
    return w_range, w_angle


def selection_from_yaml_dict(cfg: Dict[str, Any]) -> DspSelection:
    """
    Atteso YAML tipo:
    dsp:
      window_range: blackman
      window_angle: hanning
    """
    dsp = cfg.get("dsp", {}) or {}
    return DspSelection(
        window_range=str(dsp.get("window_range", "blackman")),
        window_angle=str(dsp.get("window_angle", "hanning")),
    )
### ---------------------------- ------------------ ----------------------------
