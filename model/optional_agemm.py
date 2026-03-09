from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Optional


def _kernels_build_dir() -> Path:
    # model/ -> repo root -> kernels/build
    return Path(__file__).resolve().parent.parent / "kernels" / "build"


def _try_import_agemm() -> tuple[Optional[ModuleType], Optional[BaseException]]:
    build_dir = _kernels_build_dir()
    if build_dir.exists():
        build_dir_str = str(build_dir)
        if build_dir_str not in sys.path:
            sys.path.append(build_dir_str)

    try:
        import agemm  # type: ignore
    except BaseException as exc:
        return None, exc

    return agemm, None


def _device_sm() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    except BaseException:
        return None


_FORCE_DISABLE = os.getenv("ARCQUANT_DISABLE_AGEMM", "").strip().lower() in {"1", "true", "yes"}

agemm, AGEMM_IMPORT_ERROR = _try_import_agemm()
DEVICE_SM = _device_sm()

# NOTE: kernels/CMakeLists.txt builds for SM120a only.
HAS_AGEMM = (not _FORCE_DISABLE) and (agemm is not None) and (DEVICE_SM is not None) and (DEVICE_SM >= 120)

_warned = False


def warn_agemm_fallback_once(reason: str) -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        f"agemm is unavailable/disabled ({reason}); falling back to fakequant + PyTorch ops. "
        "Set ARCQUANT_DISABLE_AGEMM=0 and run on SM120+ to enable kernels.",
        RuntimeWarning,
        stacklevel=2,
    )


def require_agemm() -> ModuleType:
    if HAS_AGEMM and agemm is not None:
        return agemm
    if _FORCE_DISABLE:
        raise RuntimeError("agemm is disabled by ARCQUANT_DISABLE_AGEMM=1")
    if DEVICE_SM is None:
        raise RuntimeError("agemm requires CUDA with SM120+ (no CUDA device detected)")
    if DEVICE_SM < 120:
        raise RuntimeError(f"agemm requires CUDA SM120+ (detected SM{DEVICE_SM})")
    if AGEMM_IMPORT_ERROR is not None:
        raise RuntimeError(f"failed to import agemm: {AGEMM_IMPORT_ERROR}") from AGEMM_IMPORT_ERROR
    raise RuntimeError("agemm is unavailable")

