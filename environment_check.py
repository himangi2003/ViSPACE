#!/usr/bin/env python3
"""
environment_check.py
=====================
Pre-flight environment check for the ViSPACE pipeline.

Run this BEFORE `run_vispace.py` (or any individual stage script) to catch
missing dependencies, GPU/precision issues, and bad config paths up front —
instead of discovering them partway through a multi-hour run.

Checks performed
-----------------
1. Python version
2. Required packages importable (torch, timm, huggingface_hub, cv2, pandas,
   numpy, shapely, geopandas, tqdm, PIL, ml_dtypes, open_clip)
3. Optional packages (openslide — only needed for the WSI thumbnail overlay
   in tumor_roi_overlay.py)
4. GPU / CUDA: availability, device name, compute capability, exclusive-mode
   lock detection, and which autocast precision (bf16/fp16/fp32)
   segmenter.py will select on this hardware.
5. HuggingFace auth: whether HF_TOKEN is set (needed to download Virchow2,
   unless cfg.VIRCHOW2_PATH points at local weights)
6. Config paths: WSI_PATH and CHECKPOINT exist; OUT_DIR is writable

Usage (as a library)
---------------------
    from environment_check import run_environment_check
    from config import cfg

    ok = run_environment_check(cfg)
    if not ok:
        raise SystemExit("Environment check failed — see above.")

Usage (from the command line)
------------------------------
    python environment_check.py --from-json run_config.json
    python environment_check.py   # deps + GPU only, skips path checks
    python environment_check.py --help

Exit codes
----------
0  — all checks passed
1  — one or more required checks failed
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as package_version
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Console formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_OK   = "  [OK]   "
_WARN = "  [WARN] "
_FAIL = "  [FAIL] "


def _print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python version
# ─────────────────────────────────────────────────────────────────────────────

def check_python_version(min_version=(3, 10)) -> bool:
    _print_header("Python version")
    v = sys.version_info
    print(f"  Running     : Python {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) >= min_version:
        print(f"{_OK}Python {v.major}.{v.minor} meets minimum "
              f"{min_version[0]}.{min_version[1]}")
        return True
    print(f"{_FAIL}Python {v.major}.{v.minor} is older than the recommended "
          f"{min_version[0]}.{min_version[1]} (ViSPACE_environment.yml pins 3.11)")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2 & 3. Package imports
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("mussel",            "mussel-pathology"),
    ("torch",             "torch"),
    ("torchvision",       "torchvision"),
    ("timm",              "timm"),
    ("huggingface_hub",   "huggingface_hub"),
    ("numpy",             "numpy"),
    ("pandas",            "pandas"),
    ("cv2",               "opencv-python-headless"),
    ("PIL",               "pillow"),
    ("tqdm",              "tqdm"),
    ("shapely",           "shapely"),
    ("geopandas",         "geopandas"),
    ("scipy",             "scipy"),
    ("matplotlib",        "matplotlib"),
    ("ml_dtypes",         "ml_dtypes"),
    ("open_clip",         "open_clip_torch"),
    ("omegaconf",         "omegaconf"),
    ("h5py",              "h5py"),
]

OPTIONAL_PACKAGES = [
    ("openslide", "openslide-python + openslide-bin",
     "only needed for the WSI thumbnail overlay in tumor_roi_overlay.py "
     "(step 6) — everything else runs fine without it"),
]


def _distribution_version(pip_name: str, module) -> str:
    """Return the installed distribution version when available."""
    try:
        return package_version(pip_name)
    except PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown version"))
    except Exception:
        return str(getattr(module, "__version__", "unknown version"))


def check_required_packages() -> bool:
    _print_header("Required packages")
    all_ok = True

    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            installed_version = _distribution_version(pip_name, mod)
            print(
                f"{_OK}{import_name:<18} ({pip_name}) — "
                f"{installed_version}"
            )
        except ImportError:
            all_ok = False
            print(
                f"{_FAIL}{import_name:<18} ({pip_name}) — NOT INSTALLED"
            )
            if pip_name == "mussel-pathology":
                print(
                    '           Fix: pip install '
                    '"mussel-pathology[torch-gpu]==1.4.4"'
                )
            else:
                print(f"           Fix: pip install {pip_name}")

    try:
        mussel_version = package_version("mussel-pathology")
        if mussel_version == "1.4.4":
            print(
                f"{_OK}mussel-pathology version matches "
                "the tested ViSPACE version (1.4.4)."
            )
        else:
            print(
                f"{_WARN}mussel-pathology {mussel_version} is installed; "
                "ViSPACE was tested with 1.4.4."
            )
    except PackageNotFoundError:
        pass

    return all_ok


def check_optional_packages() -> None:
    _print_header("Optional packages")
    for import_name, pip_name, note in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", None) or getattr(
                mod, "__library_version__", "unknown version")
            print(f"{_OK}{import_name:<18} ({pip_name}) — {version}")
        except ImportError:
            print(f"{_WARN}{import_name:<18} ({pip_name}) — not installed")
            print(f"           {note}")
            print(f"           Fix: pip install {pip_name}")
        except OSError as e:
            print(f"{_WARN}{import_name:<18} bindings installed, but the "
                  f"compiled library is missing")
            print(f"           {e}")
            print(f"           Fix: pip install openslide-bin")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPU / CUDA / precision  (includes busy-device detection)
# ─────────────────────────────────────────────────────────────────────────────

def _nvidia_smi_gpu_info() -> list[dict]:
    """
    Query nvidia-smi for per-GPU process and compute-mode info.
    Returns a list of dicts (one per GPU); empty list if nvidia-smi unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_mode",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5:
            gpus.append({
                "index":        parts[0],
                "name":         parts[1],
                "memory_total": parts[2],
                "memory_free":  parts[3],
                "compute_mode": parts[4],   # "Default" | "Exclusive_Process" | "Prohibited"
            })
    return gpus


def _nvidia_smi_processes(gpu_index: str) -> list[str]:
    """Return PIDs using gpu_index according to nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
                f"--id={gpu_index}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return [p.strip() for p in out.strip().splitlines() if p.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def _is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def check_gpu(device_pref: str = "cuda") -> bool:
    """
    Check GPU availability, memory, compute mode, exclusive-process locks,
    and the autocast dtype segmenter.py will use.

    Also detects the 'CUDA-capable device(s) is/are busy or unavailable'
    error that surfaces when:
      - The GPU is in Exclusive_Process mode and another job owns it
      - The GPU is in Prohibited mode (sysadmin locked)
      - CUDA initialisation fails for any other reason (driver mismatch, etc.)

    Colab note
    ----------
    On Google Colab, nvidia-smi process/mode scanning is skipped because:
      - Colab GPUs are always in Default compute mode (Exclusive_Process
        never occurs)
      - --query-compute-apps returns no results inside the container even
        when the GPU is in use, so the scan would give false confidence
    The CUDA smoke test and device-property checks still run normally.
    """
    _print_header("GPU / CUDA")
    try:
        import torch
    except ImportError:
        print(f"{_FAIL}torch not installed — cannot check GPU.")
        return False

    if device_pref != "cuda":
        print(f"  cfg.DEVICE={device_pref!r} — GPU checks skipped, "
              f"inference will run on CPU (slow but correct).")
        return True

    # ── Step 1: nvidia-smi sanity check (driver-level, before CUDA init) ──
    colab = _is_colab()
    if colab:
        # On Colab: still show basic GPU info from nvidia-smi (useful to
        # confirm which GPU was assigned), but skip compute-mode and
        # per-process checks — they're unreliable inside the container.
        print(f"  Running on Google Colab — skipping compute-mode / "
              f"exclusive-process scan (not applicable in this environment).")
        smi_gpus = _nvidia_smi_gpu_info()
        if smi_gpus:
            for g in smi_gpus:
                print(f"  nvidia-smi : GPU {g['index']} {g['name']}  "
                      f"{g['memory_free']}/{g['memory_total']} MiB free")
    else:
        smi_gpus = _nvidia_smi_gpu_info()
        if not smi_gpus:
            print(f"{_WARN}nvidia-smi not available or returned no GPUs.")
            print(f"           Cannot detect driver-level issues before CUDA init.")
        else:
            print(f"  nvidia-smi sees {len(smi_gpus)} GPU(s):")
            for g in smi_gpus:
                pids = _nvidia_smi_processes(g["index"])
                mode = g["compute_mode"]
                print(f"    GPU {g['index']}: {g['name']}  "
                      f"{g['memory_free']}/{g['memory_total']} MiB free  "
                      f"mode={mode}  "
                      f"procs={pids or 'none'}")

                if mode == "Prohibited":
                    print(f"{_FAIL}    GPU {g['index']} is in Prohibited compute mode — "
                          f"no process can use it.")
                    print(f"           Ask your sysadmin to run: "
                          f"nvidia-smi -c 0 -i {g['index']}")
                    return False

                if mode == "Exclusive_Process" and pids:
                    print(f"{_FAIL}    GPU {g['index']} is in Exclusive_Process mode "
                          f"and is held by PID(s): {', '.join(pids)}")
                    print(f"           This is the most common cause of:")
                    print(f"           'CUDA error: CUDA-capable device(s) is/are "
                          f"busy or unavailable'")
                    print(f"           Options:")
                    print(f"             1. Wait for PID {pids[0]} to finish")
                    print(f"             2. Kill it:  kill {pids[0]}")
                    print(f"             3. Switch to a free GPU: "
                          f"CUDA_VISIBLE_DEVICES=<other_id> python ...")
                    print(f"             4. Reset compute mode (if you have perms): "
                          f"nvidia-smi -c 0 -i {g['index']}")
                    return False

    # ── Step 2: CUDA init via torch (catches driver mismatch etc.) ────────
    if not torch.cuda.is_available():
        print(f"{_WARN}torch.cuda.is_available() = False")
        print(f"           Possible causes:")
        print(f"             - No GPU present")
        print(f"             - CUDA driver/toolkit version mismatch")
        print(f"             - torch built without CUDA (check: python -c "
              f"\"import torch; print(torch.version.cuda)\")")
        print(f"           Inference will run on CPU — significantly slower.")
        return True   # not fatal; pipeline still runs on CPU

    # ── Step 3: Attempt a minimal CUDA allocation to catch 'busy' early ───
    try:
        import torch
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except RuntimeError as e:
        err = str(e)
        print(f"{_FAIL}CUDA initialisation failed: {err}")
        if "busy or unavailable" in err:
            print(f"           This usually means another process holds the GPU")
            print(f"           in Exclusive_Process mode (see GPU list above).")
            print(f"           Debug tip: re-run with CUDA_LAUNCH_BLOCKING=1")
            print(f"           to get a synchronous, accurate stack trace.")
        elif "out of memory" in err.lower():
            print(f"           The GPU doesn't have enough free memory even for")
            print(f"           a minimal allocation. Kill other GPU processes first.")
        else:
            print(f"           Try: CUDA_LAUNCH_BLOCKING=1 python run_vispace.py ...")
        return False

    # ── Step 4: Device properties ─────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    name   = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  torch sees   : {n_gpus} GPU(s)")
    print(f"  Device 0     : {name}")
    print(f"  Compute cap  : {major}.{minor}")
    print(f"  VRAM         : {total_mem_gb:.1f} GB")

    if total_mem_gb < 8:
        print(f"{_WARN}Under 8 GB VRAM — Virchow2 needs ~8 GB. "
              f"Lower cfg.BATCH_SIZE or free other GPU processes.")

    # ── Step 5: Autocast dtype (mirrors segmenter.py exactly) ────────────
    try:
        from segmenter import _best_autocast_dtype
        dtype = _best_autocast_dtype("cuda")
    except ImportError:
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    print(f"  AMP dtype    : {dtype}  "
          f"({'tensor-core accelerated' if major >= 7 else 'no tensor cores — old GPU'})")

    if major < 6:
        print(f"{_WARN}Compute capability {major}.{minor} is pre-Pascal. "
              f"Mixed precision may give no speedup on this GPU.")
    else:
        print(f"{_OK}GPU ready for inference.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. HuggingFace auth
# ─────────────────────────────────────────────────────────────────────────────

def check_huggingface_auth(virchow2_path: Optional[str] = None) -> bool:
    _print_header("HuggingFace authentication")

    if virchow2_path:
        p = Path(virchow2_path)
        if p.exists():
            print(f"{_OK}cfg.VIRCHOW2_PATH exists: {virchow2_path}")
            print(f"           Using local Virchow2 weights. Hugging Face authentication is not required.")
            return True
        else:
            print(f"{_FAIL}cfg.VIRCHOW2_PATH set but does not exist: {virchow2_path}")
            return False

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"{_WARN}HF_TOKEN not set and cfg.VIRCHOW2_PATH not configured.")
        print(f"           Virchow2 will be downloaded from HF Hub — auth required.")
        print(f"           Fix: huggingface-cli login  OR  export HF_TOKEN=hf_...")
        return False

    try:
        from huggingface_hub import whoami
        user = whoami(token=token)
        print(f"{_OK}HF_TOKEN valid — authenticated as {user['name']}")
        return True
    except ImportError:
        print(f"{_FAIL}huggingface_hub not installed — cannot verify token.")
        return False
    except Exception as e:
        print(f"{_FAIL}HF_TOKEN set but invalid: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Config paths
# ─────────────────────────────────────────────────────────────────────────────

def check_config_paths(cfg) -> bool:
    _print_header("Config paths")
    all_ok = True

    checks = [
        ("WSI_PATH",   cfg.WSI_PATH,   "the slide file (.svs / .tif)"),
        ("CHECKPOINT", cfg.CHECKPOINT, "the ViSPACE segmentation model checkpoint (.pt)"),
    ]
    for field_name, value, description in checks:
        if not value or value == "your data path":
            all_ok = False
            print(f"{_FAIL}cfg.{field_name} is not set — {description}")
            continue
        p = Path(value)
        if p.exists():
            print(f"{_OK}cfg.{field_name:<11} exists: {value}")
        else:
            all_ok = False
            print(f"{_FAIL}cfg.{field_name:<11} does not exist: {value}")
            print(f"           Expected: {description}")

    out_dir = Path(cfg.OUT_DIR).expanduser()

    if out_dir.exists():
        if not out_dir.is_dir():
            all_ok = False
            print(
                f"{_FAIL}cfg.OUT_DIR exists but is not a directory: "
                f"{cfg.OUT_DIR}"
            )
        elif os.access(out_dir, os.W_OK):
            print(
                f"{_OK}cfg.OUT_DIR exists and is writable: "
                f"{cfg.OUT_DIR}"
            )
        else:
            all_ok = False
            print(
                f"{_FAIL}cfg.OUT_DIR exists but is not writable: "
                f"{cfg.OUT_DIR}"
            )
    else:
        parent = out_dir.parent if str(out_dir.parent) else Path(".")
        if parent.exists() and os.access(parent, os.W_OK):
            print(f"{_WARN}cfg.OUT_DIR does not exist yet: {cfg.OUT_DIR}")
            print("           It will be created automatically on first run.")
        else:
            all_ok = False
            print(
                f"{_FAIL}cfg.OUT_DIR cannot be created because its parent "
                f"is missing or not writable: {parent}"
            )

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Top-level callable
# ─────────────────────────────────────────────────────────────────────────────

def run_environment_check(cfg=None) -> bool:
    results = []

    results.append(("Python version",    check_python_version()))
    results.append(("Required packages", check_required_packages()))
    check_optional_packages()

    device_pref = getattr(cfg, "DEVICE", "cuda") if cfg is not None else "cuda"
    results.append(("GPU / CUDA",        check_gpu(device_pref)))

    if cfg is not None:
        results.append(("HuggingFace auth", check_huggingface_auth(
            getattr(cfg, "VIRCHOW2_PATH", None))))
        results.append(("Config paths",     check_config_paths(cfg)))
    else:
        print(f"\n{_WARN}No config passed — skipping HuggingFace auth and "
              f"path checks.")

    _print_header("Summary")
    all_passed = True
    for name, passed in results:
        print(f"{'  [OK]  ' if passed else '  [FAIL]'} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All required checks passed.")
        print("  The ViSPACE environment is ready for analysis.")
    else:
        print("  One or more required checks failed.")
        print("  Resolve the issues above before running ViSPACE.")
    print()
    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    from config import config_from_args
    cfg, _ = config_from_args(argv)
    has_real_cfg = bool(cfg.WSI_PATH) and cfg.WSI_PATH != "your data path"
    ok = run_environment_check(cfg if has_real_cfg else None)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()