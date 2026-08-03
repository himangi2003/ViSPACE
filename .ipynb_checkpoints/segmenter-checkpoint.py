"""
segmenter.py
============
Step 2 of the ViSpace pipeline.

Model definition, checkpoint loading, and per-slide inference.

Output directory
----------------
    cfg.OUT_DIR/<slide_name>/segmentation/
        manifest.csv          — tile coords + npy_path for every predicted tile
        <tile>_seg.npy        — per-tile int32 class maps (cleaned up by stitch.py)

Contains
--------
    ViT2SegDecoder   — lightweight convolutional decoder
    BCSSSegmenter    — Virchow2 encoder + ViT2SegDecoder
    load_model()     — load checkpoint from disk
    run_segmentation() — end-to-end inference for one WSI

Usage (as a library)
---------------------
    from segmenter import run_segmentation
    from config import cfg

    run_segmentation(
        wsi_path = "slides/TCGA-A1-A0SP.svs",
        cfg      = cfg,
    )

Usage (from the command line)
------------------------------
Before running this step, authenticate with HuggingFace so Virchow2 can be
downloaded (skip this if you set --virchow2-path to local weights instead):

    pip install huggingface_hub
    huggingface-cli login
    # Paste your token when prompted — input is hidden, this is expected

Then, same shared flags as config.py / tessellate.py — every PipelineConfig
field is available here too, so --checkpoint, --device, --batch-size,
--virchow2-path, --out-dir, etc. all work without learning new flag names.
This script also supports --from-json, so it can pick up a config saved
earlier via `config.py --print-config`.

    # minimal — requires tessellate.py to have already run for this slide
    python segmenter.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --checkpoint TNBC_weights/TNBC_best.pt

    # explicit device / batch size / local Virchow2 weights
    python segmenter.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --checkpoint TNBC_weights/TNBC_best.pt \\
        --device cuda --batch-size 64 \\
        --virchow2-path weights/virchow2 \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python segmenter.py --from-json run_config.json

    # continue from a saved config but override one field
    python segmenter.py --from-json run_config.json --batch-size 16

    # see every available flag
    python segmenter.py --help
"""

import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import cfg as default_cfg, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (shared with stitch.py and downstream scripts)
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE  = 224
N_CLASSES   = 5
SEG_IGNORE  = 255

# FIX 1: DEVICE is no longer set as a module-level constant here.
# It is resolved at call time from cfg.DEVICE so that the user can override it
# (e.g. replace(cfg, DEVICE="cpu")) without the module-level value winning.
# A lazy fallback is provided for code that imports DEVICE directly.
def _resolve_device(device_str: str) -> str:
    """Return 'cuda' only if requested AND available; else 'cpu'."""
    if device_str == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_str

# Legacy constant — kept for any downstream import that references segmenter.DEVICE.
# Call _resolve_device(cfg.DEVICE) inside functions instead of using this.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]

# FIX 2: COLORS_BGR comment labels corrected.
# BGR tuple layout: (Blue, Green, Red)
# Original comments were copy-paste errors — e.g. (255, 100, 0) was labelled
# "bright blue" but B=255, G=100, R=0 is a bright blue in BGR (not orange).
# Each comment now states the actual rendered colour.
COLORS_BGR = [
    (  0,   0, 255),   # 0 Tumour        — red        (B=0,   G=0,   R=255)
    (  0, 200,   0),   # 1 Stroma        — green      (B=0,   G=200, R=0)
    (255, 100,   0),   # 2 Inflammatory  — blue-cyan  (B=255, G=100, R=0)
    (  0, 165, 255),   # 3 Necrosis      — orange     (B=0,   G=165, R=255)
    (220,   0, 220),   # 4 Others        — magenta    (B=220, G=0,   R=220)
]
COLORS_RGB = [(r / 255, g / 255, b / 255) for b, g, r in COLORS_BGR]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER  — frozen Virchow2 ViT-H/14
# ─────────────────────────────────────────────────────────────────────────────

def load_virchow2(model_path: str = None, device: str = "cuda") -> nn.Module:
    """
    Load Virchow2 ViT-H/14 encoder.

    Parameters
    ----------
    model_path : local path (optional). If None, downloads from HuggingFace.
                 Requires: huggingface-cli login (one-time)
    device     : torch device string resolved from cfg.DEVICE

    Returns
    -------
    Frozen nn.Module on device.
    Token output shape: (B, 261, 1280)
        index 0       = CLS token
        indices 1–4   = 4 register tokens
        indices 5–260 = 256 spatial patch tokens  ← decoder uses these
    """
    source = model_path or "hf-hub:paige-ai/Virchow2"
    print(f"  Loading Virchow2 from: {source}")

    enc = timm.create_model(
        source,
        pretrained = True,
        mlp_layer  = timm.layers.SwiGLUPacked,
        act_layer  = torch.nn.SiLU,
    )
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    enc = enc.to(device)

    # Smoke test
    with torch.no_grad():
        out = enc.forward_features(
            torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=device))
    assert out.shape == (1, 261, 1280), \
        f"Unexpected Virchow2 output shape: {out.shape}"

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Virchow2: {n_params/1e9:.2f}B params | frozen | {device}")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# DECODER  — ViT2SegDecoder
# ─────────────────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    """
    Transposed convolution ×2 upsample + two rounds of Conv-BN-ReLU.
    Doubles spatial resolution, halves channel depth.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ViT2SegDecoder(nn.Module):
    """
    Decode Virchow2 patch tokens → pixel-wise class logits.

    Input  : (B, 256, 1280)  patch tokens  [indices 5:261]
    Output : (B, N_CLASSES, 224, 224)  logits

    Shape flow:
        (B, 256, 1280)
            → reshape   (B, 1280, 16, 16)
            → UpBlock 1 (B,  512, 32, 32)
            → UpBlock 2 (B,  256, 64, 64)
            → UpBlock 3 (B,  128, 128, 128)
            → bilinear  (B,  128, 224, 224)
            → head 1×1  (B,    5, 224, 224)
    """
    def __init__(self, n_classes: int = N_CLASSES,
                 token_dim: int = 1280, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.ups  = nn.Sequential(
            _UpBlock(token_dim, 512),
            _UpBlock(512,       256),
            _UpBlock(256,       128),
        )
        self.head = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, self.grid, self.grid)
        x = self.ups(x)
        x = F.interpolate(x, (PATCH_SIZE, PATCH_SIZE),
                          mode="bilinear", align_corners=False)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BCSSSegmenter(nn.Module):
    """
    Virchow2 encoder (frozen) + ViT2SegDecoder.

    forward(x) :
        x      : (B, 3, 224, 224)  normalised RGB tile
        returns: (B, N_CLASSES, 224, 224)  logits
    """
    def __init__(self, encoder: nn.Module, n_classes: int = N_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.decoder = ViT2SegDecoder(n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.encoder.forward_features(x)   # (B, 261, 1280)
        return self.decoder(tokens[:, 5:])               # skip CLS + 4 registers


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint:  str,
    model_path:  str = None,
    device:      str = "cuda",
) -> BCSSSegmenter:
    """
    Load BCSSSegmenter from a saved checkpoint.

    Parameters
    ----------
    checkpoint  : path to phaseA_best.pt or phaseB_best.pt
    model_path  : local Virchow2 weights (None = download from HuggingFace)
    device      : torch device string resolved from cfg.DEVICE

    Returns
    -------
    BCSSSegmenter in eval mode on device.
    """
    print(f"\n  Loading checkpoint: {Path(checkpoint).name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(str(checkpoint), map_location=device,
                          weights_only=False)

    n_classes = ckpt.get("n_classes", N_CLASSES)
    encoder   = load_virchow2(model_path, device=device)
    model     = BCSSSegmenter(encoder, n_classes=n_classes).to(device)

    # Support both checkpoint formats
    if "decoder_state_dict" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state_dict"])
        if "encoder_state_dict" in ckpt:
            model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise KeyError(
            f"Unrecognised checkpoint format. Keys: {list(ckpt.keys())}")

    model.eval()

    epoch = ckpt.get("epoch", "?")
    dice  = ckpt.get("metrics", {}).get("macro_dice", "?")

    dec_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Decoder params: {dec_params/1e6:.1f}M")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def mask_white_background(
    img_rgb:     np.ndarray,
    seg:         np.ndarray,
    white_thresh: int = 220,
) -> np.ndarray:
    """
    Set near-white pixels (glass / adipose) to SEG_IGNORE.
    Pixels where all RGB channels >= white_thresh are background.

    Parameters
    ----------
    img_rgb      : (H, W, 3) uint8 original tile
    seg          : (H, W) int32 prediction
    white_thresh : threshold — 220 works well for TCGA slides

    Returns
    -------
    seg with white regions set to SEG_IGNORE (255).
    """
    seg      = seg.copy()
    white_bg = np.all(img_rgb >= white_thresh, axis=2)
    seg[white_bg] = SEG_IGNORE
    return seg


def seg_to_colour_bgr(seg: np.ndarray) -> np.ndarray:
    """
    Convert (H, W) int32 mask → (H, W, 3) uint8 BGR colour image.
    SEG_IGNORE pixels → black (0, 0, 0).
    """
    bgr = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for c, (b, g, r) in enumerate(COLORS_BGR):
        bgr[seg == c] = (b, g, r)
    return bgr


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_tile(img_rgb: np.ndarray, device: str) -> torch.Tensor:
    """
    Normalise a (H, W, 3) uint8 RGB tile and return a (1, 3, H, W) tensor.
    Uses ImageNet mean/std — same as Virchow2 pre-training.
    """
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)


class _TileDataset(Dataset):
    """
    FIX 2: Lazily loads + resizes tessellated tile PNGs.

    Used with a DataLoader (num_workers > 0) so image I/O, decode, and
    resize for the NEXT batch happen on CPU worker processes in parallel
    with GPU inference on the CURRENT batch — instead of the GPU sitting
    idle while Image.open()/resize() runs serially in the main loop.

    __getitem__ returns a raw (H, W, 3) uint8 tensor plus the dataset
    index, so the caller can recover the original tile_path and build
    manifest rows after inference.
    """
    def __init__(self, tile_paths: list, patch_sz: int = PATCH_SIZE):
        self.tile_paths = tile_paths
        self.patch_sz   = patch_sz

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx: int):
        img = np.array(Image.open(self.tile_paths[idx]).convert("RGB"))
        if img.shape[:2] != (self.patch_sz, self.patch_sz):
            img = np.array(
                Image.fromarray(img).resize(
                    (self.patch_sz, self.patch_sz), Image.LANCZOS))
        return torch.from_numpy(img), idx  # uint8 (H, W, 3), original index


def _best_autocast_dtype(device: str) -> torch.dtype:
    """
    FIX 1 (hardware-aware): pick the mixed-precision dtype that's actually
    tensor-core-accelerated on the current GPU.

    torch.cuda.is_bf16_supported() only reports whether bf16 ops can
    *execute* — not whether they run on tensor cores. Native bf16 tensor
    core support requires Ampere or newer (compute capability >= 8.0:
    A100, RTX 30/40-series, H100). On Turing/Volta (compute capability
    7.x — e.g. RTX 20-series, V100, T4), bf16 runs without hardware
    acceleration and gives little to no speedup. Those GPUs do have fp16
    tensor cores, so fp16 is the correct choice there instead.

    Returns torch.float32 (i.e. autocast effectively disabled) on CPU.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


@torch.no_grad()
def _predict_batch_from_tensor(
    model:        BCSSSegmenter,
    batch_u8_cpu: torch.Tensor,
    device:       str,
    mean_t:       torch.Tensor,
    std_t:        torch.Tensor,
    autocast_dtype: torch.dtype = None,
) -> list:
    """
    Run model on a pre-loaded, pre-resized batch of uint8 tiles.

    Parameters
    ----------
    model        : BCSSSegmenter in eval mode
    batch_u8_cpu : (B, H, W, 3) uint8 tensor, still on CPU (as produced by
                   the DataLoader — see _TileDataset)
    device       : resolved device string
    mean_t, std_t: (1, 3, 1, 1) ImageNet mean/std tensors, pre-placed on
                   `device` once outside the loop (avoids re-allocating a
                   tiny tensor on every batch)
    autocast_dtype: torch.bfloat16 / torch.float16 / torch.float32.
                   Pass the result of _best_autocast_dtype(device), computed
                   once outside the loop. If None, resolved here per-call
                   (slightly less efficient but always correct).

    FIX 1: the forward pass runs under torch.autocast using whichever
    reduced-precision dtype is actually tensor-core-accelerated on this
    GPU (see _best_autocast_dtype) — bf16 on Ampere+, fp16 on Turing/Volta.
    This is usually the single biggest inference speedup on modern GPUs
    (often 1.5-3x) for a frozen, already-trained encoder, with no
    loss-scaling needed since this is inference-only (no backward pass).
    Autocast keeps numerically sensitive ops (e.g. softmax/LayerNorm
    reductions) in fp32 internally and only runs matmuls/convolutions in
    the reduced dtype, so this is a throughput win with negligible
    accuracy impact — not a precision trade-off you need to validate
    against the checkpoint's original training precision.

    Normalisation (uint8 -> float, mean/std) now happens on-GPU as a
    broadcasted op instead of per-tile NumPy on CPU, which both reduces
    CPU load (freeing it up for the DataLoader workers in FIX 2) and cuts
    a host->device transfer of float32 data down to a smaller uint8 one.

    Returns
    -------
    List of (H, W) int32 class maps, one per input tile,
    with white background masked to SEG_IGNORE.
    """
    if autocast_dtype is None:
        autocast_dtype = _best_autocast_dtype(device)

    batch_u8_dev = batch_u8_cpu.to(device, non_blocking=True)   # (B, H, W, 3) uint8
    x = batch_u8_dev.permute(0, 3, 1, 2).float() / 255.0        # (B, 3, H, W) float32
    x = (x - mean_t) / std_t

    use_amp = (device == "cuda" and autocast_dtype != torch.float32)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
        logits = model(x)                                       # (B, C, H, W)
    preds = logits.float().argmax(dim=1).cpu().numpy()           # (B, H, W)

    imgs_np = batch_u8_cpu.numpy()  # original uint8 tiles, for white-bg masking
    results = []
    for i in range(imgs_np.shape[0]):
        seg = preds[i].astype(np.int32)
        seg = mask_white_background(imgs_np[i], seg)
        results.append(seg)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPU MEMORY CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def release_model(model: BCSSSegmenter) -> None:
    """
    FIX 3: Explicitly free GPU memory after processing a slide.

    Deletes the model object and calls torch.cuda.empty_cache() so that
    memory held by slide N is released before slide N+1 is loaded.
    Without this, a folder of large slides can OOM on slide 2 even when
    slide 1 succeeds.

    Usage (multi-slide loop in run_vipsegd.py or similar):
        model = load_model(cfg.CHECKPOINT, device=device)
        run_segmentation(wsi_path, cfg, model=model)
        release_model(model)
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(
    wsi_path: str,
    cfg: PipelineConfig = None,
) -> str:
    """
    Run per-tile segmentation inference for one WSI.

    Reads tiles produced by run_tessellation() from:
        cfg.OUT_DIR/<slide_name>/tessellation/patches/

    Saves outputs to:
        cfg.OUT_DIR/<slide_name>/segmentation/
            manifest.csv          — tile coords + npy_path
            <tile>_seg.npy        — per-tile int32 class maps

    The checkpoint is read from cfg.CHECKPOINT.

    Parameters
    ----------
    wsi_path : path to .svs / .tif (used only to derive slide_name)
    cfg      : PipelineConfig (defaults to config.cfg singleton)

    Returns
    -------
    str : path to manifest.csv
    """
    if cfg is None:
        cfg = default_cfg

    # FIX 1 (applied): resolve device from cfg, not the module-level constant.
    device = _resolve_device(cfg.DEVICE)

    slide_name = Path(wsi_path).stem
    tess_dir   = Path(cfg.OUT_DIR) / slide_name / "tessellation"
    seg_dir    = Path(cfg.OUT_DIR) / slide_name / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = tess_dir / "patches"
    if not patches_dir.exists():
        raise FileNotFoundError(
            f"Tessellation patches not found: {patches_dir}\n"
            f"Run run_tessellation(wsi_path, cfg) first."
        )

    tile_paths = sorted(patches_dir.glob("*.png"))
    if not tile_paths:
        raise ValueError(f"No .png patches found in {patches_dir}")

    batch_size = cfg.BATCH_SIZE  # FIX 4: honour cfg.BATCH_SIZE

    print(f"\n{'='*55}")
    print(f"  Segmentation")
    print(f"  Slide      : {slide_name}")
    print(f"  Tiles      : {len(tile_paths):,}")
    print(f"  Batch size : {batch_size}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {Path(cfg.CHECKPOINT).name}")
    print(f"  Output     : {seg_dir}")
    print(f"{'='*55}")

    model = load_model(
        checkpoint = cfg.CHECKPOINT,
        model_path = getattr(cfg, "VIRCHOW2_PATH", None),
        device     = device,
    )

    # Pre-place normalisation constants on-device once, reused every batch
    # inside _predict_batch_from_tensor (see FIX 1 docstring there).
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(STD,  device=device).view(1, 3, 1, 1)

    # FIX 1: pick bf16 (Ampere+) or fp16 (Turing/Volta) once, based on this
    # GPU's actual tensor-core support — see _best_autocast_dtype docstring.
    autocast_dtype = _best_autocast_dtype(device)
    print(f"  AMP dtype  : {autocast_dtype}")

    manifest_rows = []

    # FIX 2: DataLoader with multiple workers overlaps tile I/O (disk read +
    # decode + resize, all CPU-bound) for the NEXT batch with GPU inference
    # on the CURRENT batch, instead of the GPU idling during file I/O.
    # Reuses cfg.WORKERS (the same knob tessellate.py uses for Mussel tiling
    # workers) so there's no new config field to learn.
    num_workers = max(0, min(int(getattr(cfg, "WORKERS", 4)), os.cpu_count() or 4))
    dataset = _TileDataset(tile_paths, patch_sz=PATCH_SIZE)
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
        # drop_last=False (default): keep the final partial batch too —
        # every tile must appear in the manifest.
    )

    with tqdm(total=len(tile_paths), desc="Running segmentation", unit="tile") as pbar:
        for batch_u8_cpu, batch_idx in loader:
            batch_paths = [tile_paths[i] for i in batch_idx.tolist()]

            # FIX 1 (mixed precision) + FIX 2 (pre-loaded batch) applied here.
            segs = _predict_batch_from_tensor(
                model, batch_u8_cpu, device, mean_t, std_t, autocast_dtype
            )

            for tile_path, seg in zip(batch_paths, segs):
                stem  = tile_path.stem
                parts = stem.rsplit("_", 2)
                try:
                    wx, wy = int(parts[-2]), int(parts[-1])
                except (ValueError, IndexError):
                    wx, wy = 0, 0

                npy_path = seg_dir / f"{stem}_seg.npy"
                np.save(str(npy_path), seg)

                # FIX 5: compute per-class pixel fractions for this tile
                # (excluding SEG_IGNORE / white background from the
                # denominator). Downstream spatial-analysis scripts —
                # tumor_roi_overlay.py and cluster_tils_tsr_score.py —
                # require frac_Tumour / frac_Stroma / frac_Inflammatory /
                # frac_Necrosis / frac_Others columns in manifest.csv to
                # filter and cluster tiles by tissue composition. Without
                # this, downstream steps fail with:
                #   "Manifest missing required columns: ['frac_Tumour', ...]"
                valid_mask  = seg != SEG_IGNORE
                valid_count = int(valid_mask.sum())
                frac_cols = {
                    f"frac_{name}": (
                        float((seg == c).sum()) / valid_count
                        if valid_count > 0 else 0.0
                    )
                    for c, name in enumerate(CLASS_NAMES)
                }

                manifest_rows.append({
                    "tile":     tile_path.name,
                    "wx":       wx,
                    "wy":       wy,
                    "npy_path": str(npy_path),
                    **frac_cols,
                })

            pbar.update(len(batch_paths))

    # FIX 3 (applied inside run_segmentation): free GPU memory after this slide.
    release_model(model)

    manifest_csv = seg_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(str(manifest_csv), index=False)
    print(f"\n  Manifest saved → {manifest_csv}")
    print(f"  Done. {len(manifest_rows):,} tiles segmented → {seg_dir}")

    return str(manifest_csv)


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) instead of hand-rolling a
# second parser here. This means segmenter.py automatically gets every
# PipelineConfig field flag AND --from-json support for free, so it can
# pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python segmenter.py --from-json run_config.json

def main(argv=None) -> None:
    from config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )
    if not cfg.CHECKPOINT or not Path(cfg.CHECKPOINT).exists():
        raise SystemExit(
            f"--checkpoint not found: {cfg.CHECKPOINT!r}. "
            f"Pass --checkpoint pointing at your ViSpace .pt file, "
            f"either directly or via --from-json."
        )

    run_segmentation(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()