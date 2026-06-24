"""
segmenter.py
============
Step 2 of the ViP-SegD pipeline.
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

Usage
-----
    from segmenter import run_segmentation
    from config import cfg

    run_segmentation(
        wsi_path = "slides/TCGA-A1-A0SP.svs",
        cfg      = cfg,
    )
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config import cfg as default_cfg, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (shared with stitch.py and downstream scripts)
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE  = 224
N_CLASSES   = 5
SEG_IGNORE  = 255
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]

COLORS_BGR = [
    (  0,   0, 255),   # 0 Tumour        — pure red
    (  0, 200,   0),   # 1 Stroma        — pure green
    (255, 100,   0),   # 2 Inflammatory  — bright blue
    (  0, 165, 255),   # 3 Necrosis      — orange
    (220,   0, 220),   # 4 Others        — magenta
]
COLORS_RGB = [(r / 255, g / 255, b / 255) for b, g, r in COLORS_BGR]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODER  — frozen Virchow2 ViT-H/14
# ─────────────────────────────────────────────────────────────────────────────

def load_virchow2(model_path: str = None) -> nn.Module:
    """
    Load Virchow2 ViT-H/14 encoder.

    Parameters
    ----------
    model_path : local path (optional). If None, downloads from HuggingFace.
                 Requires: huggingface-cli login (one-time)

    Returns
    -------
    Frozen nn.Module on DEVICE.
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
    enc = enc.to(DEVICE)

    # Smoke test
    with torch.no_grad():
        out = enc.forward_features(
            torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=DEVICE))
    assert out.shape == (1, 261, 1280), \
        f"Unexpected Virchow2 output shape: {out.shape}"

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Virchow2: {n_params/1e9:.2f}B params | frozen | {DEVICE}")
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
) -> BCSSSegmenter:
    """
    Load BCSSSegmenter from a saved checkpoint.

    Parameters
    ----------
    checkpoint  : path to phaseA_best.pt or phaseB_best.pt
    model_path  : local Virchow2 weights (None = download from HuggingFace)

    Returns
    -------
    BCSSSegmenter in eval mode on DEVICE.
    """
    print(f"\n  Loading checkpoint: {Path(checkpoint).name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(str(checkpoint), map_location=DEVICE,
                          weights_only=False)

    n_classes = ckpt.get("n_classes", N_CLASSES)
    encoder   = load_virchow2(model_path)
    model     = BCSSSegmenter(encoder, n_classes=n_classes).to(DEVICE)

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
    print(f"  epoch={epoch}  macro_dice={dice}  device={DEVICE}")

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

def _preprocess_tile(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Normalise a (H, W, 3) uint8 RGB tile and return a (1, 3, H, W) tensor.
    Uses ImageNet mean/std — same as Virchow2 pre-training.
    """
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def _predict_tile(
    model:    BCSSSegmenter,
    img_rgb:  np.ndarray,
    patch_sz: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Run model on a single tile.

    Parameters
    ----------
    model   : BCSSSegmenter in eval mode
    img_rgb : (H, W, 3) uint8 RGB

    Returns
    -------
    (H, W) int32 class map with white background masked to SEG_IGNORE.
    """
    from PIL import Image as _Image
    # Resize to model input size if needed
    if img_rgb.shape[:2] != (patch_sz, patch_sz):
        img_rgb = np.array(
            _Image.fromarray(img_rgb).resize(
                (patch_sz, patch_sz), _Image.LANCZOS))

    x    = _preprocess_tile(img_rgb)
    logits = model(x)                           # (1, C, H, W)
    seg  = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    seg  = mask_white_background(img_rgb, seg)
    return seg


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

    print(f"\n{'='*55}")
    print(f"  Segmentation")
    print(f"  Slide      : {slide_name}")
    print(f"  Tiles      : {len(tile_paths):,}")
    print(f"  Checkpoint : {Path(cfg.CHECKPOINT).name}")
    print(f"  Output     : {seg_dir}")
    print(f"{'='*55}")

    model = load_model(
        checkpoint = cfg.CHECKPOINT,
        model_path = getattr(cfg, "VIRCHOW2_PATH", None),
    )

    manifest_rows = []

    for tile_path in tqdm(tile_paths, desc="Running segmentation", unit="tile"):
        # Parse WSI-level coordinates from filename: <slide>_<wx>_<wy>.png
        # Falls back to (0, 0) if filename does not carry coordinates.
        stem  = tile_path.stem          # e.g. "TCGA-A1-A0SP_4096_8192"
        parts = stem.rsplit("_", 2)
        try:
            wx, wy = int(parts[-2]), int(parts[-1])
        except (ValueError, IndexError):
            wx, wy = 0, 0

        img_rgb = np.array(Image.open(tile_path).convert("RGB"))
        seg     = _predict_tile(model, img_rgb)

        npy_path = seg_dir / f"{tile_path.stem}_seg.npy"
        np.save(str(npy_path), seg)

        manifest_rows.append({
            "tile":     tile_path.name,
            "wx":       wx,
            "wy":       wy,
            "npy_path": str(npy_path),
        })

    manifest_csv = seg_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(str(manifest_csv), index=False)
    print(f"\n  Manifest saved → {manifest_csv}")
    print(f"  Done. {len(manifest_rows):,} tiles segmented → {seg_dir}")

    return str(manifest_csv)