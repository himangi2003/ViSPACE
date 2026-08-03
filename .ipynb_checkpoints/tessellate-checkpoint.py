"""
tessellate.py
=============
Step 1 of the ViSpace pipeline.

Runs Mussel tiling on one whole-slide image using settings from config.py.

Mussel must be installed in the active Python environment, for example:

    pip install "mussel-pathology[torch-gpu]"

Output directory
----------------
    cfg.OUT_DIR/<slide_name>/tessellation/
        patches/
        mask.png
        grid_mask.png
        thumbnail.png
        <slide>.h5

Usage as a library
------------------
    from tessellate import run_tessellation
    from config import cfg

    tess_dir = run_tessellation(
        wsi_path=cfg.WSI_PATH,
        cfg=cfg,
    )

Usage from the command line
---------------------------
tessellate.py shares its CLI with config.py. Relevant PipelineConfig fields,
including PATCH_SIZE, WORKERS, SEGMENT_THRESH, THUMBNAIL_SIZE, OUT_DIR,
and WSI_PATH, are available as command-line flags.

    # Minimal
    python tessellate.py --wsi-path slides/TCGA-A1-A0SP.svs

    # Custom settings
    python tessellate.py \
        --wsi-path slides/TCGA-A1-A0SP.svs \
        --patch-size 224 \
        --workers 8 \
        --segment-thresh 20 \
        --out-dir vipsegd_output

    # Continue from a saved configuration
    python tessellate.py --from-json run_config.json

    # Override one saved field
    python tessellate.py \
        --from-json run_config.json \
        --patch-size 256

    # Display available flags
    python tessellate.py --help
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Tuple, Union

from omegaconf import OmegaConf

from config import PipelineConfig
from config import cfg as default_cfg


ThumbnailSize = Union[int, Tuple[int, int]]


def _load_mussel_tessellate():
    """
    Import Mussel's tessellation module and configuration classes.

    Returns
    -------
    tuple
        tessellate_module, TessellateConfig, SegConfig

    Raises
    ------
    ImportError
        If mussel-pathology is not installed or cannot be imported.
    """
    try:
        import mussel.cli.tessellate as tessellate_module
        from mussel.cli.tessellate import SegConfig, TessellateConfig
    except ImportError as exc:
        raise ImportError(
            "Mussel could not be imported.\n\n"
            "Install it in the active environment with:\n"
            '  pip install "mussel-pathology[torch-gpu]"\n\n'
            "In Google Colab, restart the runtime after installation."
        ) from exc

    return tessellate_module, TessellateConfig, SegConfig


def _get_mussel_version() -> str:
    """
    Return the installed mussel-pathology distribution version.
    """
    try:
        return version("mussel-pathology")
    except PackageNotFoundError:
        return "unknown"


def _validate_thumbnail_size(value) -> ThumbnailSize:
    """
    Validate and normalize THUMBNAIL_SIZE.

    Supported formats
    -----------------
    int
        A single positive integer.

    tuple/list
        Two positive integers representing width and height.

    Examples
    --------
    1024

    (1024, 1024)

    [1024, 768]

    Parameters
    ----------
    value
        Value from cfg.THUMBNAIL_SIZE.

    Returns
    -------
    int or tuple[int, int]
        Validated and normalized thumbnail size.

    Raises
    ------
    TypeError
        If the value is not an integer or a two-element sequence.
    ValueError
        If dimensions are missing, non-integer, or non-positive.
    """
    if isinstance(value, bool):
        raise TypeError(
            "THUMBNAIL_SIZE cannot be a Boolean value. "
            f"Got {value!r}."
        )

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                "THUMBNAIL_SIZE must be greater than zero. "
                f"Got {value}."
            )

        return value

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                "THUMBNAIL_SIZE must contain exactly two values: "
                "(width, height). "
                f"Got {value!r}."
            )

        width, height = value

        if isinstance(width, bool) or isinstance(height, bool):
            raise TypeError(
                "THUMBNAIL_SIZE dimensions must be integers, not Boolean "
                f"values. Got {value!r}."
            )

        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError(
                "THUMBNAIL_SIZE width and height must be integers. "
                f"Got {value!r}."
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                "THUMBNAIL_SIZE width and height must be greater than zero. "
                f"Got {value!r}."
            )

        return width, height

    raise TypeError(
        "THUMBNAIL_SIZE must be either a positive integer or a "
        "two-element tuple/list such as (1024, 1024). "
        f"Got {type(value).__name__}: {value!r}"
    )


def _count_patch_images(patches_dir: Path) -> int:
    """
    Count patch image files in the Mussel patch output directory.
    """
    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
    }

    if not patches_dir.exists():
        return 0

    return sum(
        1
        for path in patches_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    )


def run_tessellation(
    wsi_path: str,
    cfg: Optional[PipelineConfig] = None,
) -> str:
    """
    Run Mussel tiling on a single whole-slide image.

    Parameters
    ----------
    wsi_path
        Path to the input whole-slide image, such as an SVS, SCN, TIF,
        or TIFF file.

    cfg
        Pipeline configuration. Defaults to the config.cfg singleton.

    Returns
    -------
    str
        Path to:

        cfg.OUT_DIR/<slide_name>/tessellation/

    Raises
    ------
    FileNotFoundError
        If the input slide does not exist.

    ValueError
        If configuration values are invalid.

    ImportError
        If mussel-pathology cannot be imported.

    RuntimeError
        If Mussel fails or does not create the expected HDF5 file.
    """
    if cfg is None:
        cfg = default_cfg

    if not wsi_path:
        raise ValueError("wsi_path cannot be empty.")

    wsi = Path(wsi_path).expanduser()

    if not wsi.exists():
        raise FileNotFoundError(
            f"Whole-slide image not found: {wsi}"
        )

    if not wsi.is_file():
        raise FileNotFoundError(
            f"WSI path is not a file: {wsi}"
        )

    if isinstance(cfg.PATCH_SIZE, bool) or not isinstance(
        cfg.PATCH_SIZE,
        int,
    ):
        raise TypeError(
            "PATCH_SIZE must be an integer. "
            f"Got {type(cfg.PATCH_SIZE).__name__}: {cfg.PATCH_SIZE!r}"
        )

    if cfg.PATCH_SIZE <= 0:
        raise ValueError(
            "PATCH_SIZE must be greater than zero. "
            f"Got {cfg.PATCH_SIZE}."
        )

    if isinstance(cfg.WORKERS, bool) or not isinstance(cfg.WORKERS, int):
        raise TypeError(
            "WORKERS must be an integer. "
            f"Got {type(cfg.WORKERS).__name__}: {cfg.WORKERS!r}"
        )

    if cfg.WORKERS < 0:
        raise ValueError(
            "WORKERS must be zero or greater. "
            f"Got {cfg.WORKERS}."
        )

    if not isinstance(cfg.SEGMENT_THRESH, (int, float)):
        raise TypeError(
            "SEGMENT_THRESH must be numeric. "
            f"Got {type(cfg.SEGMENT_THRESH).__name__}: "
            f"{cfg.SEGMENT_THRESH!r}"
        )

    thumbnail_size = _validate_thumbnail_size(
        cfg.THUMBNAIL_SIZE
    )

    tessellate_module, TessellateConfig, SegConfig = (
        _load_mussel_tessellate()
    )

    slide_name = wsi.stem

    outdir = (
        Path(cfg.OUT_DIR).expanduser()
        / slide_name
        / "tessellation"
    )

    patches_dir = outdir / "patches"

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patches_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_h5_path = outdir / f"{slide_name}.h5"
    output_mask_path = outdir / "mask.png"
    output_grid_mask_path = outdir / "grid_mask.png"
    output_thumbnail_path = outdir / "thumbnail.png"

    print(f"\n{'=' * 60}")
    print("  ViSpace Tessellation")
    print(f"  Slide          : {wsi.name}")
    print(f"  Slide path     : {wsi.resolve()}")
    print(f"  Mussel version : {_get_mussel_version()}")
    print(f"  Patch size     : {cfg.PATCH_SIZE}")
    print(f"  Workers        : {cfg.WORKERS}")
    print(f"  Segment thresh : {cfg.SEGMENT_THRESH}")
    print(f"  Thumbnail size : {thumbnail_size}")
    print(f"  Output         : {outdir.resolve()}")
    print(f"{'=' * 60}")

    seg_config = SegConfig(
        patch_size=cfg.PATCH_SIZE,
        use_otsu=True,
        segment_threshold=cfg.SEGMENT_THRESH,
    )

    tess_config = TessellateConfig(
        slide_path=str(wsi.resolve()),
        output_h5_path=str(output_h5_path.resolve()),
        output_png_dir=str(patches_dir.resolve()),
        output_mask_path=str(output_mask_path.resolve()),
        output_grid_mask_path=str(
            output_grid_mask_path.resolve()
        ),
        output_thumbnail_path=str(
            output_thumbnail_path.resolve()
        ),
        thumbnail_size=thumbnail_size,
        seg_config=seg_config,
        num_workers=cfg.WORKERS,
    )

    print(
        "\n  Running Mussel tessellation. "
        "Large slides may take some time..."
    )

    try:
        mussel_config = OmegaConf.structured(tess_config)
    except Exception:
        # Fallback for Mussel configuration classes that are not registered
        # as structured OmegaConf dataclasses.
        mussel_config = OmegaConf.create(tess_config)

    try:
        tessellate_module.main(mussel_config)
    except Exception as exc:
        raise RuntimeError(
            "Mussel tessellation failed.\n"
            f"Slide: {wsi.resolve()}\n"
            f"Output directory: {outdir.resolve()}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    if not output_h5_path.exists():
        raise RuntimeError(
            "Tessellation finished without creating the expected HDF5 "
            "output.\n"
            f"Slide: {wsi.resolve()}\n"
            f"Expected HDF5 file: {output_h5_path.resolve()}"
        )

    n_patches = _count_patch_images(patches_dir)

    print("\n  Tessellation completed successfully.")
    print(f"  Patch images : {n_patches:,}")
    print(f"  HDF5 output  : {output_h5_path.resolve()}")
    print(f"  Output folder: {outdir.resolve()}")

    return str(outdir.resolve())


# =========================================================================
# CLI entry point
# =========================================================================

def main(argv=None) -> None:
    """
    Run tessellation using arguments handled by config.py.
    """
    from config import config_from_args

    cfg, _ = config_from_args(argv)

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required. Provide a path to a whole-slide "
            "image directly or through --from-json."
        )

    run_tessellation(
        wsi_path=cfg.WSI_PATH,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()