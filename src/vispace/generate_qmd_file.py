#!/usr/bin/env python3
"""
report.py
=========

Generate a ViSpace Quarto pathology report for one whole-slide image.

Output directory
----------------

    cfg.OUT_DIR/<slide_name>/report/
        vispace_report_slide.qmd
        vispace_report_style.scss

Usage as a library
------------------

    from vispace.report import generate_report
    from vispace import cfg

    report_path = generate_report(
        wsi_path=cfg.WSI_PATH,
        cfg=cfg,
    )

Usage from the command line
---------------------------

This module shares its CLI with config.py.

    # Minimal
    python report.py --wsi-path slides/TCGA-A1-A0SP.svs

    # Custom output root
    python report.py \
        --wsi-path slides/TCGA-A1-A0SP.svs \
        --out-dir vispace_output

    # Continue from a saved configuration
    python report.py --from-json run_config.json

    # Override a saved value
    python report.py \
        --from-json run_config.json \
        --out-dir another_output

    # Display available flags
    python report.py --help
"""

from __future__ import annotations

import json
import re
import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def py_value(value) -> str:
    """
    Convert a value into valid Python source code.
    """
    return repr(value)


def yaml_value(value) -> str:
    """
    Convert a value into a safely quoted YAML scalar.
    """
    return json.dumps(value)


def build_parameter_cell(
    wsi_path: str,
    out_dir: str,
) -> str:
    """
    Build the Quarto Python parameter cell.
    """
    return f"""```{{python}}
#| tags: [parameters]

from pathlib import Path

wsi_path = {py_value(wsi_path)}
wsi = Path(wsi_path)
slide_name = wsi.stem
out_dir = {py_value(out_dir)}
```"""


def update_yaml_params(
    qmd_text: str,
    wsi_path: str,
    out_dir: str,
) -> str:
    """
    Insert or replace the params block in Quarto YAML front matter.
    """
    params_block = (
        "params:\n"
        f"  wsi_path: {yaml_value(wsi_path)}\n"
        f"  out_dir: {yaml_value(out_dir)}\n"
    )

    front_matter_pattern = r"\A---\r?\n(.*?)\r?\n---\r?\n"

    match = re.search(
        front_matter_pattern,
        qmd_text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Could not find YAML front matter in QMD template."
        )

    front_matter = match.group(1)

    params_pattern = (
        r"(?ms)"
        r"^params:\s*\n"
        r"(?:^[ \t]+.*(?:\n|$))*"
    )

    if re.search(params_pattern, front_matter):
        new_front_matter = re.sub(
            params_pattern,
            params_block,
            front_matter,
            count=1,
        )
    else:
        new_front_matter = (
            front_matter.rstrip()
            + "\n"
            + params_block
        )

    return (
        "---\n"
        + new_front_matter.rstrip()
        + "\n---\n"
        + qmd_text[match.end():]
    )


def replace_parameter_cell(
    qmd_text: str,
    parameter_cell: str,
) -> str:
    """
    Replace the tagged Quarto Python parameters cell.
    """
    parameter_cell_pattern = (
        r"```{python}\s*\n"
        r"#\|\s*tags:\s*\[parameters\]\s*\n"
        r".*?"
        r"```"
    )

    if not re.search(
        parameter_cell_pattern,
        qmd_text,
        flags=re.DOTALL,
    ):
        raise ValueError(
            "Could not find Python parameters cell in QMD template. "
            "Expected a cell containing '#| tags: [parameters]'."
        )

    return re.sub(
        parameter_cell_pattern,
        lambda _: parameter_cell,
        qmd_text,
        count=1,
        flags=re.DOTALL,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    wsi_path: str,
    cfg: Optional[PipelineConfig] = None,
    template_qmd: Optional[str] = None,
) -> Path:
    """
    Generate the ViSpace QMD pathology report.

    The report is written to:

        cfg.OUT_DIR/<slide_name>/report/vispace_report_slide.qmd

    When the bundled template is used, its SCSS stylesheet is copied beside
    the generated QMD file.

    The function creates the QMD report source only. It does not invoke Quarto
    to render HTML or PDF output.

    Parameters
    ----------
    wsi_path : str
        Path to the input whole-slide image.

    cfg : PipelineConfig, optional
        Pipeline configuration. Defaults to PipelineConfig().

    template_qmd : str, optional
        Path to a custom QMD template. When omitted, the bundled ViSpace
        report template is used.

    Returns
    -------
    Path
        Absolute path to the generated QMD file.
    """
    if cfg is None:
        cfg = PipelineConfig()

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

    slide_name = wsi.stem

    pipeline_out_dir = Path(
        getattr(cfg, "OUT_DIR", "") or "vispace_output"
    ).expanduser()

    report_dir = (
        pipeline_out_dir
        / slide_name
        / "report"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load report template
    # ------------------------------------------------------------------

    using_bundled_template = template_qmd is None

    if using_bundled_template:
        template_resource = files("vispace").joinpath(
            "assets",
            "report",
            "vispace_report_template.qmd",
        )

        if not template_resource.is_file():
            raise FileNotFoundError(
                "Bundled ViSpace report template was not found: "
                "assets/report/vispace_report_template.qmd"
            )

        qmd_text = template_resource.read_text(
            encoding="utf-8",
        )

    else:
        template_path = Path(template_qmd).expanduser()

        if not template_path.exists():
            raise FileNotFoundError(
                f"Report template not found: {template_path}"
            )

        if not template_path.is_file():
            raise FileNotFoundError(
                f"Report template path is not a file: "
                f"{template_path}"
            )

        qmd_text = template_path.read_text(
            encoding="utf-8",
        )

    # Use absolute paths in the generated report so rendering does not
    # depend on the caller's current working directory.
    resolved_wsi_path = str(wsi.resolve())
    resolved_out_dir = str(pipeline_out_dir.resolve())

    # ------------------------------------------------------------------
    # Update template parameters
    # ------------------------------------------------------------------

    qmd_text = update_yaml_params(
        qmd_text=qmd_text,
        wsi_path=resolved_wsi_path,
        out_dir=resolved_out_dir,
    )

    parameter_cell = build_parameter_cell(
        wsi_path=resolved_wsi_path,
        out_dir=resolved_out_dir,
    )

    qmd_text = replace_parameter_cell(
        qmd_text=qmd_text,
        parameter_cell=parameter_cell,
    )

    # ------------------------------------------------------------------
    # Write generated QMD
    # ------------------------------------------------------------------

    output_path = (
        report_dir
        / "vispace_report_slide.qmd"
    )

    output_path.write_text(
        qmd_text,
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Copy bundled stylesheet
    # ------------------------------------------------------------------
    #
    # The bundled template references:
    #
    #     theme: [cosmo, vispace_report_style.scss]
    #
    # Copy the stylesheet next to the generated QMD so Quarto can resolve
    # it regardless of the directory from which Quarto is invoked.
    #
    # Do not overwrite a stylesheet already supplied by the user.

    if using_bundled_template:
        style_name = "vispace_report_style.scss"

        style_resource = files("vispace").joinpath(
            "assets",
            "report",
            style_name,
        )

        style_dst = report_dir / style_name

        if not style_resource.is_file():
            raise FileNotFoundError(
                "Bundled ViSpace report stylesheet was not found: "
                f"assets/report/{style_name}"
            )

        if not style_dst.exists():
            with as_file(style_resource) as style_src:
                shutil.copyfile(
                    style_src,
                    style_dst,
                )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("  ViSpace Report")
    print(f"  Slide         : {wsi.name}")
    print(f"  Slide path    : {wsi.resolve()}")
    print(f"  Report file   : {output_path.resolve()}")
    print(f"  Output folder : {report_dir.resolve()}")
    print(f"{'=' * 60}\n")

    return output_path.resolve()


# =========================================================================
# CLI entry point
# =========================================================================

def main(argv=None) -> None:
    """
    Generate a report using arguments handled by config.py.
    """
    from .config import config_from_args

    cfg, _ = config_from_args(argv)

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required. Provide a path to a whole-slide "
            "image directly or through --from-json."
        )

    generate_report(
        wsi_path=cfg.WSI_PATH,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()