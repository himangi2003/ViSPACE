"""
report.py
=========
Generate a ViSpace Quarto pathology report for one whole-slide image.

Output directory
----------------
    cfg.OUT_DIR/<slide_name>/report/
        vispace_report_slide.qmd

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

from pathlib import Path
import json
import re
from typing import Optional

from .config import PipelineConfig


def py_value(value):
    """
    Convert a value into valid Python source code.
    """
    return repr(value)


def yaml_value(value):
    """
    Convert a value into a safely quoted YAML scalar.
    """
    return json.dumps(value)


def build_parameter_cell(
    wsi_path: str,
    out_dir: str,
) -> str:
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
    params_block = f"""params:
  wsi_path: {yaml_value(wsi_path)}
  out_dir: {yaml_value(out_dir)}
"""

    front_matter_pattern = r"^---\n(.*?)\n---\n"
    match = re.search(
        front_matter_pattern,
        qmd_text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Could not find YAML front matter in QMD file."
        )

    front_matter = match.group(1)
    params_pattern = r"(?ms)^params:\n(?:^[ \t]+.*\n)*"

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
    parameter_cell_pattern = (
        r"```{python}\n"
        r"#\| tags: \[parameters\]\n"
        r".*?"
        r"```"
    )

    if not re.search(
        parameter_cell_pattern,
        qmd_text,
        flags=re.DOTALL,
    ):
        raise ValueError(
            "Could not find Python parameters cell in QMD file."
        )

    return re.sub(
        parameter_cell_pattern,
        parameter_cell,
        qmd_text,
        count=1,
        flags=re.DOTALL,
    )


def generate_report(
    wsi_path: str,
    cfg: Optional[PipelineConfig] = None,
    template_qmd: Optional[str] = None,
) -> Path:
    """
    Generate the ViSpace QMD pathology report.

    The report is written to:

        cfg.OUT_DIR/<slide_name>/report/vispace_report_slide.qmd

    Only the QMD file is created. The function does not render the report.

    Parameters
    ----------
    wsi_path
        Path to the input whole-slide image.

    cfg
        Pipeline configuration. Defaults to PipelineConfig().

    template_qmd
        Optional path to a custom QMD template. When omitted, the bundled
        ViSpace report template is used.

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
        getattr(cfg, "OUT_DIR", "")
        or "vispace_output"
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

    if template_qmd is None:
        from importlib.resources import files

        template_resource = files("vispace").joinpath(
            "assets",
            "report",
            "vispace_report_template.qmd",
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

    resolved_wsi_path = str(wsi.resolve())
    resolved_out_dir = str(pipeline_out_dir.resolve())

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

    output_path = (
        report_dir
        / "vispace_report_slide.qmd"
    )

    output_path.write_text(
        qmd_text,
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("  ViSpace Report")
    print(f"  Slide         : {wsi.name}")
    print(f"  Slide path    : {wsi.resolve()}")
    print(f"  Report file   : {output_path.resolve()}")
    print(f"  Output folder : {report_dir.resolve()}")
    print(f"{'=' * 60}")

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