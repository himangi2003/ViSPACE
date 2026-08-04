from pathlib import Path
import re
import json
from typing import Optional

from .config import PipelineConfig


def py_value(value):
    """
    Convert value into valid Python code.
    Example:
        TNBC -> 'TNBC'
        0.9  -> 0.9
    """
    return repr(value)


def yaml_value(value):
    """
    JSON string literals are valid YAML scalars.
    This safely handles spaces, quotes, etc.
    """
    return json.dumps(value)


def build_parameter_cell(
    wsi_path: str,
    out_dir: str
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
    match = re.search(front_matter_pattern, qmd_text, flags=re.DOTALL)

    if not match:
        raise ValueError("Could not find YAML front matter in QMD file.")

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
        new_front_matter = front_matter.rstrip() + "\n" + params_block

    updated_qmd = (
        "---\n"
        + new_front_matter.rstrip()
        + "\n---\n"
        + qmd_text[match.end():]
    )

    return updated_qmd


def replace_parameter_cell(qmd_text: str, parameter_cell: str) -> str:
    parameter_cell_pattern = (
        r"```{python}\n"
        r"#\| tags: \[parameters\]\n"
        r".*?"
        r"```"
    )

    if not re.search(parameter_cell_pattern, qmd_text, flags=re.DOTALL):
        raise ValueError("Could not find Python parameters cell in QMD file.")

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
    template_qmd: str = "vispace_report_template.qmd",
    output_qmd: Optional[str] = None,
) -> Path:
    """
    Generate a QMD pathology report file from PipelineConfig.

    This function only creates the .qmd file.
    It does not render the report.
    """

    if cfg is None:
        cfg = PipelineConfig()

    out_dir = getattr(cfg, "OUT_DIR", "") or "vispace_output"

    template_path = Path(template_qmd)
    qmd_text = template_path.read_text(encoding="utf-8")

    qmd_text = update_yaml_params(
        qmd_text=qmd_text,
        wsi_path=wsi_path,
        out_dir=out_dir
    )

    parameter_cell = build_parameter_cell(
        wsi_path=wsi_path,
        out_dir=out_dir
    )

    qmd_text = replace_parameter_cell(qmd_text, parameter_cell)

    slide_name = Path(wsi_path).stem

    if output_qmd is None:
        output_qmd = "vispace_report_slide.qmd"

    output_path = Path(output_qmd)
    output_path.write_text(qmd_text, encoding="utf-8")

    return output_path