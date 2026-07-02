#!/usr/bin/env python3
"""
Generate a ViP-SegD HTML report for one slide.

Usage
-----
    python report/generate_report.py \
        --slide  TCGA-A1-A0SP \
        --out    vipsegd_output \
        --report-dir report/

    # With optional metadata
    python report/generate_report.py \
        --slide  TCGA-A1-A0SP \
        --out    vipsegd_output \
        --institution "Oncology Dept, General Hospital" \
        --pathologist "Dr. J. Smith" \
        --notes "TNBC, pre-treatment biopsy"

The rendered HTML is saved to:
    <report-dir>/<slide_name>_report.html
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Render a ViP-SegD Quarto report for one slide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--slide",       required=True,
                        help="Slide stem name (e.g. TCGA-A1-A0SP — no file extension).")
    parser.add_argument("--out",         default="vipsegd_output",
                        help="ViP-SegD output directory (default: vipsegd_output/).")
    parser.add_argument("--report-dir",  default="report",
                        help="Directory where the HTML report is saved (default: report/).")
    args = parser.parse_args()

    # Paths
    report_dir = Path(args.report_dir)
    qmd        = report_dir / "report.qmd"
    out_html   = report_dir / f"{args.slide}_report.html"

    if not qmd.exists():
        print(f"Error: report template not found at {qmd}", file=sys.stderr)
        sys.exit(1)

    slide_dir = Path(args.out) / args.slide
    if not slide_dir.exists():
        print(f"Error: pipeline output not found at {slide_dir}", file=sys.stderr)
        print(f"       Run run.py for this slide first.", file=sys.stderr)
        sys.exit(1)

    # Build quarto render command
    cmd = [
        "quarto", "render", str(qmd),
        "--output",    out_html.name,
        "--output-dir", str(report_dir),
        "--execute-params", (
            f"slide_name={args.slide},"
            f"out_dir={args.out}"
        ),
    ]

    print(f"\nRendering report for: {args.slide}")
    print(f"Output: {out_html}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nQuarto rendering failed. Check the error above.", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nDone. Report saved to:\n  {out_html.resolve()}")


if __name__ == "__main__":
    main()