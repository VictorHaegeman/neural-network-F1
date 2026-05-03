from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Inches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "report" / "Report.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "Report.docx"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report/Report.docx from report/Report.md.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_markdown_line(document: Document, line: str) -> None:
    stripped = line.strip()

    if not stripped:
        return
    if stripped.startswith("# "):
        document.add_heading(stripped[2:], level=1)
    elif stripped.startswith("## "):
        document.add_heading(stripped[3:], level=2)
    elif stripped.startswith("### "):
        document.add_heading(stripped[4:], level=3)
    elif stripped.startswith("- "):
        document.add_paragraph(stripped[2:], style="List Bullet")
    elif stripped.startswith("```"):
        return
    else:
        document.add_paragraph(stripped.replace("`", ""))


def add_figures(document: Document) -> None:
    figure_names = [
        "target_distribution.png",
        "rows_by_season.png",
        "top10_rate_by_season.png",
        "model_comparison.png",
        "neural_network_tuning.png",
        "position_model_comparison.png",
        "rolling_backtest.png",
        "feature_importance.png",
        "confusion_matrix.png",
    ]

    document.add_heading("Figures", level=2)
    for figure_name in figure_names:
        figure_path = FIGURES_PATH / figure_name
        if not figure_path.exists() or figure_path.stat().st_size == 0:
            continue

        document.add_paragraph(figure_name)
        document.add_picture(str(figure_path), width=Inches(5.8))


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Missing report markdown: {input_path}")

    document = Document()
    for line in input_path.read_text(encoding="utf-8").splitlines():
        add_markdown_line(document, line)

    add_figures(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
