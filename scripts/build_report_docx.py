from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "report" / "Report.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "Report.docx"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"

FIGURES = [
    (
        "assignment_pipeline_overview.png",
        "Figure 1. End-to-end project pipeline from raw data sources to validated assignment outputs.",
    ),
    (
        "target_distribution.png",
        "Figure 2. Target distribution for the binary top-10 classification task.",
    ),
    (
        "rows_by_season.png",
        "Figure 3. Number of driver-race rows available per season.",
    ),
    (
        "top10_rate_by_season.png",
        "Figure 4. Top-10 target rate by season, used to check class stability over time.",
    ),
    (
        "grid_vs_finish.png",
        "Figure 5. Relationship between starting grid and final position.",
    ),
    (
        "algorithm_holdout_summary.png",
        "Figure 6. Holdout comparison of the main classification algorithms on the 2025 season.",
    ),
    (
        "model_metrics_table.png",
        "Figure 7. Validation metric table for the compared classification algorithms.",
    ),
    (
        "model_comparison.png",
        "Figure 8. Compact model comparison across F1, ROC-AUC and race precision@10.",
    ),
    (
        "algorithm_validation_summary.png",
        "Figure 9. Expanding-window validation showing model stability across seasons.",
    ),
    (
        "rolling_backtest.png",
        "Figure 10. Rolling backtest race precision@10 by model and test season.",
    ),
    (
        "feature_importance.png",
        "Figure 11. Most influential features for the selected top-10 classifier.",
    ),
    (
        "confusion_matrix.png",
        "Figure 12. Confusion matrix for the selected holdout classifier.",
    ),
    (
        "neural_network_tuning.png",
        "Figure 13. Neural-network classifier tuning results, included as an extension.",
    ),
    (
        "position_model_comparison.png",
        "Figure 14. Finish-position ranking model comparison, included as an extension.",
    ),
]


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


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def add_markdown_table(document: Document, lines: list[str]) -> None:
    rows = []
    for line in lines:
        if is_table_separator(line):
            continue
        rows.append([cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")])

    if not rows:
        return

    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.text = value
            if row_index == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True


def add_figures(document: Document) -> None:
    document.add_heading("Figures", level=2)
    for figure_name, caption in FIGURES:
        figure_path = FIGURES_PATH / figure_name
        if not figure_path.exists() or figure_path.stat().st_size == 0:
            continue

        document.add_paragraph(caption)
        document.add_picture(str(figure_path), width=Inches(5.8))


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Missing report markdown: {input_path}")

    document = Document()
    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(12)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if is_table_line(line):
            table_lines = []
            while index < len(lines) and is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1
            add_markdown_table(document, table_lines)
            continue

        add_markdown_line(document, line)
        index += 1

    add_figures(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
