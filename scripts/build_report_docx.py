from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "report" / "Report.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "Report.docx"
FIGURES_PATH = PROJECT_ROOT / "outputs" / "figures"

PRIMARY = "123C43"
SECONDARY = "D8A31A"
LIGHT = "F5F2E8"
PALE_BLUE = "EAF3F4"
ROW_ALT = "F7F7F7"

FIGURES = [
    ("assignment_pipeline_overview.png", "Figure 1. End-to-end project pipeline from raw data sources to validated assignment outputs."),
    ("target_distribution.png", "Figure 2. Target distribution for the binary top-10 classification task."),
    ("rows_by_season.png", "Figure 3. Number of driver-race rows available per season."),
    ("top10_rate_by_season.png", "Figure 4. Top-10 target rate by season, used to check class stability over time."),
    ("grid_vs_finish.png", "Figure 5. Relationship between starting grid and final position."),
    ("algorithm_holdout_summary.png", "Figure 6. Holdout comparison of the main classification algorithms on the 2025 season."),
    ("model_metrics_table.png", "Figure 7. Validation metric table for the compared classification algorithms."),
    ("model_comparison.png", "Figure 8. Compact model comparison across F1, ROC-AUC and race precision@10."),
    ("algorithm_validation_summary.png", "Figure 9. Expanding-window validation showing model stability across seasons."),
    ("rolling_backtest.png", "Figure 10. Rolling backtest race precision@10 by model and test season."),
    ("feature_importance.png", "Figure 11. Most influential features for the selected top-10 classifier."),
    ("confusion_matrix.png", "Figure 12. Confusion matrix for the selected holdout classifier."),
    ("neural_network_tuning.png", "Figure 13. Neural-network classifier tuning results, included as an extension."),
    ("neural_network_embedding_3d.png", "Figure 14. Static view of the interactive 3D neural-network hidden-space embedding."),
    ("position_model_comparison.png", "Figure 15. Finish-position ranking model comparison, included as an extension."),
    ("predictions/model_precision_by_race.png", "Figure 16. Correct predicted top-10 drivers per race and model on the 2025 holdout season."),
    ("predictions/model_hit_heatmap.png", "Figure 17. Race-level heatmap of top-10 hits by model."),
    ("predictions/model_points_captured.png", "Figure 18. Average actual points captured by each model's predicted top 10."),
    ("predictions/race_overviews/2025_08_monaco_grand_prix.png", "Figure 19. Example race overview with real result, model scoreboard, consensus picks and predicted top 10 chips."),
    ("predictions/race_cards/2025_08_monaco_grand_prix.png", "Figure 20. Detailed race card showing actual podium, virtual podiums and predicted top 10 lists."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report/Report.docx from report/Report.md.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_text(cell, text: str, *, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Arial"
    run.font.size = Pt(9)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    styles = document.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(12)
    styles["Normal"].paragraph_format.line_spacing = 1.15
    styles["Heading 1"].font.name = "Arial"
    styles["Heading 1"].font.size = Pt(18)
    styles["Heading 1"].font.bold = True
    styles["Heading 1"].font.color.rgb = RGBColor.from_string(PRIMARY)
    styles["Heading 2"].font.name = "Arial"
    styles["Heading 2"].font.size = Pt(15)
    styles["Heading 2"].font.bold = True
    styles["Heading 2"].font.color.rgb = RGBColor.from_string(PRIMARY)
    styles["Heading 3"].font.name = "Arial"
    styles["Heading 3"].font.size = Pt(13)
    styles["Heading 3"].font.bold = True
    styles["Heading 3"].font.color.rgb = RGBColor.from_string(PRIMARY)

    footer = section.footer.paragraphs[0]
    footer.text = "F1 Top-10 Finish Prediction | CX016-2.5-3-IML"
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor.from_string("666666")


def add_cover_page(document: Document) -> None:
    for _ in range(3):
        document.add_paragraph()

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("F1 Top-10 Finish Prediction\nwith Machine Learning")
    run.bold = True
    run.font.name = "Arial"
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor.from_string(PRIMARY)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle.add_run("Introduction to Machine Learning - Group Assignment")
    subtitle_run.font.name = "Arial"
    subtitle_run.font.size = Pt(14)
    subtitle_run.font.color.rgb = RGBColor.from_string("555555")

    document.add_paragraph()
    table = document.add_table(rows=5, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    metadata = [
        ("Module", "CX016-2.5-3-IML - Introduction to Machine Learning"),
        ("Deliverable", "Report, notebook, dataset, scripts and submission ZIP"),
        ("Main task", "Predict whether each Formula 1 driver finishes in the top 10"),
        ("Dataset", "6,999 driver-race rows, 198 variables, seasons 2010-2026"),
        ("Primary model", "Histogram Gradient Boosting classifier"),
    ]
    for row_index, (label, value) in enumerate(metadata):
        set_cell_shading(table.cell(row_index, 0), PRIMARY)
        set_cell_text(table.cell(row_index, 0), label, bold=True, color="FFFFFF")
        set_cell_shading(table.cell(row_index, 1), LIGHT if row_index % 2 == 0 else PALE_BLUE)
        set_cell_text(table.cell(row_index, 1), value)

    document.add_paragraph()
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note_run = note.add_run("Generated from report/Report.md using scripts/build_report_docx.py")
    note_run.font.size = Pt(9)
    note_run.font.color.rgb = RGBColor.from_string("666666")

    document.add_page_break()


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
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            if row_index == 0:
                set_cell_shading(cell, PRIMARY)
                set_cell_text(cell, value, bold=True, color="FFFFFF")
            else:
                set_cell_shading(cell, PALE_BLUE if row_index % 2 else ROW_ALT)
                set_cell_text(cell, value)
    document.add_paragraph()


def add_code_block(document: Document, lines: list[str]) -> None:
    for line in lines:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.2)
        run = paragraph.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string("333333")


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
        document.add_paragraph(stripped[2:].replace("`", ""), style="List Bullet")
    elif len(stripped) > 3 and stripped[0].isdigit() and ". " in stripped[:4]:
        document.add_paragraph(stripped.split(". ", 1)[1].replace("`", ""), style="List Number")
    else:
        paragraph = document.add_paragraph(stripped.replace("`", ""))
        paragraph.paragraph_format.line_spacing = 1.15


def add_figures(document: Document) -> None:
    document.add_page_break()
    document.add_heading("Figures", level=2)
    for figure_name, caption in FIGURES:
        figure_path = FIGURES_PATH / figure_name
        if not figure_path.exists() or figure_path.stat().st_size == 0:
            continue

        caption_paragraph = document.add_paragraph()
        caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption_run = caption_paragraph.add_run(caption)
        caption_run.italic = True
        caption_run.font.size = Pt(9)
        caption_run.font.color.rgb = RGBColor.from_string("333333")
        document.add_picture(str(figure_path), width=Inches(6.2))


def add_markdown_body(document: Document, lines: list[str]) -> None:
    index = 0
    in_code = False
    code_lines: list[str] = []
    skipped_title = False

    while index < len(lines):
        line = lines[index]

        if not skipped_title and line.startswith("# "):
            skipped_title = True
            index += 1
            continue

        if line.strip().startswith("```"):
            if in_code:
                add_code_block(document, code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if is_table_line(line):
            table_lines = []
            while index < len(lines) and is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1
            add_markdown_table(document, table_lines)
            continue

        add_markdown_line(document, line)
        index += 1


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Missing report markdown: {input_path}")

    document = Document()
    configure_document(document)
    add_cover_page(document)
    add_markdown_body(document, input_path.read_text(encoding="utf-8").splitlines())
    add_figures(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
