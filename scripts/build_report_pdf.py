from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "report" / "Report.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "Report.pdf"
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
    parser = argparse.ArgumentParser(description="Build report/Report.pdf from report/Report.md.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "ProjectTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            spaceAfter=18,
        ),
        "Heading2": ParagraphStyle(
            "ProjectHeading2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "Heading3": ParagraphStyle(
            "ProjectHeading3",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "ProjectBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=7,
        ),
        "Bullet": ParagraphStyle(
            "ProjectBullet",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "Code": ParagraphStyle(
            "ProjectCode",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f2f2f2"),
            borderColor=colors.HexColor("#dddddd"),
            borderWidth=0.5,
            borderPadding=6,
            spaceAfter=8,
        ),
        "Caption": ParagraphStyle(
            "ProjectCaption",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            alignment=1,
            textColor=colors.HexColor("#333333"),
            spaceBefore=4,
            spaceAfter=8,
        ),
    }
    return styles


def clean_inline_markdown(text: str) -> str:
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", escaped)
    return escaped


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def add_markdown_table(story: list, lines: list[str], styles: dict[str, ParagraphStyle], width: float) -> None:
    rows: list[list[Paragraph]] = []
    for line in lines:
        if is_table_separator(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append([Paragraph(clean_inline_markdown(cell), styles["Body"]) for cell in cells])

    if not rows:
        return

    col_width = width / len(rows[0])
    table = Table(rows, colWidths=[col_width] * len(rows[0]), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f5d62")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8b8b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8))


def add_markdown_line(story: list, line: str, styles: dict[str, ParagraphStyle]) -> None:
    stripped = line.strip()
    if not stripped:
        story.append(Spacer(1, 5))
        return

    if stripped.startswith("# "):
        story.append(Paragraph(clean_inline_markdown(stripped[2:]), styles["Title"]))
    elif stripped.startswith("## "):
        story.append(Paragraph(clean_inline_markdown(stripped[3:]), styles["Heading2"]))
    elif stripped.startswith("### "):
        story.append(Paragraph(clean_inline_markdown(stripped[4:]), styles["Heading3"]))
    elif stripped.startswith("- "):
        story.append(Paragraph(clean_inline_markdown(stripped[2:]), styles["Bullet"], bulletText="-"))
    else:
        story.append(Paragraph(clean_inline_markdown(stripped), styles["Body"]))


def add_figures(story: list, styles: dict[str, ParagraphStyle], max_width: float) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Figures", styles["Heading2"]))

    for figure_name, caption in FIGURES:
        figure_path = FIGURES_PATH / figure_name
        if not figure_path.exists() or figure_path.stat().st_size == 0:
            continue

        image = Image(str(figure_path))
        scale = min(max_width / image.drawWidth, 4.8 * inch / image.drawHeight, 1)
        image.drawWidth *= scale
        image.drawHeight *= scale
        story.append(Paragraph(caption, styles["Caption"]))
        story.append(image)
        story.append(Spacer(1, 12))


def add_page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(document.pagesize[0] - 0.65 * inch, 0.45 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_story(input_path: Path, styles: dict[str, ParagraphStyle], width: float) -> list:
    story: list = []
    lines = input_path.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            if in_code:
                code_text = "<br/>".join(html.escape(code_line) for code_line in code_lines)
                story.append(Paragraph(code_text, styles["Code"]))
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
            add_markdown_table(story, table_lines, styles, width)
            continue

        add_markdown_line(story, line, styles)
        index += 1

    add_figures(story, styles, width)
    return story


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Missing report markdown: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="F1 Top-10 Finish Prediction with Machine Learning",
    )
    styles = make_styles()
    story = build_story(input_path, styles, document.width)
    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
