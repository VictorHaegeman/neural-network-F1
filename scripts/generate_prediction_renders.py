from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.exceptions import ConvergenceWarning

from train_model import DATA_PATH, FIGURES_PATH, MODEL_NAMES, TARGET, build_features, build_pipeline, latest_season_split


OUTPUTS_PATH = Path("outputs")
PREDICTIONS_PATH = OUTPUTS_PATH / "predictions"
RENDER_PATH = PREDICTIONS_PATH / "race_model_renders"
FIGURE_PATH = FIGURES_PATH / "predictions"
RACE_CARD_PATH = FIGURE_PATH / "race_cards"
RACE_OVERVIEW_PATH = FIGURE_PATH / "race_overviews"
HEADSHOT_PATH = OUTPUTS_PATH / "driver_headshots"
OPENF1_BASE_URL = "https://api.openf1.org/v1"

TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "McLaren": "#FF8000",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "RB F1 Team": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "Sauber": "#52E252",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "extra_trees": "Extra\nTrees",
    "hist_gradient_boosting": "Histogram\nGradient Boosting",
    "neural_network_mlp": "Neural\nNetwork MLP",
}

MODEL_PRIORITY = {model_name: index for index, model_name in enumerate(MODEL_NAMES)}
MODEL_SCORE_COLUMNS = [
    "top10_hits",
    "predicted_top10_actual_points",
    "podium_hits",
    "midfield_top10_hits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate race-level prediction renders, podium cards and model comparison tables."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--with-headshots", action="store_true", help="Fetch OpenF1/F1 CDN driver headshots.")
    parser.add_argument("--max-races", type=int, default=None, help="Optional cap for generated race cards.")
    parser.add_argument("--skip-cards", action="store_true")
    parser.add_argument("--skip-overviews", action="store_true")
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_") or "race"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def constructor_color(name: str) -> str:
    for key, value in TEAM_COLORS.items():
        if key.lower() in str(name).lower():
            return value
    return "#44515A"


def request_json(session: requests.Session, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    response = session.get(f"{OPENF1_BASE_URL}/{endpoint}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def load_cached_headshots() -> pd.DataFrame:
    metadata = HEADSHOT_PATH / "driver_headshots_openf1.csv"
    if metadata.exists() and metadata.stat().st_size > 0:
        return pd.read_csv(metadata)
    return pd.DataFrame()


def fetch_openf1_driver_metadata(driver_codes: set[str], season: int) -> pd.DataFrame:
    HEADSHOT_PATH.mkdir(parents=True, exist_ok=True)
    metadata_path = HEADSHOT_PATH / "driver_headshots_openf1.csv"
    cached = load_cached_headshots()
    if not cached.empty and driver_codes <= set(cached["driver_code"].dropna().astype(str)):
        return cached

    session = requests.Session()
    rows: list[dict[str, Any]] = []

    def add_driver_rows(items: list[dict[str, Any]]) -> None:
        for item in items:
            code = str(item.get("name_acronym", "")).upper()
            if not code or code not in driver_codes:
                continue
            rows.append(
                {
                    "driver_code": code,
                    "full_name": item.get("full_name", ""),
                    "headshot_url": item.get("headshot_url", ""),
                    "team_name": item.get("team_name", ""),
                    "team_colour": item.get("team_colour", ""),
                    "source": "OpenF1 drivers endpoint",
                }
            )

    try:
        add_driver_rows(request_json(session, "drivers", {"session_key": "latest"}))
        sessions = request_json(session, "sessions", {"year": season, "session_name": "Race"})
        for session_row in sessions[:8]:
            if driver_codes <= {row["driver_code"] for row in rows}:
                break
            session_key = session_row.get("session_key")
            if session_key is None:
                continue
            time.sleep(2.1)
            add_driver_rows(request_json(session, "drivers", {"session_key": session_key}))
    except requests.RequestException as exc:
        print(f"OpenF1 headshot metadata unavailable: {exc}")

    fetched = pd.DataFrame(rows).drop_duplicates("driver_code", keep="last") if rows else pd.DataFrame()
    if not cached.empty:
        fetched = pd.concat([cached, fetched], ignore_index=True).drop_duplicates("driver_code", keep="last")
    if not fetched.empty:
        fetched.to_csv(metadata_path, index=False)
    return fetched


def download_headshots(metadata: pd.DataFrame) -> dict[str, Path]:
    images: dict[str, Path] = {}
    HEADSHOT_PATH.mkdir(parents=True, exist_ok=True)
    if metadata.empty:
        return images

    session = requests.Session()
    for row in metadata.to_dict(orient="records"):
        code = str(row.get("driver_code", "")).upper()
        url = str(row.get("headshot_url", ""))
        if not code or not url.startswith("http"):
            continue

        image_path = HEADSHOT_PATH / f"{code}.png"
        if image_path.exists() and image_path.stat().st_size > 0:
            images[code] = image_path
            continue

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGBA")
            image.save(image_path)
            images[code] = image_path
        except Exception as exc:
            print(f"Could not download headshot for {code}: {exc}")
    return images


def load_headshot_map(driver_codes: set[str], season: int, enabled: bool) -> dict[str, Path]:
    cached_images = {
        path.stem.upper(): path
        for path in HEADSHOT_PATH.glob("*.png")
        if path.exists() and path.stat().st_size > 0
    } if HEADSHOT_PATH.exists() else {}
    if not enabled:
        return cached_images

    metadata = fetch_openf1_driver_metadata(driver_codes, season)
    downloaded = download_headshots(metadata)
    cached_images.update(downloaded)
    return cached_images


def generate_predictions(
    df: pd.DataFrame,
    test_season: int,
    model_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = latest_season_split(df, test_season, 200)
    output_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        X_train = build_features(train_df)
        y_train = train_df[TARGET].astype(int)
        X_test = build_features(test_df)

        model = build_pipeline(X_train, model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)

        scored = test_df[
            [
                "race_id",
                "season",
                "round",
                "grand_prix",
                "driver_id",
                "driver_code",
                "driver_name",
                "constructor_name",
                "grid",
                "final_position",
                "points",
                TARGET,
            ]
        ].copy()
        scored["model"] = model_name
        scored["top10_probability"] = model.predict_proba(X_test)[:, 1]
        scored["predicted_rank"] = (
            scored.groupby("race_id")["top10_probability"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        scored["predicted_top10"] = (scored["predicted_rank"] <= 10).astype(int)
        scored["hit_top10"] = ((scored["predicted_top10"] == 1) & (scored[TARGET] == 1)).astype(int)
        scored["predicted_podium"] = (scored["predicted_rank"] <= 3).astype(int)
        scored["hit_podium"] = ((scored["predicted_podium"] == 1) & (scored["final_position"] <= 3)).astype(int)
        output_rows.append(scored)

        for (race_id, grand_prix, round_number), group in scored.groupby(["race_id", "grand_prix", "round"]):
            top10 = group[group["predicted_top10"] == 1]
            summary_rows.append(
                {
                    "race_id": race_id,
                    "season": int(group["season"].iloc[0]),
                    "round": int(round_number),
                    "grand_prix": grand_prix,
                    "model": model_name,
                    "top10_hits": int(top10["hit_top10"].sum()),
                    "race_precision_at_10": float(top10["hit_top10"].mean()) if not top10.empty else 0.0,
                    "predicted_top10_actual_points": float(top10["points"].sum()),
                    "podium_hits": int(group[group["predicted_podium"] == 1]["hit_podium"].sum()),
                    "midfield_top10_hits": int(top10[(top10["hit_top10"] == 1) & (top10["grid"] > 10)].shape[0]),
                    "front_grid_false_positives": int(top10[(top10[TARGET] == 0) & (top10["grid"] <= 10)].shape[0]),
                    "actual_top10_missed": int(group[(group[TARGET] == 1) & (group["predicted_top10"] == 0)].shape[0]),
                }
            )

    rankings = pd.concat(output_rows, ignore_index=True).sort_values(
        ["season", "round", "model", "predicted_rank"]
    )
    summary = pd.DataFrame(summary_rows)
    return rankings, add_race_explanations(summary)


def add_race_explanations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, race_group in summary.groupby("race_id"):
        race_group = race_group.copy()
        race_group["model_priority"] = race_group["model"].map(MODEL_PRIORITY).fillna(999).astype(int)
        ordered = race_group.sort_values(
            MODEL_SCORE_COLUMNS + ["model_priority"],
            ascending=[False, False, False, False, True],
        )
        best = ordered.iloc[0]
        best_score = tuple(best[column] for column in MODEL_SCORE_COLUMNS)
        best_models = ordered[
            ordered.apply(lambda row: tuple(row[column] for column in MODEL_SCORE_COLUMNS) == best_score, axis=1)
        ]["model"].astype(str).tolist()
        second = ordered.iloc[1] if len(ordered) > 1 else best

        unique_scores = (
            ordered[MODEL_SCORE_COLUMNS]
            .drop_duplicates()
            .sort_values(MODEL_SCORE_COLUMNS, ascending=[False, False, False, False])
        )
        score_to_rank = {
            tuple(row[column] for column in MODEL_SCORE_COLUMNS): rank
            for rank, (_, row) in enumerate(unique_scores.iterrows(), start=1)
        }

        for _, row in race_group.iterrows():
            row_score = tuple(row[column] for column in MODEL_SCORE_COLUMNS)
            race_rank = score_to_rank[row_score]
            if str(row["model"]) in best_models:
                result_label = "TIED BEST" if len(best_models) > 1 else "BEST"
                reason = (
                    f"{result_label.title()} race-level model: {model_label(str(row['model'])).replace(chr(10), ' ')} "
                    f"found {int(row['top10_hits'])}/10 actual top-10 drivers, captured "
                    f"{row['predicted_top10_actual_points']:.0f} actual points and got "
                    f"{int(row['podium_hits'])}/3 podium driver(s)."
                )
            else:
                gap = int(best["top10_hits"] - row["top10_hits"])
                points_gap = float(best["predicted_top10_actual_points"] - row["predicted_top10_actual_points"])
                if gap == 0:
                    if points_gap > 0:
                        reason = (
                            f"Matched the best group on top-10 hits ({int(row['top10_hits'])}/10), "
                            f"but captured {points_gap:.0f} fewer actual point(s) in its predicted top 10."
                        )
                    else:
                        reason = (
                            f"Matched the best group on top-10 hits and points, but ranked lower on podium "
                            f"or midfield tie-break indicators."
                        )
                else:
                    reason = (
                        f"Behind the best model by {gap} top-10 hit(s). It missed "
                        f"{int(row['actual_top10_missed'])} actual top-10 driver(s), while the best model "
                        f"reached {int(best['top10_hits'])}/10 hits."
                    )
                result_label = f"RANK {race_rank}"
            row_dict = row.to_dict()
            row_dict["best_model_for_race"] = best["model"]
            row_dict["best_models_for_race"] = "|".join(best_models)
            row_dict["best_model_top10_hits"] = int(best["top10_hits"])
            row_dict["runner_up_model_for_race"] = second["model"]
            row_dict["race_rank_among_models"] = int(race_rank)
            row_dict["model_result_label"] = result_label
            row_dict["race_explanation"] = reason
            row_dict.pop("model_priority", None)
            rows.append(row_dict)
    return pd.DataFrame(rows).sort_values(["season", "round", "model"])


def circular_driver_image(
    code: str,
    driver_name: str,
    image_map: dict[str, Path],
    diameter: int,
    team_color: str,
) -> Image.Image:
    if code in image_map:
        try:
            image = Image.open(image_map[code]).convert("RGBA")
            image = ImageOps.fit(image, (diameter, diameter), method=Image.Resampling.LANCZOS, centering=(0.5, 0.2))
        except Exception:
            image = Image.new("RGBA", (diameter, diameter), team_color)
    else:
        image = Image.new("RGBA", (diameter, diameter), team_color)
        draw = ImageDraw.Draw(image)
        initials = code[:3] if code else "".join(part[0] for part in driver_name.split()[:2]).upper()
        draw.text((diameter // 2, diameter // 2), initials, fill="white", anchor="mm", font=font(22, bold=True))

    mask = Image.new("L", (diameter, diameter), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
    circular = Image.new("RGBA", (diameter, diameter), (255, 255, 255, 0))
    circular.paste(image, (0, 0), mask)
    border = ImageDraw.Draw(circular)
    border.ellipse((2, 2, diameter - 3, diameter - 3), outline="white", width=4)
    return circular


def draw_rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=width)


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    size: int,
    fill: str,
    bold: bool = False,
) -> None:
    current_size = size
    while current_size >= 10:
        fnt = font(current_size, bold=bold)
        if draw.textlength(text, font=fnt) <= max_width:
            draw.text(xy, text, fill=fill, font=fnt)
            return
        current_size -= 1
    draw.text(xy, text[: max(8, max_width // 8)], fill=fill, font=font(10, bold=bold))


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    line_height: int,
    size: int,
    fill: str,
    bold: bool = False,
    max_lines: int = 3,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    fnt = font(size, bold=bold)
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=fnt) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    for index, line in enumerate(lines[:max_lines]):
        suffix = "..." if index == max_lines - 1 and len(lines) > max_lines else ""
        draw.text((xy[0], xy[1] + index * line_height), line + suffix, fill=fill, font=fnt)


def render_race_card(
    race_id: str,
    race_rankings: pd.DataFrame,
    race_summary: pd.DataFrame,
    image_map: dict[str, Path],
    output_path: Path,
) -> None:
    race_summary = race_summary.copy()
    race_summary["model_priority"] = race_summary["model"].map(MODEL_PRIORITY).fillna(999).astype(int)
    race_summary = race_summary.sort_values(
        ["race_rank_among_models", "model_priority"],
        ascending=[True, True],
    )
    models = race_summary["model"].astype(str).tolist()
    best_models = set(str(race_summary.iloc[0].get("best_models_for_race", "")).split("|"))
    race_name = str(race_rankings["grand_prix"].iloc[0])
    season = int(race_rankings["season"].iloc[0])
    round_number = int(race_rankings["round"].iloc[0])

    width, height = 2200, 1450
    image = Image.new("RGB", (width, height), "#F3F0E8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 150), fill="#123C43")
    draw.text((60, 42), f"{season} Round {round_number:02d} - {race_name}", fill="white", font=font(40, bold=True))
    draw.text(
        (60, 98),
        "Models are sorted by this race's score. Green rows finished top 10; red rows missed it; points are actual race points.",
        fill="#D8A31A",
        font=font(20),
    )

    actual = (
        race_rankings.drop_duplicates("driver_id")
        .sort_values("final_position")
        .head(3)[["driver_code", "driver_name", "constructor_name", "final_position", "points"]]
    )
    draw_rounded(draw, (60, 180, 2140, 335), "#FFFFFF", "#D8A31A", 3)
    draw.text((85, 205), "Actual podium", fill="#123C43", font=font(24, bold=True))
    for idx, (_, row) in enumerate(actual.iterrows()):
        x = 310 + idx * 540
        color = constructor_color(str(row["constructor_name"]))
        avatar = circular_driver_image(str(row["driver_code"]), str(row["driver_name"]), image_map, 88, color)
        image.paste(avatar, (x, 210), avatar)
        draw.text((x + 105, 218), f"P{int(row['final_position'])} {row['driver_code']}", fill="#111111", font=font(26, bold=True))
        draw_text_fit(draw, (x + 105, 252), str(row["driver_name"]), 340, 18, "#333333")
        draw.text((x + 105, 282), f"{row['points']:.0f} pts", fill="#666666", font=font(17))

    card_top = 380
    card_width = math.floor((width - 120 - 24 * (len(models) - 1)) / len(models))
    for index, model_name in enumerate(models):
        model_df = race_rankings[race_rankings["model"] == model_name].sort_values("predicted_rank")
        summary_row = race_summary[race_summary["model"] == model_name].iloc[0]
        result_label = str(summary_row.get("model_result_label", f"RANK {index + 1}"))
        is_best = model_name in best_models
        left = 60 + index * (card_width + 24)
        right = left + card_width
        outline = "#D8A31A" if is_best else "#D0D0D0"
        outline_width = 5 if is_best else 2
        header_fill = "#123C43" if is_best else "#263E47"
        draw_rounded(draw, (left, card_top, right, height - 55), "#FFFFFF", outline, outline_width)
        draw.rectangle((left, card_top, right, card_top + 92), fill=header_fill)
        draw.text((left + 18, card_top + 16), model_label(model_name), fill="white", font=font(22, bold=True))
        badge_fill = "#D8A31A" if is_best else "#E7E1D3"
        badge_text = "#123C43" if is_best else "#333333"
        badge_x1 = right - 122
        draw.rounded_rectangle((badge_x1, card_top + 16, right - 18, card_top + 45), radius=9, fill=badge_fill)
        draw.text((badge_x1 + 12, card_top + 22), result_label.replace("TIED BEST", "TIED"), fill=badge_text, font=font(12, bold=True))
        draw.text(
            (left + 18, card_top + 58),
            f"{int(summary_row['top10_hits'])}/10 hits | {summary_row['predicted_top10_actual_points']:.0f} pts | {int(summary_row['podium_hits'])}/3 podium",
            fill="#D8A31A",
            font=font(15, bold=True),
        )

        podium = model_df.head(3)
        y = card_top + 116
        for _, row in podium.iterrows():
            color = constructor_color(str(row["constructor_name"]))
            avatar = circular_driver_image(str(row["driver_code"]), str(row["driver_name"]), image_map, 66, color)
            image.paste(avatar, (left + 18, y), avatar)
            draw.text((left + 96, y + 3), f"#{int(row['predicted_rank'])} {row['driver_code']}", fill="#111111", font=font(18, bold=True))
            draw.text(
                (left + 96, y + 27),
                f"Actual P{int(row['final_position'])} | {row['points']:.0f} pts | {row['top10_probability']:.0%}",
                fill="#555555",
                font=font(14),
            )
            y += 76

        y += 14
        draw.text((left + 18, y), "Predicted top 10", fill="#123C43", font=font(17, bold=True))
        draw.text((right - 155, y + 3), "prob", fill="#666666", font=font(11, bold=True))
        draw.text((right - 106, y + 3), "real", fill="#666666", font=font(11, bold=True))
        draw.text((right - 59, y + 3), "pts", fill="#666666", font=font(11, bold=True))
        y += 30
        for _, row in model_df.head(10).iterrows():
            hit = int(row["hit_top10"]) == 1
            row_fill = "#E7F4EA" if hit else "#F7E4E1"
            draw.rounded_rectangle((left + 16, y, right - 16, y + 44), radius=9, fill=row_fill)
            draw.text((left + 28, y + 11), f"{int(row['predicted_rank']):02d}", fill="#111111", font=font(13, bold=True))
            draw_text_fit(draw, (left + 62, y + 9), f"{row['driver_code']} {row['driver_name']}", card_width - 245, 14, "#222222", bold=True)
            draw.text((right - 156, y + 9), f"{row['top10_probability']:.0%}", fill="#555555", font=font(13, bold=True))
            draw.text((right - 107, y + 9), f"P{int(row['final_position'])}", fill="#222222", font=font(13, bold=True))
            draw.text((right - 59, y + 9), f"{row['points']:.0f}", fill="#555555", font=font(13))
            y += 50

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_race_overview(
    race_id: str,
    race_rankings: pd.DataFrame,
    race_summary: pd.DataFrame,
    image_map: dict[str, Path],
    output_path: Path,
) -> None:
    race_summary = race_summary.copy()
    race_summary["model_priority"] = race_summary["model"].map(MODEL_PRIORITY).fillna(999).astype(int)
    race_summary = race_summary.sort_values(
        ["race_rank_among_models", "model_priority"],
        ascending=[True, True],
    )

    race_name = str(race_rankings["grand_prix"].iloc[0])
    season = int(race_rankings["season"].iloc[0])
    round_number = int(race_rankings["round"].iloc[0])
    best_models = set(str(race_summary.iloc[0].get("best_models_for_race", "")).split("|"))

    width, height = 2200, 1400
    image = Image.new("RGB", (width, height), "#F3F0E8")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, 150), fill="#123C43")
    draw.text((58, 40), f"{season} Round {round_number:02d} - {race_name}", fill="white", font=font(40, bold=True))
    draw.text(
        (58, 98),
        "Race overview: real top 10, model scoreboard, virtual podiums and per-model predicted top 10.",
        fill="#D8A31A",
        font=font(21),
    )

    actual_top10 = (
        race_rankings.drop_duplicates("driver_id")
        .sort_values("final_position")
        .head(10)
        [["driver_code", "driver_name", "constructor_name", "final_position", "points"]]
    )

    left = 60
    top = 185
    panel_w = 610
    panel_h = 1128
    draw_rounded(draw, (left, top, left + panel_w, top + panel_h), "#FFFFFF", "#D8A31A", 3)
    draw.text((left + 26, top + 24), "Real race result - top 10", fill="#123C43", font=font(26, bold=True))
    y = top + 76
    for _, row in actual_top10.iterrows():
        color = constructor_color(str(row["constructor_name"]))
        row_fill = "#F7F7F7" if int(row["final_position"]) % 2 else "#EAF3F4"
        draw.rounded_rectangle((left + 22, y, left + panel_w - 22, y + 82), radius=12, fill=row_fill)
        avatar = circular_driver_image(str(row["driver_code"]), str(row["driver_name"]), image_map, 58, color)
        image.paste(avatar, (left + 36, y + 12), avatar)
        draw.text((left + 108, y + 13), f"P{int(row['final_position'])} {row['driver_code']}", fill="#111111", font=font(20, bold=True))
        draw_text_fit(draw, (left + 108, y + 39), str(row["driver_name"]), 330, 15, "#333333")
        draw.text((left + panel_w - 110, y + 28), f"{row['points']:.0f} pts", fill="#123C43", font=font(18, bold=True))
        y += 94

    middle = 705
    middle_w = 610
    draw_rounded(draw, (middle, top, middle + middle_w, top + 540), "#FFFFFF", "#D0D0D0", 2)
    draw.text((middle + 26, top + 24), "Model scoreboard", fill="#123C43", font=font(26, bold=True))
    draw.text((middle + 28, top + 64), "hits / points / podium", fill="#666666", font=font(15))
    y = top + 100
    for _, row in race_summary.iterrows():
        model_name = str(row["model"])
        is_best = model_name in best_models
        fill = "#FFF4CF" if is_best else "#F7F7F7"
        outline = "#D8A31A" if is_best else "#DDDDDD"
        draw.rounded_rectangle((middle + 22, y, middle + middle_w - 22, y + 72), radius=13, fill=fill, outline=outline, width=2)
        draw.text((middle + 42, y + 13), model_label(model_name).replace("\n", " "), fill="#123C43", font=font(18, bold=True))
        draw.text(
            (middle + 42, y + 42),
            f"{int(row['top10_hits'])}/10 hits | {row['predicted_top10_actual_points']:.0f} pts | {int(row['podium_hits'])}/3 podium",
            fill="#333333",
            font=font(15),
        )
        badge = str(row.get("model_result_label", "RANK"))
        badge_fill = "#D8A31A" if is_best else "#E7E1D3"
        draw.rounded_rectangle((middle + middle_w - 140, y + 20, middle + middle_w - 42, y + 48), radius=9, fill=badge_fill)
        draw.text((middle + middle_w - 126, y + 27), badge.replace("TIED BEST", "TIED"), fill="#123C43", font=font(12, bold=True))
        y += 84

    draw_rounded(draw, (middle, top + 575, middle + middle_w, top + panel_h), "#FFFFFF", "#D0D0D0", 2)
    draw.text((middle + 26, top + 600), "Virtual podiums", fill="#123C43", font=font(26, bold=True))
    y = top + 652
    for _, row in race_summary.iterrows():
        model_name = str(row["model"])
        model_df = race_rankings[race_rankings["model"] == model_name].sort_values("predicted_rank").head(3)
        podium_codes = []
        for _, podium_row in model_df.iterrows():
            podium_codes.append(f"{int(podium_row['predicted_rank'])}. {podium_row['driver_code']} (real P{int(podium_row['final_position'])})")
        draw_text_fit(draw, (middle + 30, y), model_label(model_name).replace("\n", " "), middle_w - 70, 16, "#123C43", bold=True)
        draw_wrapped_text(draw, (middle + 30, y + 26), " | ".join(podium_codes), middle_w - 70, 19, 14, "#333333", max_lines=2)
        y += 86

    right = 1350
    right_w = 790
    draw_rounded(draw, (right, top, right + right_w, top + panel_h), "#FFFFFF", "#D0D0D0", 2)
    draw.text((right + 26, top + 24), "Predicted top 10 by model", fill="#123C43", font=font(26, bold=True))
    draw.text((right + 28, top + 64), "Green chips finished in the real top 10; red chips missed it.", fill="#666666", font=font(15))
    best_text = str(race_summary.iloc[0].get("race_explanation", ""))
    draw.rounded_rectangle((right + 28, top + 112, right + right_w - 28, top + 172), radius=10, outline="#D8A31A", width=2)
    draw_wrapped_text(draw, (right + 45, top + 126), best_text, right_w - 100, 18, 13, "#333333", max_lines=2)

    y = top + 205
    chip_w = 48
    chip_gap = 4
    for _, summary_row in race_summary.iterrows():
        model_name = str(summary_row["model"])
        model_df = race_rankings[race_rankings["model"] == model_name].sort_values("predicted_rank").head(10)
        draw_text_fit(draw, (right + 28, y + 11), model_label(model_name).replace("\n", " "), 210, 16, "#123C43", bold=True)
        x = right + 250
        for _, row in model_df.iterrows():
            hit = int(row["hit_top10"]) == 1
            chip_fill = "#DFF0E5" if hit else "#F4DCD8"
            chip_outline = "#92B69E" if hit else "#CC9992"
            draw.rounded_rectangle((x, y, x + chip_w, y + 42), radius=10, fill=chip_fill, outline=chip_outline, width=1)
            draw.text((x + chip_w / 2, y + 11), str(row["driver_code"]), fill="#111111", font=font(10, bold=True), anchor="ma")
            x += chip_w + chip_gap
        y += 62

    consensus = (
        race_rankings[race_rankings["predicted_top10"] == 1]
        .groupby(["driver_code", "driver_name", "constructor_name", "final_position", "points"], as_index=False)
        .agg(model_votes=("model", "nunique"))
        .sort_values(["model_votes", "points", "final_position"], ascending=[False, False, True])
        .head(5)
    )
    draw.text((right + 28, top + 750), "Consensus picks", fill="#123C43", font=font(22, bold=True))
    y = top + 795
    for _, row in consensus.iterrows():
        hit = int(row["final_position"]) <= 10
        fill = "#DFF0E5" if hit else "#F4DCD8"
        draw.rounded_rectangle((right + 28, y, right + right_w - 28, y + 48), radius=10, fill=fill)
        draw.text((right + 46, y + 13), f"{row['driver_code']} {row['driver_name']}", fill="#111111", font=font(15, bold=True))
        draw.text(
            (right + right_w - 250, y + 13),
            f"{int(row['model_votes'])}/{len(race_summary)} models | real P{int(row['final_position'])} | {row['points']:.0f} pts",
            fill="#333333",
            font=font(14),
        )
        y += 58

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_summary_figures(rankings: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    plot_summary = summary.copy()
    plot_summary["model_display"] = plot_summary["model"].map(lambda value: model_label(str(value)).replace("\n", " "))

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=plot_summary,
        x="round",
        y="top10_hits",
        hue="model_display",
        marker="o",
    )
    plt.ylim(0, 10)
    plt.title("Top-10 Prediction Hits by Race and Model")
    plt.xlabel("2025 race round")
    plt.ylabel("Correct predicted top-10 drivers")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "model_precision_by_race.png", dpi=180)
    plt.close()

    heatmap = plot_summary.pivot_table(index="grand_prix", columns="model_display", values="top10_hits", aggfunc="mean")
    heatmap = heatmap.loc[plot_summary.sort_values("round")["grand_prix"].drop_duplicates()]
    plt.figure(figsize=(11, 10))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=10, cbar_kws={"label": "Top-10 hits"})
    plt.title("Race-Level Top-10 Hits Heatmap")
    plt.xlabel("Model")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "model_hit_heatmap.png", dpi=180)
    plt.close()

    points = plot_summary.pivot_table(index="model_display", values="predicted_top10_actual_points", aggfunc="mean").reset_index()
    points = points.sort_values("predicted_top10_actual_points", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=points, x="predicted_top10_actual_points", y="model_display", color="#D8A31A")
    plt.title("Average Actual Points Captured by Each Model's Predicted Top 10")
    plt.xlabel("Average actual points among predicted top-10 drivers")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "model_points_captured.png", dpi=180)
    plt.close()

    consensus = (
        rankings[rankings["predicted_top10"] == 1]
        .groupby(["race_id", "grand_prix", "driver_code", "driver_name", "constructor_name", "final_position", "points"], as_index=False)
        .agg(model_votes=("model", "nunique"))
        .sort_values(["race_id", "model_votes", "points"], ascending=[True, False, False])
    )
    consensus.to_csv(RENDER_PATH / "model_consensus_top10.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError(f"Dataset is empty: {args.data}")

    RENDER_PATH.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    RACE_CARD_PATH.mkdir(parents=True, exist_ok=True)
    RACE_OVERVIEW_PATH.mkdir(parents=True, exist_ok=True)

    rankings, summary = generate_predictions(df, args.test_season, args.models)
    rankings_path = RENDER_PATH / f"race_model_rankings_{args.test_season}.csv"
    summary_path = RENDER_PATH / f"race_model_summary_{args.test_season}.csv"
    notes_path = RENDER_PATH / f"race_model_notes_{args.test_season}.json"
    rankings.to_csv(rankings_path, index=False)
    summary.to_csv(summary_path, index=False)

    driver_codes = set(rankings["driver_code"].dropna().astype(str).str.upper())
    image_map = load_headshot_map(driver_codes, args.test_season, args.with_headshots)

    save_summary_figures(rankings, summary)

    races = rankings[["race_id", "round", "grand_prix"]].drop_duplicates().sort_values("round")
    if args.max_races is not None:
        races = races.head(args.max_races)

    race_overviews: list[str] = []
    if not args.skip_overviews:
        for _, race in races.iterrows():
            race_id = str(race["race_id"])
            race_name = str(race["grand_prix"])
            output = RACE_OVERVIEW_PATH / f"{race_id}_{slugify(race_name)}.png"
            render_race_overview(
                race_id=race_id,
                race_rankings=rankings[rankings["race_id"] == race_id],
                race_summary=summary[summary["race_id"] == race_id],
                image_map=image_map,
                output_path=output,
            )
            race_overviews.append(str(output))

    race_cards: list[str] = []
    if not args.skip_cards:
        for _, race in races.iterrows():
            race_id = str(race["race_id"])
            race_name = str(race["grand_prix"])
            output = RACE_CARD_PATH / f"{race_id}_{slugify(race_name)}.png"
            render_race_card(
                race_id=race_id,
                race_rankings=rankings[rankings["race_id"] == race_id],
                race_summary=summary[summary["race_id"] == race_id],
                image_map=image_map,
                output_path=output,
            )
            race_cards.append(str(output))

    notes = {
        "rankings_path": str(rankings_path),
        "summary_path": str(summary_path),
        "figure_dir": str(FIGURE_PATH),
        "race_card_dir": str(RACE_CARD_PATH),
        "race_overview_dir": str(RACE_OVERVIEW_PATH),
        "race_overviews_generated": len(race_overviews),
        "race_cards_generated": len(race_cards),
        "headshots_available": len(image_map),
        "headshot_source": "OpenF1 drivers endpoint headshot_url when --with-headshots is used; local fallback initials otherwise.",
        "interpretation_note": "Race explanations are heuristic summaries based on top-10 hits, points captured, grid position and missed actual top-10 drivers.",
    }
    with notes_path.open("w", encoding="utf-8") as file:
        json.dump(notes, file, indent=2)

    print("Prediction renders generated.")
    print(f"Rankings: {rankings_path}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {FIGURE_PATH}")
    print(f"Race overviews: {len(race_overviews)}")
    print(f"Race cards: {len(race_cards)}")
    print(f"Headshots available: {len(image_map)}")


if __name__ == "__main__":
    main()
