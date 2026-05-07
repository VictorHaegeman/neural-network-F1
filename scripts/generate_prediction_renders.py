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
        ordered = race_group.sort_values(
            ["top10_hits", "predicted_top10_actual_points", "podium_hits"],
            ascending=[False, False, False],
        )
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else best
        for _, row in race_group.iterrows():
            if row["model"] == best["model"]:
                reason = (
                    f"Best race-level model: {model_label(str(row['model'])).replace(chr(10), ' ')} found "
                    f"{int(row['top10_hits'])}/10 actual top-10 drivers and captured "
                    f"{int(row['midfield_top10_hits'])} top-10 finisher(s) starting outside the top 10."
                )
            else:
                gap = int(best["top10_hits"] - row["top10_hits"])
                if gap == 0:
                    reason = (
                        f"Tied on top-10 hits with the best model group at {int(row['top10_hits'])}/10. "
                        f"The selected best model captured slightly more actual race points or podium hits."
                    )
                else:
                    reason = (
                        f"Behind the best model by {gap} top-10 hit(s). It missed "
                        f"{int(row['actual_top10_missed'])} actual top-10 driver(s), while the best model "
                        f"reached {int(best['top10_hits'])}/10 hits."
                    )
            row_dict = row.to_dict()
            row_dict["best_model_for_race"] = best["model"]
            row_dict["best_model_top10_hits"] = int(best["top10_hits"])
            row_dict["runner_up_model_for_race"] = second["model"]
            row_dict["race_explanation"] = reason
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


def render_race_card(
    race_id: str,
    race_rankings: pd.DataFrame,
    race_summary: pd.DataFrame,
    image_map: dict[str, Path],
    output_path: Path,
) -> None:
    available = set(race_summary["model"].unique())
    models = [model for model in MODEL_NAMES if model in available]
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
        "Virtual podium and predicted top 10 by model. Green rows finished in the real top 10; points are actual race points.",
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
        left = 60 + index * (card_width + 24)
        right = left + card_width
        draw_rounded(draw, (left, card_top, right, height - 55), "#FFFFFF", "#D0D0D0", 2)
        draw.rectangle((left, card_top, right, card_top + 82), fill="#123C43")
        draw.text((left + 18, card_top + 16), model_label(model_name), fill="white", font=font(22, bold=True))
        draw.text(
            (left + 18, card_top + 58),
            f"{int(summary_row['top10_hits'])}/10 hits | {summary_row['predicted_top10_actual_points']:.0f} pts",
            fill="#D8A31A",
            font=font(15, bold=True),
        )

        podium = model_df.head(3)
        y = card_top + 105
        for _, row in podium.iterrows():
            color = constructor_color(str(row["constructor_name"]))
            avatar = circular_driver_image(str(row["driver_code"]), str(row["driver_name"]), image_map, 66, color)
            image.paste(avatar, (left + 18, y), avatar)
            draw.text((left + 96, y + 3), f"#{int(row['predicted_rank'])} {row['driver_code']}", fill="#111111", font=font(18, bold=True))
            draw.text((left + 96, y + 27), f"Actual P{int(row['final_position'])} | {row['points']:.0f} pts", fill="#555555", font=font(14))
            y += 76

        y += 14
        draw.text((left + 18, y), "Predicted top 10", fill="#123C43", font=font(17, bold=True))
        y += 30
        for _, row in model_df.head(10).iterrows():
            hit = int(row["hit_top10"]) == 1
            row_fill = "#E7F4EA" if hit else "#F7E4E1"
            draw.rounded_rectangle((left + 16, y, right - 16, y + 44), radius=9, fill=row_fill)
            draw.text((left + 28, y + 11), f"{int(row['predicted_rank']):02d}", fill="#111111", font=font(13, bold=True))
            draw_text_fit(draw, (left + 62, y + 9), f"{row['driver_code']} {row['driver_name']}", card_width - 190, 14, "#222222", bold=True)
            draw.text((right - 120, y + 9), f"P{int(row['final_position'])}", fill="#222222", font=font(13, bold=True))
            draw.text((right - 72, y + 9), f"{row['points']:.0f} pts", fill="#555555", font=font(13))
            y += 50

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_summary_figures(rankings: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    sns.lineplot(
        data=summary,
        x="round",
        y="top10_hits",
        hue="model",
        marker="o",
    )
    plt.ylim(0, 10)
    plt.title("Top-10 Prediction Hits by Race and Model")
    plt.xlabel("2025 race round")
    plt.ylabel("Correct predicted top-10 drivers")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "model_precision_by_race.png", dpi=180)
    plt.close()

    heatmap = summary.pivot_table(index="grand_prix", columns="model", values="top10_hits", aggfunc="mean")
    heatmap = heatmap.loc[summary.sort_values("round")["grand_prix"].drop_duplicates()]
    plt.figure(figsize=(11, 10))
    sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=10, cbar_kws={"label": "Top-10 hits"})
    plt.title("Race-Level Top-10 Hits Heatmap")
    plt.xlabel("Model")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "model_hit_heatmap.png", dpi=180)
    plt.close()

    points = summary.pivot_table(index="model", values="predicted_top10_actual_points", aggfunc="mean").reset_index()
    points = points.sort_values("predicted_top10_actual_points", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=points, x="predicted_top10_actual_points", y="model", color="#D8A31A")
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

    rankings, summary = generate_predictions(df, args.test_season, args.models)
    rankings_path = RENDER_PATH / f"race_model_rankings_{args.test_season}.csv"
    summary_path = RENDER_PATH / f"race_model_summary_{args.test_season}.csv"
    notes_path = RENDER_PATH / f"race_model_notes_{args.test_season}.json"
    rankings.to_csv(rankings_path, index=False)
    summary.to_csv(summary_path, index=False)

    driver_codes = set(rankings["driver_code"].dropna().astype(str).str.upper())
    image_map = load_headshot_map(driver_codes, args.test_season, args.with_headshots)

    save_summary_figures(rankings, summary)

    race_cards: list[str] = []
    if not args.skip_cards:
        races = rankings[["race_id", "round", "grand_prix"]].drop_duplicates().sort_values("round")
        if args.max_races is not None:
            races = races.head(args.max_races)
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
    print(f"Race cards: {len(race_cards)}")
    print(f"Headshots available: {len(image_map)}")


if __name__ == "__main__":
    main()
