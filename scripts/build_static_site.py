from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = PROJECT_ROOT / "public"


def copytree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def main() -> None:
    if PUBLIC_DIR.exists():
        shutil.rmtree(PUBLIC_DIR)
    PUBLIC_DIR.mkdir(parents=True)

    shutil.copy2(PROJECT_ROOT / "index.html", PUBLIC_DIR / "index.html")
    copytree(PROJECT_ROOT / "webapp", PUBLIC_DIR / "webapp")

    headshots = PROJECT_ROOT / "outputs" / "driver_headshots"
    if headshots.exists():
        output_dir = PUBLIC_DIR / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        copytree(headshots, output_dir / "driver_headshots")

    print(f"Static site written to {PUBLIC_DIR}")


if __name__ == "__main__":
    main()
