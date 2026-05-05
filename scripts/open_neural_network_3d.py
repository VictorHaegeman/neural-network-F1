from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEW_PATH = PROJECT_ROOT / "outputs" / "figures" / "neural_network_embedding_3d.html"
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "visualize_neural_network_3d.py"


def main() -> None:
    if not VIEW_PATH.exists() or VIEW_PATH.stat().st_size == 0:
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--color-by",
                "cluster",
                "--max-rows",
                "3500",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    webbrowser.open(VIEW_PATH.resolve().as_uri())
    print(f"Opened {VIEW_PATH}")
    print("Use the browser tab: drag to rotate, scroll to zoom, hover points for details.")


if __name__ == "__main__":
    main()
