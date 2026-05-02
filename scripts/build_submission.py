from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "submission" / "IML_Assignment_GroupX.zip"

INCLUDE_PATHS = [
    "README.md",
    "requirements.txt",
    "docs",
    "scripts",
    "data",
    "outputs",
    "notebooks",
    "report",
]

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".fastf1_cache",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "models",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the assignment submission ZIP.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def should_include(path: Path) -> bool:
    relative_parts = set(path.relative_to(PROJECT_ROOT).parts)
    if relative_parts & EXCLUDED_PARTS:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.stat().st_size == 0:
        return False
    if path == DEFAULT_OUTPUT:
        return False
    return path.is_file()


def iter_submission_files() -> list[Path]:
    files: list[Path] = []
    for include_path in INCLUDE_PATHS:
        path = PROJECT_ROOT / include_path
        if path.is_file() and should_include(path):
            files.append(path)
        elif path.is_dir():
            files.extend(file for file in path.rglob("*") if should_include(file))
    return sorted(files)


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    files = iter_submission_files()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(PROJECT_ROOT))

    print(f"Wrote {output}")
    print(f"Included {len(files)} files")


if __name__ == "__main__":
    main()
