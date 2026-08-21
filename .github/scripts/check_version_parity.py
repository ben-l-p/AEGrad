"""Fail if flapjax and flapjax-full pyproject.toml versions differ."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECTS = {
    "flapjax": REPO_ROOT / "pyproject.toml",
    "flapjax-full": REPO_ROOT / "packaging" / "full" / "pyproject.toml",
}


def read_version(path: Path) -> str:
    with path.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def main() -> int:
    versions = {name: read_version(path) for name, path in PYPROJECTS.items()}
    unique = set(versions.values())
    if len(unique) == 1:
        print(f"OK: all pyproject.toml versions match ({unique.pop()})")
        return 0
    print("ERROR: pyproject.toml versions differ:", file=sys.stderr)
    for name, version in versions.items():
        print(f"  {name}: {version}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
