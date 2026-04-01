from __future__ import annotations

from pathlib import Path


def discover_backbone_repo(root_dir: Path, repo_candidates: list[str]) -> Path | None:
    for candidate in repo_candidates:
        path = root_dir / candidate
        if path.exists():
            return path
    return None
