from __future__ import annotations

import json
from pathlib import Path

from data.io import atomic_write_text, ensure_directory

from .spec import ExperimentSpec


def write_manifest(path: Path, specs: list[ExperimentSpec]) -> None:
    ensure_directory(path.parent)
    lines = [
        json.dumps(
            {
                "setting_id": spec.setting_id,
                "task_name": spec.task_name,
                "status": "pending",
                "spec": spec.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for spec in specs
    ]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_manifest(path: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        spec_payload = payload.get("spec", payload)
        specs.append(ExperimentSpec.from_dict(spec_payload))
    return specs
