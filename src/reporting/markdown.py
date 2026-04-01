from __future__ import annotations

from pathlib import Path

from data.io import ensure_directory


def append_stage_progress(
    report_path: str | Path,
    stage_id: str,
    completed: str,
    generated_files: list[str],
    mainline_results: list[str],
    supporting_results: list[str],
    blockers: list[str],
) -> None:
    path = Path(report_path)
    ensure_directory(path.parent)
    lines = [
        f"## {stage_id}",
        "",
        "1. 本阶段完成了什么",
        completed,
        "",
        "2. 生成了哪些文件",
        *[f"- `{item}`" for item in generated_files],
        "",
        "3. 哪些结果能支持主线",
        *[f"- {item}" for item in mainline_results],
        "",
        "4. 哪些结果只是 supporting evidence",
        *[f"- {item}" for item in supporting_results],
        "",
        "5. 下一步的阻塞点是什么",
        *[f"- {item}" for item in blockers],
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
