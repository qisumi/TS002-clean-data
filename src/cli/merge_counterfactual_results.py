from __future__ import annotations

import argparse
from pathlib import Path

from data.paths import ROOT_DIR
from experiments.runners.counterfactual import load_config, merge_counterfactual_outputs, summarize_backbone_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-setting counterfactual outputs into aggregate tables.")
    parser.add_argument("--config", default=str(Path("configs") / "counterfactual_eval.yaml"))
    parser.add_argument("--task-root", default=str(Path("results") / "counterfactual"))
    parser.add_argument("--report-out", default=str(Path("reports") / "counterfactual_eval_summary.md"))
    parser.add_argument("--setting-logs-dir", default=str(Path("logs") / "counterfactual_eval_settings"))
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    task_root = resolve_path(args.task_root)
    report_out = resolve_path(args.report_out)
    setting_logs_dir = resolve_path(args.setting_logs_dir)

    config = load_config(config_path)
    merge_counterfactual_outputs(
        task_root=task_root,
        report_out=report_out,
        setting_logs_dir=setting_logs_dir,
        backbone_status=summarize_backbone_status(config),
    )


if __name__ == "__main__":
    main()
