from __future__ import annotations

import argparse
from pathlib import Path

from data.paths import ROOT_DIR
from experiments.manifest import read_manifest, write_manifest
from experiments.runners.counterfactual import (
    build_counterfactual_specs,
    default_counterfactual_manifest_path,
    load_config,
    merge_counterfactual_outputs,
    run_counterfactual_specs,
    summarize_backbone_status,
)
from experiments.selectors import RunSelector
from experiments.spec import RunContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifest-driven counterfactual forecasting evaluation.")
    parser.add_argument("--config", default=str(Path("configs") / "counterfactual_eval.yaml"))
    parser.add_argument("--manifest", default=None, help="Experiment manifest jsonl. Legacy csv values are ignored for compatibility.")
    parser.add_argument("--manifest-out", default=None, help="Where to write the built experiment manifest jsonl.")
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--view-manifest", default=str(Path("statistic_results") / "eval_view_manifest.csv"))
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--registry", default=str(Path("statistic_results") / "dataset_registry.csv"))
    parser.add_argument("--task-root", default=None, help="Per-setting output root. Defaults to --results-dir when provided.")
    parser.add_argument("--results-dir", default=str(Path("results") / "counterfactual"))
    parser.add_argument("--report-out", default=str(Path("reports") / "counterfactual_eval_summary.md"))
    parser.add_argument("--setting-logs-dir", default=str(Path("logs") / "counterfactual_eval_settings"))
    parser.add_argument("--setting-id", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--backbone", action="append", default=[])
    parser.add_argument("--horizon", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--train-view", action="append", default=[])
    parser.add_argument("--eval-view", action="append", default=[])
    parser.add_argument("--start-setting-index", type=int, default=1)
    parser.add_argument("--end-setting-index", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--merge-after-run", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    return parser.parse_args()


def resolve_path(raw_path: str | None, default: Path | None = None) -> Path:
    if raw_path is None:
        if default is None:
            raise ValueError("Missing path.")
        return default
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def split_values(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for piece in str(value).split(","):
            item = piece.strip()
            if item:
                tokens.add(item)
    return tokens


def int_values(values: list[str]) -> set[int]:
    return {int(item) for item in split_values(values)}


def infer_auto_merge(args: argparse.Namespace) -> bool:
    if args.skip_merge:
        return False
    if args.merge_after_run:
        return True
    if args.merge_only:
        return True
    return args.shard_id is None and (args.num_shards is None or args.num_shards == 1)


def resolve_experiment_manifest_path(args: argparse.Namespace) -> Path | None:
    if args.manifest is None:
        return None
    manifest_path = resolve_path(args.manifest)
    if manifest_path.suffix.lower() == ".csv":
        return None
    return manifest_path


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    task_root = resolve_path(args.task_root) if args.task_root is not None else resolve_path(args.results_dir)
    report_out = resolve_path(args.report_out)
    setting_logs_dir = resolve_path(args.setting_logs_dir)
    views_dir = resolve_path(args.views_dir)
    view_manifest_path = resolve_path(args.view_manifest)
    events_path = resolve_path(args.events)
    registry_path = resolve_path(args.registry)

    config = load_config(config_path)
    manifest_out = resolve_path(args.manifest_out, default=default_counterfactual_manifest_path(task_root))
    manifest_path = resolve_experiment_manifest_path(args)
    if manifest_path is None and args.build_manifest_only:
        manifest_path = manifest_out

    if manifest_path is not None and manifest_path.exists():
        specs = read_manifest(manifest_path)
    else:
        specs = build_counterfactual_specs(config)
        if args.build_manifest_only or args.manifest_out is not None:
            write_manifest(manifest_out, specs)

    if args.build_manifest_only:
        return

    context = RunContext(
        config_path=config_path,
        view_manifest_path=view_manifest_path,
        views_dir=views_dir,
        events_path=events_path,
        registry_path=registry_path,
        task_root=task_root,
        report_out=report_out,
        setting_logs_dir=setting_logs_dir,
    )

    selector = RunSelector(
        setting_ids=split_values(args.setting_id),
        datasets=split_values(args.dataset),
        backbones=split_values(args.backbone),
        horizons=int_values(args.horizon),
        seeds=int_values(args.seed),
        train_views=split_values(args.train_view),
        eval_views=split_values(args.eval_view),
        start_setting_index=max(1, int(args.start_setting_index)) if args.start_setting_index is not None else None,
        end_setting_index=args.end_setting_index,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    if args.merge_only:
        merge_counterfactual_outputs(
            task_root=task_root,
            report_out=report_out,
            setting_logs_dir=setting_logs_dir,
            backbone_status=summarize_backbone_status(config),
        )
        return

    run_counterfactual_specs(
        specs=specs,
        selector=selector,
        context=context,
        auto_merge=infer_auto_merge(args),
    )


if __name__ == "__main__":
    main()
