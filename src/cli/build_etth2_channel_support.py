from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import ROOT_DIR


CHANNELS = ["HUFL", "HULL", "LUFL", "LULL", "MUFL", "MULL", "OT"]


def log_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [011/channel-support] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ETTh2 channel support diagnosis table.")
    parser.add_argument("--events", default=str(Path("statistic_results") / "final_artifact_events.csv"))
    parser.add_argument("--views-dir", default=str(Path("statistic_results") / "window_views"))
    parser.add_argument("--out", default=str(Path("results") / "etth2_channel_support.csv"))
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--lookbacks", default="")
    parser.add_argument("--horizons", default="96,192")
    parser.add_argument("--min-support", type=int, default=200)
    parser.add_argument("--high-flag-rate", type=float, default=0.8)
    return parser.parse_args()


def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def build_channel_support_table(
    raw_test_windows: pd.DataFrame,
    raw_train_windows: pd.DataFrame,
    events_df: pd.DataFrame,
    channels: list[str],
    min_support: int,
    high_flag_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_cache: dict[str, pd.DataFrame] = {}

    for channel in channels:
        mask = events_df["variables"].fillna("").astype(str).str.contains(
            re.compile(fr"(?:^|,)\s*{re.escape(channel)}\s*(?:,|$)")
        )
        event_cache[channel] = events_df[mask].copy()

    for channel in channels:
        n_total = 0
        n_input_flag = 0
        n_target_flag = 0
        train_clean_support = 0
        severities: list[float] = []
        channel_events = event_cache[channel]
        log_progress(
            f"scan channel={channel} test_windows={len(raw_test_windows)} train_windows={len(raw_train_windows)} "
            f"events={len(channel_events)}"
        )

        for row in raw_train_windows.itertuples(index=False):
            hit_target = False
            for event in channel_events.itertuples(index=False):
                start_idx = int(event.start_idx)
                end_idx = int(event.end_idx)
                if overlap_len(int(row.target_start), int(row.target_end), start_idx, end_idx) > 0:
                    hit_target = True
                    break
            train_clean_support += int(not hit_target)

        for row in raw_test_windows.itertuples(index=False):
            n_total += 1
            hit_input = False
            hit_target = False
            for event in channel_events.itertuples(index=False):
                start_idx = int(event.start_idx)
                end_idx = int(event.end_idx)
                if overlap_len(int(row.input_start), int(row.input_end), start_idx, end_idx) > 0:
                    hit_input = True
                    severities.append(float(getattr(event, "severity", 0.0) or 0.0))
                if overlap_len(int(row.target_start), int(row.target_end), start_idx, end_idx) > 0:
                    hit_target = True
                    severities.append(float(getattr(event, "severity", 0.0) or 0.0))
            n_input_flag += int(hit_input)
            n_target_flag += int(hit_target)

        target_clean_support = n_total - n_target_flag
        input_flag_rate = n_input_flag / max(n_total, 1)
        target_flag_rate = n_target_flag / max(n_total, 1)
        mean_severity = float(np.mean(severities)) if severities else 0.0

        if target_clean_support == 0:
            status = "collapsed"
        elif target_clean_support < min_support or input_flag_rate > high_flag_rate:
            status = "mixed"
        else:
            status = "recoverable"

        rows.append(
            {
                "channel": channel,
                "n_windows": n_total,
                "input_flag_rate": input_flag_rate,
                "target_flag_rate": target_flag_rate,
                "train_clean_support": train_clean_support,
                "target_clean_support": target_clean_support,
                "test_target_clean_support": target_clean_support,
                "mean_severity": mean_severity,
                "status": status,
            }
        )
        log_progress(
            f"done channel={channel} train_clean_support={train_clean_support} "
            f"target_clean_support={target_clean_support} "
            f"input_flag_rate={input_flag_rate:.3f} status={status}"
        )

    return pd.DataFrame(rows).sort_values(
        ["status", "target_clean_support", "input_flag_rate"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    events = pd.read_csv(ROOT_DIR / Path(args.events), low_memory=False)
    events = events[events["dataset_name"] == "ETTh2"].copy()
    out_rows: list[pd.DataFrame] = []
    lookback_values = (
        [int(item.strip()) for item in str(args.lookbacks).split(",") if item.strip()]
        if str(args.lookbacks).strip()
        else [int(args.lookback)]
    )
    horizon_values = [item.strip() for item in str(args.horizons).split(",") if item.strip()]
    log_progress(f"start lookbacks={lookback_values} horizons={horizon_values} events={len(events)}")

    for lookback in lookback_values:
        for horizon_str in horizon_values:
            horizon = int(horizon_str)
            view_path = ROOT_DIR / Path(args.views_dir) / f"ETTh2_L{lookback}_H{horizon}.csv"
            if not view_path.exists():
                log_progress(f"missing view file {view_path}")
                continue
            view_df = pd.read_csv(view_path, low_memory=False)
            raw_train = view_df[(view_df["split_name"] == "train") & (view_df["is_raw_view"] == 1)].copy()
            raw_test = view_df[(view_df["split_name"] == "test") & (view_df["is_raw_view"] == 1)].copy()
            if raw_test.empty or raw_train.empty:
                log_progress(
                    f"skip lookback={lookback} horizon={horizon} raw support missing "
                    f"train={len(raw_train)} test={len(raw_test)}"
                )
                continue
            log_progress(
                f"build lookback={lookback} horizon={horizon} "
                f"raw_train_rows={len(raw_train)} raw_test_rows={len(raw_test)}"
            )
            support = build_channel_support_table(
                raw_test_windows=raw_test,
                raw_train_windows=raw_train,
                events_df=events,
                channels=CHANNELS,
                min_support=int(args.min_support),
                high_flag_rate=float(args.high_flag_rate),
            )
            support.insert(0, "lookback", lookback)
            support.insert(1, "horizon", horizon)
            out_rows.append(support)

    out_path = ROOT_DIR / Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    out_df.to_csv(out_path, index=False)
    log_progress(f"wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
