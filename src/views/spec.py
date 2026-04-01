from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_VIEW_SPEC: dict[str, Any] = {
    "defaults": {
        "lookback": 96,
        "horizons": [96, 192, 336, 720],
        "support": {
            "min_anchor_train_windows": 500,
            "min_anchor_train_ratio": 0.05,
            "min_conservative_train_windows": 1000,
            "min_conservative_train_ratio": 0.10,
        },
        "weights": {
            "corrupted_high": 1.00,
            "suspicious_high": 0.80,
            "suspicious_medium": 0.60,
            "valid_high": 0.40,
            "valid_medium": 0.20,
            "valid_low": 0.00,
        },
    },
    "ETTh2": {
        "anchor_clean": {
            "max_input_contam": 0.02,
            "max_target_contam": 0.00,
            "forbid_target_validity": ["corrupted", "suspicious"],
            "veto_multivar_severe": True,
            "veto_ot_severe": True,
        },
        "conservative_clean": {
            "max_input_contam": 0.10,
            "max_target_contam": 0.02,
            "forbid_target_validity": ["corrupted"],
        },
        "intervened": {
            "allow_repairable_input": True,
            "allow_unrecoverable_input_mask_only": True,
            "forbid_unrecoverable_target": True,
        },
    },
    "ETTm2": {
        "anchor_clean": {
            "max_input_contam": 0.01,
            "max_target_contam": 0.00,
            "forbid_target_validity": ["corrupted", "suspicious"],
            "veto_multivar_severe": True,
        },
        "conservative_clean": {
            "max_input_contam": 0.08,
            "max_target_contam": 0.01,
            "forbid_target_validity": ["corrupted"],
            "fallback_to_intervention_if_support_low": True,
        },
        "intervened": {
            "allow_repairable_input": True,
            "allow_unrecoverable_input_mask_only": True,
            "forbid_unrecoverable_target": True,
        },
    },
    "solar_AL": {
        "anchor_clean": {
            "require_target_dominant_phase": "active",
            "min_target_active_share": 0.80,
            "max_target_contam": 0.05,
            "max_input_contam": 0.20,
            "max_input_night_share": 0.20,
            "forbid_target_suspicious": True,
        },
        "conservative_clean": {
            "max_target_night_share": 0.00,
            "min_target_daylike_share": 0.80,
            "max_target_contam": 0.10,
            "max_input_contam": 0.25,
        },
        "phase_balanced": {
            "enabled": True,
            "stratify_by": "dominant_phase_target",
            "min_phases": 2,
            "min_per_phase": 100,
            "weight_clip": [0.5, 3.0],
        },
        "active_only": {"enabled": True, "min_target_active_share": 0.95, "max_input_night_share": 0.10},
        "daytime_only": {"enabled": True, "max_target_night_share": 0.00, "max_input_night_share": 0.05},
        "intervened": {
            "mask_night_phase": True,
            "mask_transition_phase": True,
            "never_repair_night_values": True,
            "forbid_unrecoverable_target": True,
        },
    },
    "ETTh1": {
        "anchor_clean": {
            "max_input_contam": 0.03,
            "max_target_contam": 0.00,
            "forbid_target_validity": ["corrupted", "suspicious"],
        },
        "conservative_clean": {
            "max_input_contam": 0.10,
            "max_target_contam": 0.02,
            "forbid_target_validity": ["corrupted"],
        },
        "intervened": {"forbid_unrecoverable_target": True},
    },
    "ETTm1": {
        "anchor_clean": {
            "max_input_contam": 0.03,
            "max_target_contam": 0.00,
            "forbid_target_validity": ["corrupted", "suspicious"],
        },
        "conservative_clean": {
            "max_input_contam": 0.10,
            "max_target_contam": 0.02,
            "forbid_target_validity": ["corrupted"],
        },
        "intervened": {"forbid_unrecoverable_target": True},
    },
    "weather": {
        "anchor_clean": {
            "max_input_contam": 0.12,
            "max_target_contam": 0.10,
            "forbid_target_validity": ["corrupted"],
        },
        "conservative_clean": {
            "max_input_contam": 0.35,
            "max_target_contam": 0.30,
            "forbid_target_validity": [],
        },
        "intervened": {"forbid_unrecoverable_target": True},
    },
    "exchange_rate": {
        "anchor_clean": {
            "max_input_contam": 0.08,
            "max_target_contam": 0.05,
            "forbid_target_validity": ["corrupted"],
        },
        "conservative_clean": {
            "max_input_contam": 0.20,
            "max_target_contam": 0.10,
            "forbid_target_validity": [],
        },
        "intervened": {"forbid_unrecoverable_target": True},
    },
    "electricity": {
        "anchor_clean": {
            "max_input_contam": 0.12,
            "max_target_contam": 0.08,
            "forbid_target_validity": ["corrupted"],
        },
        "conservative_clean": {
            "max_input_contam": 0.30,
            "max_target_contam": 0.20,
            "forbid_target_validity": [],
        },
        "intervened": {"forbid_unrecoverable_target": True},
    },
}


def load_view_spec(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return DEFAULT_VIEW_SPEC
    loaded = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    merged = DEFAULT_VIEW_SPEC.copy()
    merged.update(loaded)
    merged["defaults"] = {**DEFAULT_VIEW_SPEC["defaults"], **loaded.get("defaults", {})}
    return merged


def dump_view_spec(path: str | Path, spec: dict[str, Any] | None = None) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(yaml.safe_dump(spec or DEFAULT_VIEW_SPEC, sort_keys=False, allow_unicode=True), encoding="utf-8")
