from __future__ import annotations

import json
from typing import Any

import numpy as np


def parse_intervention_recipe(recipe_text: str) -> list[dict[str, Any]]:
    text = recipe_text.strip()
    if not text or text == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def resolve_event_variable_indices(
    artifact_id: str,
    events_lookup: dict[str, dict[str, Any]],
    column_index: dict[str, int],
) -> list[int]:
    event = events_lookup.get(artifact_id)
    if event is None:
        return list(range(len(column_index)))

    variables = str(event.get("variables", "")).strip()
    if not variables or variables == "ALL" or variables == "nan":
        return list(range(len(column_index)))

    indices = [column_index[name] for name in variables.split(",") if name in column_index]
    return indices or list(range(len(column_index)))


def fill_with_context_mean(window: np.ndarray, start: int, end: int, indices: list[int]) -> None:
    left = window[:start, indices]
    right = window[end + 1 :, indices]
    context = np.concatenate([left, right], axis=0) if left.size or right.size else np.empty((0, len(indices)), dtype=window.dtype)
    if context.size == 0:
        fill_values = np.zeros(len(indices), dtype=window.dtype)
    else:
        fill_values = context.mean(axis=0)
    window[start : end + 1, indices] = fill_values


def linear_interpolate_span(window: np.ndarray, start: int, end: int, indices: list[int]) -> None:
    seq_len = window.shape[0]
    span_len = end - start + 1
    if span_len <= 0:
        return

    for var_idx in indices:
        left_idx = start - 1 if start > 0 else None
        right_idx = end + 1 if end + 1 < seq_len else None

        if left_idx is not None and right_idx is not None:
            left_val = float(window[left_idx, var_idx])
            right_val = float(window[right_idx, var_idx])
            window[start : end + 1, var_idx] = np.linspace(left_val, right_val, span_len + 2, dtype=np.float32)[1:-1]
        elif left_idx is not None:
            window[start : end + 1, var_idx] = float(window[left_idx, var_idx])
        elif right_idx is not None:
            window[start : end + 1, var_idx] = float(window[right_idx, var_idx])
        else:
            window[start : end + 1, var_idx] = 0.0


def apply_intervention_recipe(
    input_window: np.ndarray,
    global_input_start: int,
    recipe_text: str,
    events_lookup: dict[str, dict[str, Any]],
    column_index: dict[str, int],
) -> np.ndarray:
    if not recipe_text or recipe_text == "nan":
        return input_window

    window = input_window.copy()
    input_end = global_input_start + window.shape[0] - 1
    ops = parse_intervention_recipe(recipe_text)
    for op in ops:
        op_name = str(op.get("op", ""))
        if op_name == "drop_window":
            continue

        start_idx = max(int(op.get("start", global_input_start)), global_input_start)
        end_idx = min(int(op.get("end", input_end)), input_end)
        if start_idx > end_idx:
            continue

        local_start = start_idx - global_input_start
        local_end = end_idx - global_input_start
        artifact_id = str(op.get("artifact_id", ""))
        indices = resolve_event_variable_indices(artifact_id, events_lookup=events_lookup, column_index=column_index)

        if op_name == "local_trend_interp_span":
            linear_interpolate_span(window, local_start, local_end, indices)
        else:
            fill_with_context_mean(window, local_start, local_end, indices)
    return window
