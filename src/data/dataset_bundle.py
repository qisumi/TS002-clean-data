from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from .paths import RAW_DIR, ROOT_DIR

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - optional dependency in CSV-only workflows
    h5py = None  # type: ignore[assignment]


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".h5", ".hdf5"}
TIME_COLUMNS = {"date", "datetime", "time", "timestamp"}
DATASET_FREQ_HINTS = {
    "etth1": "1H",
    "etth2": "1H",
    "ettm1": "15T",
    "ettm2": "15T",
    "weather": "10T",
    "exchange_rate": "1D",
    "electricity": "1H",
    "traffic": "1H",
    "solar_al": "10T",
    "metr-la": "5T",
}
ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
SOLAR_DATASETS = ["solar_AL"]
DEFAULT_ANALYSIS_DATASETS = [*ETT_DATASETS, *SOLAR_DATASETS]
AnyArray = npt.NDArray[Any]


@dataclass
class DatasetBundle:
    dataset_name: str
    file_path: Path
    frame: pd.DataFrame
    time_col: str | None
    freq: str
    source_format: str
    notes: str = ""

    @property
    def numeric_columns(self) -> list[str]:
        columns: list[str] = []
        for column in self.frame.columns:
            if column == self.time_col:
                continue
            if pd.api.types.is_numeric_dtype(self.frame[column]):
                columns.append(column)
        return columns


def default_dataset_argument() -> str:
    return ",".join(DEFAULT_ANALYSIS_DATASETS)


def _normalize_dataset_filter(dataset_names: Iterable[str] | None) -> set[str]:
    if dataset_names is None:
        return set()
    return {str(item).strip().lower() for item in dataset_names if str(item).strip()}


def list_dataset_files(dataset_names: Iterable[str] | None = None) -> list[Path]:
    dataset_filter = _normalize_dataset_filter(dataset_names)
    file_list = [
        file_path
        for file_path in RAW_DIR.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        and (not dataset_filter or dataset_name_from_path(file_path).lower() in dataset_filter)
    ]
    file_list.sort()
    return file_list


def dataset_name_from_path(file_path: Path) -> str:
    return file_path.stem


def is_time_column(column_name: str) -> bool:
    return column_name.strip().lower() in TIME_COLUMNS


def read_dataset(file_path: Path) -> DatasetBundle:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame, time_col, notes = _read_csv_dataset(file_path)
    elif suffix == ".txt":
        frame, time_col, notes = _read_txt_dataset(file_path)
    elif suffix in {".h5", ".hdf5"}:
        frame, time_col, notes = _read_hdf5_dataset(file_path)
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix}")

    dataset_name = dataset_name_from_path(file_path)
    freq = infer_frequency(dataset_name, frame, time_col)
    return DatasetBundle(
        dataset_name=dataset_name,
        file_path=file_path,
        frame=frame,
        time_col=time_col,
        freq=freq,
        source_format=suffix.lstrip("."),
        notes=notes,
    )


def _read_csv_dataset(file_path: Path) -> tuple[pd.DataFrame, str | None, str]:
    frame = pd.read_csv(file_path, low_memory=False)
    time_col = next((col for col in frame.columns if is_time_column(col)), None)
    if time_col is not None:
        parsed = pd.to_datetime(frame[time_col], errors="coerce")
        if parsed.notna().any():
            frame[time_col] = parsed
    return frame, time_col, "CSV dataset loaded with pandas."


def _read_txt_dataset(file_path: Path) -> tuple[pd.DataFrame, str | None, str]:
    try:
        matrix = np.loadtxt(file_path, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(file_path)

    matrix = _reshape_to_2d_array(matrix)
    columns = [f"var_{index:03d}" for index in range(matrix.shape[1])]
    frame = pd.DataFrame(matrix, columns=pd.Index(columns))
    return frame, None, "Numeric text matrix loaded without an explicit time column."


def _read_hdf5_dataset(file_path: Path) -> tuple[pd.DataFrame, str | None, str]:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to read .h5/.hdf5 datasets")
    with h5py.File(file_path, "r") as handle:
        if {"df/axis0", "df/block0_values"}.issubset(_collect_hdf5_keys(handle)):
            axis_dataset = _require_hdf5_dataset(handle["df/axis0"], "df/axis0")
            values_dataset = _require_hdf5_dataset(handle["df/block0_values"], "df/block0_values")
            columns = _decode_axis(axis_dataset[()])
            values = _reshape_to_2d_array(values_dataset[()])
            frame = pd.DataFrame(values, columns=pd.Index(columns))
            return frame, None, "Pandas-style HDF5 dataset loaded via h5py."

        datasets = _collect_hdf5_datasets(handle)
        if not datasets:
            raise ValueError("No datasets found in HDF5 file.")

        largest_key = ""
        largest_dataset: h5py.Dataset | None = None
        largest_size = -1
        for dataset_key in datasets:
            dataset = _require_hdf5_dataset(handle[dataset_key], dataset_key)
            dataset_size = int(np.prod(dataset.shape))
            if dataset_size > largest_size:
                largest_key = dataset_key
                largest_dataset = dataset
                largest_size = dataset_size

        if largest_dataset is None:
            raise ValueError("No tabular datasets found in HDF5 file.")

        values = _reshape_to_2d_array(largest_dataset[()])
        columns = [f"var_{index:03d}" for index in range(values.shape[1])]
        frame = pd.DataFrame(values, columns=pd.Index(columns))
        return frame, None, f"Largest HDF5 dataset `{largest_key}` loaded as a tabular matrix."


def _collect_hdf5_keys(handle: h5py.File) -> set[str]:
    keys: set[str] = set()
    handle.visit(keys.add)
    return keys


def _collect_hdf5_datasets(handle: h5py.File) -> list[str]:
    datasets: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)

    handle.visititems(visit)
    return datasets


def _require_hdf5_dataset(node: h5py.Group | h5py.Dataset | h5py.Datatype, key: str) -> h5py.Dataset:
    if isinstance(node, h5py.Dataset):
        return node
    raise ValueError(f"HDF5 key `{key}` is not a dataset.")


def _reshape_to_2d_array(values: Any) -> AnyArray:
    array = cast(AnyArray, np.asarray(values))
    if array.ndim == 0:
        return cast(AnyArray, array.reshape(1, 1))
    if array.ndim == 1:
        return cast(AnyArray, array.reshape(-1, 1))
    if array.ndim > 2:
        return cast(AnyArray, array.reshape(array.shape[0], -1))
    return array


def _decode_axis(raw_values: Any) -> list[str]:
    values = np.asarray(raw_values)
    decoded: list[str] = []
    for value in values.tolist():
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(value))
    return decoded


def infer_frequency(dataset_name: str, frame: pd.DataFrame, time_col: str | None) -> str:
    hint = DATASET_FREQ_HINTS.get(dataset_name.lower())
    if hint:
        return hint
    if time_col is None or time_col not in frame.columns:
        return "unknown"

    series = pd.Series(pd.to_datetime(frame[time_col], errors="coerce")).dropna()
    if len(series) < 3:
        return "unknown"

    deltas = series.sort_values().diff().dropna()
    if deltas.empty:
        return "unknown"

    median_delta = deltas.median()
    if not isinstance(median_delta, pd.Timedelta):
        return "unknown"
    return timedelta_to_freq_string(median_delta)


def timedelta_to_freq_string(delta: pd.Timedelta) -> str:
    seconds = int(delta.total_seconds())
    mapping = {
        60: "1T",
        300: "5T",
        600: "10T",
        900: "15T",
        1800: "30T",
        3600: "1H",
        7200: "2H",
        86400: "1D",
        604800: "7D",
    }
    return mapping.get(seconds, f"{seconds}s")


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)).replace("\\", "/")
