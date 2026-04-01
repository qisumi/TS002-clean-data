from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


def progress(iterable: Iterable[T], *, total: int | None = None, desc: str = "", disable: bool = False) -> Iterator[T]:
    if disable or _tqdm is None:
        return iter(iterable)
    return iter(_tqdm(iterable, total=total, desc=desc))
