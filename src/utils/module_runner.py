from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from data.paths import ROOT_DIR


def with_src_pythonpath(
    *,
    env: Mapping[str, str] | None = None,
    root_dir: Path = ROOT_DIR,
) -> dict[str, str]:
    merged_env = dict(os.environ if env is None else env)
    src_dir = str(root_dir / "src")
    current = merged_env.get("PYTHONPATH", "")
    merged_env["PYTHONPATH"] = src_dir if not current else os.pathsep.join([src_dir, current])
    return merged_env


def module_command(
    module: str,
    module_args: Sequence[str],
    *,
    launcher: Sequence[str] | None = None,
) -> list[str]:
    python_launcher = list(launcher) if launcher is not None else [sys.executable]
    return [*python_launcher, "-m", module, *[str(arg) for arg in module_args]]


def run_python_module(
    module: str,
    module_args: Sequence[str],
    *,
    launcher: Sequence[str] | None = None,
    cwd: Path = ROOT_DIR,
    env: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    command = module_command(module, module_args, launcher=launcher)
    if log is not None:
        log("run " + " ".join(command))
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        env=with_src_pythonpath(env=env, root_dir=cwd),
    )
