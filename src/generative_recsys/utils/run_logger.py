"""Record each run: resolved config, metrics, git state and hardware, in runs/<timestamp>-<name>/.

Layout of a run directory:

    config.yaml    the fully merged config the run used
    meta.json      status, timestamps, command, git commit and dirty flag, hardware
    history.jsonl  metrics logged during the run (one JSON object per line), if any
    metrics.json   final metrics, written by Run.finish()
"""

import json
import os
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from generative_recsys.utils.paths import REPO_ROOT


class DirtyTreeWarning(UserWarning):
    """The run's code isn't fully captured by its git commit."""


class Run:
    """One run's directory. Use as a context manager so failures are recorded."""

    def __init__(self, path: Path, meta: dict[str, Any]):
        self.path = path
        self.meta = meta

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Append metrics from partway through the run (e.g. per epoch) to history.jsonl."""
        record = {"step": step, **metrics} if step is not None else dict(metrics)
        with (self.path / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, default=_to_builtin) + "\n")

    def finish(self, metrics: Mapping[str, Any]) -> None:
        """Write the final metrics and mark the run finished."""
        _write_json(self.path / "metrics.json", dict(metrics))
        self._set_status("finished")

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._set_status("failed", error=f"{exc_type.__name__}: {exc}")
        elif self.meta["status"] == "running":
            self._set_status("incomplete")  # exited without calling finish()

    def _set_status(self, status: str, **extra: Any) -> None:
        self.meta.update(status=status, finished_at=_now(), **extra)
        _write_json(self.path / "meta.json", self.meta)


def start_run(
    config: Any,
    *,
    device: torch.device | str | None = None,
    root: str | Path | None = None,
    repo_dir: str | Path = REPO_ROOT,
) -> Run:
    """Create the run directory and write config.yaml and meta.json.

    ``config`` is a schema instance from ``load_config`` (or any structured config).
    Its ``run.name`` names the directory and ``run.output_dir`` is the default
    ``root``, relative to the repo. Warns with ``DirtyTreeWarning`` if there are
    uncommitted changes, since the git SHA alone then doesn't reproduce the run.
    """
    cfg = OmegaConf.structured(config)
    name = OmegaConf.select(cfg, "run.name", default="run")
    root = Path(
        root if root is not None else OmegaConf.select(cfg, "run.output_dir", default="runs")
    )
    if not root.is_absolute():
        root = Path(repo_dir) / root

    git = _git_state(Path(repo_dir))  # before writing anything, so the run can't dirty the tree
    path = _make_run_dir(root, name)
    (path / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    if git["sha"] is None:
        message = f"{repo_dir} is not a git repository; run {path.name} has no commit to reproduce."
        warnings.warn(message, DirtyTreeWarning, stacklevel=2)
    elif git["dirty"]:
        files = ", ".join(git["dirty_files"])
        message = (
            f"Uncommitted changes ({files}); "
            f"run {path.name} won't be reproducible from {git['sha'][:7]} alone."
        )
        warnings.warn(message, DirtyTreeWarning, stacklevel=2)

    meta = {
        "name": name,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "command": [sys.executable, *sys.argv],
        "git": git,
        "hardware": _hardware(device),
    }
    _write_json(path / "meta.json", meta)
    return Run(path, meta)


def _make_run_dir(root: Path, name: str) -> Path:
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-") or "run"
    base = f"{_timestamp()}-{slug}"
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ["", *(f"-{i}" for i in range(2, 100))]:
        path = root / f"{base}{suffix}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"Too many runs named {base!r} in {root}")


def _git_state(repo_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout

    sha = git("rev-parse", "HEAD")
    if sha is None:
        return {"sha": None, "branch": None, "dirty": None, "dirty_files": []}
    # Porcelain lines are "XY path"; X can be a space, so don't strip the output.
    status = git("status", "--porcelain") or ""
    dirty_files = [line[3:] for line in status.splitlines() if line]
    return {
        "sha": sha.strip(),
        "branch": (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:20],
    }


def _hardware(device: torch.device | str | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device) if device is not None else None,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, ValueError, OSError):
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_builtin(value: Any) -> Any:
    # numpy and torch scalars (and 0-d tensors) expose .item()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_to_builtin) + "\n")
