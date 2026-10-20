import json
import subprocess
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from generative_recsys.utils import (
    DirtyTreeWarning,
    ExperimentConfig,
    RunConfig,
    load_config,
    run_logger,
    start_run,
)


@dataclass
class TrainSection:
    lr: float = 1e-3


@dataclass
class DemoExperiment(ExperimentConfig):
    train: TrainSection = field(default_factory=TrainSection)


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A clean git repo with one commit."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "code.py").write_text("x = 1\n")
    (path / ".gitignore").write_text("runs/\nexperiments/\n")  # as in the real repo
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture
def config():
    return load_config(DemoExperiment, overrides=["run.name=SASRec Beauty", "train.lr=0.01"])


def read_json(path):
    return json.loads(path.read_text())


def test_run_directory_has_all_expected_files(repo, config):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean tree must not warn
        with start_run(config, device="mps", repo_dir=repo) as run:
            run.log_metrics({"loss": 0.5}, step=1)
            run.finish({"ndcg@10": 0.031})

    assert run.path.parent == repo / "runs"
    assert sorted(p.name for p in run.path.iterdir()) == [
        "config.yaml",
        "history.jsonl",
        "meta.json",
        "metrics.json",
    ]


def test_config_yaml_round_trips(repo, config):
    run = start_run(config, repo_dir=repo)
    saved = OmegaConf.merge(
        OmegaConf.structured(DemoExperiment), OmegaConf.load(run.path / "config.yaml")
    )
    assert OmegaConf.to_object(saved) == config


def test_meta_records_git_and_hardware(repo, config):
    run = start_run(config, device=torch.device("cpu"), repo_dir=repo)
    meta = read_json(run.path / "meta.json")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)

    assert meta["status"] == "running"
    assert meta["git"] == {
        "sha": head.stdout.strip(),
        "branch": "main",
        "dirty": False,
        "dirty_files": [],
    }
    hw = meta["hardware"]
    assert hw["device"] == "cpu"
    assert hw["torch"] == torch.__version__
    assert {"platform", "cpu", "cpu_count", "memory_gb", "python", "mps_available"} <= hw.keys()


def test_dirty_tree_warns_and_lists_files(repo, config):
    (repo / "code.py").write_text("x = 2\n")
    (repo / "new.py").write_text("")
    with pytest.warns(DirtyTreeWarning, match=r"Uncommitted changes \(code\.py, new\.py\)"):
        run = start_run(config, repo_dir=repo)
    git_meta = read_json(run.path / "meta.json")["git"]
    assert git_meta["dirty"] is True
    assert git_meta["dirty_files"] == ["code.py", "new.py"]


def test_not_a_git_repo_warns(tmp_path, config):
    with pytest.warns(DirtyTreeWarning, match="not a git repository"):
        run = start_run(config, repo_dir=tmp_path)
    assert read_json(run.path / "meta.json")["git"]["sha"] is None


def test_directory_name_is_timestamp_and_slug(repo, config, monkeypatch):
    monkeypatch.setattr(run_logger, "_timestamp", lambda: "20261020-120000")
    first = start_run(config, repo_dir=repo)
    second = start_run(config, repo_dir=repo)
    assert first.path.name == "20261020-120000-sasrec-beauty"
    assert second.path.name == "20261020-120000-sasrec-beauty-2"


def test_root_defaults_to_output_dir_and_can_be_overridden(repo, tmp_path):
    cfg = ExperimentConfig(run=RunConfig(output_dir="experiments/out"))
    assert start_run(cfg, repo_dir=repo).path.parent == repo / "experiments" / "out"
    assert start_run(cfg, root=tmp_path / "elsewhere", repo_dir=repo).path.parent == (
        tmp_path / "elsewhere"
    )


def test_metrics_accept_numpy_and_torch_scalars(repo, config):
    run = start_run(config, repo_dir=repo)
    run.log_metrics({"loss": torch.tensor(0.25)}, step=3)
    run.finish({"recall@10": np.float32(0.5), "users": np.int64(7)})

    assert read_json(run.path / "metrics.json") == {"recall@10": 0.5, "users": 7}
    assert json.loads((run.path / "history.jsonl").read_text()) == {"step": 3, "loss": 0.25}
    meta = read_json(run.path / "meta.json")
    assert meta["status"] == "finished"
    assert meta["finished_at"] is not None


def test_failure_is_recorded_and_reraised(repo, config):
    with pytest.raises(ValueError, match="boom"), start_run(config, repo_dir=repo) as run:
        raise ValueError("boom")
    meta = read_json(run.path / "meta.json")
    assert meta["status"] == "failed"
    assert meta["error"] == "ValueError: boom"
    assert not (run.path / "metrics.json").exists()


def test_exit_without_finish_is_marked_incomplete(repo, config):
    with start_run(config, repo_dir=repo) as run:
        pass
    assert read_json(run.path / "meta.json")["status"] == "incomplete"
