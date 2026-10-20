from dataclasses import dataclass, field

import pytest
from omegaconf import MISSING

from generative_recsys.utils import (
    ConfigError,
    ExperimentConfig,
    RunConfig,
    config_from_argv,
    load_config,
)
from generative_recsys.utils.config import PROFILES_DIR


@dataclass
class TrainSection:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10


@dataclass
class DemoExperiment(ExperimentConfig):
    train: TrainSection = field(default_factory=TrainSection)


@dataclass
class RequiresDataset(ExperimentConfig):
    dataset: str = MISSING


@pytest.fixture
def write_yaml(tmp_path):
    def write(name: str, text: str):
        path = tmp_path / name
        path.write_text(text)
        return path

    return write


def test_schema_defaults_only():
    cfg = load_config(DemoExperiment)
    assert cfg == DemoExperiment()
    assert isinstance(cfg, DemoExperiment)
    assert isinstance(cfg.run, RunConfig)


def test_layers_override_in_order(write_yaml):
    # batch_size is set by every layer; each other key is last set by exactly one layer.
    config = write_yaml("exp.yaml", "train: {batch_size: 64, epochs: 50}\nrun: {seed: 1}\n")
    profile = write_yaml("prof.yaml", "train: {batch_size: 128}\nrun: {device: cpu}\n")

    cfg = load_config(DemoExperiment, config, profile, ["train.batch_size=256", "run.name=x"])

    assert cfg.train.batch_size == 256  # CLI beats profile beats file
    assert cfg.run.name == "x"  # CLI
    assert cfg.run.device == "cpu"  # profile
    assert (cfg.train.epochs, cfg.run.seed) == (50, 1)  # file
    assert cfg.train.lr == 1e-3  # schema default


def test_profile_beats_config_file(write_yaml):
    config = write_yaml("exp.yaml", "run: {device: mps}\n")
    profile = write_yaml("prof.yaml", "run: {device: cpu}\n")
    assert load_config(DemoExperiment, config, profile).run.device == "cpu"


def test_cli_values_are_converted_to_schema_types():
    cfg = load_config(DemoExperiment, overrides=["train.lr=0.5", "run.deterministic=false"])
    assert cfg.train.lr == 0.5
    assert cfg.run.deterministic is False


@pytest.mark.parametrize("name", ["local", "cuda"])
def test_repo_profiles_load(name):
    cfg = load_config(ExperimentConfig, profile=name)
    assert cfg.run.device == {"local": "auto", "cuda": "cuda"}[name]


def test_every_repo_profile_matches_the_base_schema():
    profiles = sorted(PROFILES_DIR.glob("*.yaml"))
    assert profiles
    for path in profiles:
        load_config(ExperimentConfig, profile=path)


def test_unknown_profile_lists_available():
    with pytest.raises(ConfigError, match=r"unknown profile 'gpu'; available: cuda, local"):
        load_config(ExperimentConfig, profile="gpu")


def test_unknown_key_in_file_names_file_and_key(write_yaml):
    config = write_yaml("exp.yaml", "train: {lrr: 0.1}\n")
    with pytest.raises(ConfigError) as info:
        load_config(DemoExperiment, config)
    message = str(info.value)
    assert message.startswith(str(config))
    assert "key: train.lrr" in message


def test_unknown_key_in_profile_names_profile(write_yaml):
    profile = write_yaml("prof.yaml", "train: {batch_size: 8}\n")
    with pytest.raises(ConfigError, match=r"^profile .*prof\.yaml.*key: train"):
        load_config(ExperimentConfig, profile=profile)


def test_unknown_key_on_cli():
    with pytest.raises(ConfigError, match=r"^command-line overrides: .*key: run\.sed"):
        load_config(DemoExperiment, overrides=["run.sed=3"])


def test_wrong_type_fails_with_key():
    with pytest.raises(ConfigError, match=r"'fast'.*key: train\.lr"):
        load_config(DemoExperiment, overrides=["train.lr=fast"])


def test_wrong_type_in_file(write_yaml):
    config = write_yaml("exp.yaml", "train: {epochs: [1, 2]}\n")
    with pytest.raises(ConfigError, match=r"key: train\.epochs"):
        load_config(DemoExperiment, config)


def test_override_without_equals_is_rejected():
    with pytest.raises(ConfigError, match=r"'train\.lr' must look like key=value"):
        load_config(DemoExperiment, overrides=["train.lr"])


def test_missing_required_value():
    with pytest.raises(ConfigError, match=r"^incomplete config: .*dataset"):
        load_config(RequiresDataset)
    assert load_config(RequiresDataset, overrides=["dataset=beauty"]).dataset == "beauty"


def test_config_from_argv(write_yaml):
    config = write_yaml("exp.yaml", "train: {epochs: 3}\n")
    argv = ["--config", str(config), "--profile", "local", "train.lr=0.01"]
    cfg = config_from_argv(DemoExperiment, argv)
    assert (cfg.train.epochs, cfg.train.lr, cfg.run.device) == (3, 0.01, "auto")
