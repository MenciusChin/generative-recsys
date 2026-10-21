"""Typed experiment configs: schema defaults, then a config file, a profile and CLI overrides."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from generative_recsys.utils.paths import REPO_ROOT

PROFILES_DIR = REPO_ROOT / "configs" / "profiles"


class ConfigError(ValueError):
    """A config layer has an unknown key, a wrong type or a missing value."""


@dataclass
class RunConfig:
    """Settings every run has, whatever the model."""

    name: str = "run"
    seed: int = 0
    device: str = "auto"  # see utils.get_device
    deterministic: bool = True
    output_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Base schema. Experiment schemas subclass it and add their own sections."""

    run: RunConfig = field(default_factory=RunConfig)


def load_config[T](
    schema: type[T],
    config_path: str | Path | None = None,
    profile: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> T:
    """Build a ``schema`` instance from layers, each overriding the ones before it.

    Order: schema defaults → ``config_path`` → ``profile`` → ``overrides``
    (``"train.lr=0.001"`` style). ``profile`` is a name in ``configs/profiles/``
    (``"local"``, ``"cuda"``) or a path to a YAML file. Any unknown key, wrong
    type or missing required value raises ``ConfigError`` naming the layer at fault.
    """
    cfg = OmegaConf.structured(schema)
    for source, load in _layers(config_path, profile, overrides):
        try:
            cfg = OmegaConf.merge(cfg, load())
        except OmegaConfBaseException as e:
            raise ConfigError(f"{source}: {_describe(e)}") from e
    try:
        return OmegaConf.to_object(cfg)
    except OmegaConfBaseException as e:
        raise ConfigError(f"incomplete config: {_describe(e)}") from e


def config_from_argv[T](schema: type[T], argv: Sequence[str] | None = None) -> T:
    """Parse ``[--config PATH] [--profile NAME] [key=value ...]`` and load the config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="experiment YAML file")
    parser.add_argument("--profile", help="profile name in configs/profiles/, or a YAML path")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=0.001")
    args = parser.parse_args(argv)
    return load_config(schema, args.config, args.profile, args.overrides)


def resolve_profile(profile: str | Path) -> Path:
    """Map a profile name to its YAML file; paths pass through."""
    path = Path(profile)
    if path.suffix in {".yaml", ".yml"} or len(path.parts) > 1:
        return path
    path = PROFILES_DIR / f"{profile}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))
        raise ConfigError(f"unknown profile {str(profile)!r}; available: {', '.join(available)}")
    return path


def _layers(config_path, profile, overrides):
    if config_path is not None:
        yield str(config_path), lambda: OmegaConf.load(config_path)
    if profile is not None:
        path = resolve_profile(profile)
        yield f"profile {str(profile)!r} ({path})", lambda: OmegaConf.load(path)
    if overrides:
        yield "command-line overrides", lambda: _parse_overrides(overrides)


def _parse_overrides(overrides: Sequence[str]) -> DictConfig:
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must look like key=value")
    return OmegaConf.from_dotlist(list(overrides))


def _describe(e: OmegaConfBaseException) -> str:
    # OmegaConf messages carry extra detail lines (reference_type=...); keep the summary.
    summary = str(e.msg).splitlines()[0] if e.msg else type(e).__name__
    return f"{summary} (key: {e.full_key})" if e.full_key else summary
