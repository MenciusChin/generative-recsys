# Configs

Every run is described by a typed config: a dataclass schema in code, filled in from YAML and the
command line by `generative_recsys.utils.load_config`.

## Layers

Each layer overrides the ones before it:

1. **Schema defaults:** the dataclass fields, e.g. `RunConfig.seed = 0`
2. **Experiment file:** `--config configs/<experiment>.yaml`
3. **Profile:** `--profile local` or `--profile cuda`, from `configs/profiles/`. Holds hardware settings only.
4. **Command-line overrides:** `key=value` pairs, e.g. `run.seed=3 train.lr=0.0005`

```bash
uv run python scripts/train.py --config configs/sasrec.yaml --profile cuda run.seed=3
```

## Rules

- **Unknown keys and wrong types fail at load time.** The error names the layer and the key, e.g.
  `configs/sasrec.yaml: Key 'lrr' not in 'TrainConfig' (key: train.lrr)`. Fix the typo; don't
  loosen the schema.
- **A required field** (declared as `MISSING`) that no layer sets also fails at load time.
- **Profiles only set keys that every experiment has.** Today that's the `run` section. Shared
  training settings such as batch size join the profiles once a shared training schema exists (Phase 2).
- **YAML holds values; schemas hold structure.** To add a setting, add a dataclass field first.

## Adding an experiment schema

```python
from dataclasses import dataclass, field

from generative_recsys.utils import ExperimentConfig, config_from_argv


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256


@dataclass
class SASRecExperiment(ExperimentConfig):  # inherits the `run` section
    train: TrainConfig = field(default_factory=TrainConfig)


cfg = config_from_argv(SASRecExperiment)  # parses --config, --profile and key=value
```
