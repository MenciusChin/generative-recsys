# Working on this repo

Conventions for AI-assisted work on this repo. The plan lives in [ROADMAP.md](ROADMAP.md), and tasks live
in [GitHub Issues](https://github.com/MenciusChin/generative-recsys/issues), one milestone per phase.

## How we work on an issue

1. **Design brief first, no code.** Before implementing an issue, present:
   - the problem it solves
   - the proposed design
   - alternatives and their trade-offs
   - open questions

   Then wait for agreement. This is a learning project: the reasoning matters as much as the code.
2. Branch from an up-to-date `main`, implement, and run `make check`.
3. Commit, push and open a PR whose description includes `Closes #N` and a **Design notes** section
   recording the decisions from the discussion.
4. Squash-merge only after CI is green and the user approves.
5. When a decision affects a later issue, also comment it on that issue, so it isn't only in chat.

## Commands

```bash
make install   # uv sync --locked
make check     # lint + tests (what CI runs)
make format    # ruff auto-fix + format
```

- A pre-commit hook runs ruff and file checks on every commit (`make hooks` installs it).
- After adding or changing dependencies, use `uv add` / `uv lock` and commit `uv.lock`. CI runs `uv sync --locked`.

## Git and GitHub

- **Run `git push` and every `gh` command through `direnv exec .`.** The machine's active `gh`
  account is a work account; `../.envrc` exports the personal account's token for this folder only.
  A plain `git push` fails, and `gh` would act as the wrong account.
- **Branch names**, checked in CI by `.github/scripts/check-branch-name.sh`:
  - `phase-<N>/<issue#>-<slug>`
  - `fix/<issue#>-<slug>`
  - `chore/<slug>`
  - `docs/<slug>`
- **`main` is protected.** It accepts PRs only, with the `lint and test` and `branch name` checks
  required, squash merges only and no admin bypass. Head branches are deleted on merge.
- **Each finished phase gets an annotated tag** (`v0.0-scaffold`, `v0.1-eval`, …). A ruleset blocks
  moving or deleting `v*` tags.
- **In GitHub Actions workflows,** pin third-party actions to a commit SHA, with the version in a
  comment. GitHub's own `actions/*` use major tags.

## Code

- **Every run** goes through `config_from_argv` / `load_config` → `get_device` → `seed_everything` →
  `start_run`, all in `generative_recsys.utils`.
- **Code must be device-agnostic** (CUDA → MPS → CPU). Keep CUDA-only dependencies optional.
- **Configs:** add a dataclass field before using a new setting. Profiles hold hardware settings only.
  See [configs/README.md](configs/README.md).
- **Data and run outputs:** `data/raw`, `data/processed` and `runs/` are gitignored. Never commit
  datasets.
