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

## Accounts

This machine has a work and a personal GitHub account. Everything in this repo must use the
**personal** account, `MenciusChin`. Two folder-scoped settings handle that:

| What | How | Where |
|---|---|---|
| Commit author | `includeIf "gitdir:…/vapecat/"` loads the personal identity | `~/.gitconfig` → `~/.gitconfig-personal` |
| GitHub auth (`git push`, `gh`) | exports the personal token as `GH_TOKEN` | `../.envrc`, via direnv |

Rules:
- **Run `git push` and every `gh` command through `direnv exec .`.** The machine's active `gh`
  account is the work account. Without direnv, `git push` fails and `gh` acts as the wrong account.
- **Check before the first outward action in a session** (push, PR, merge, issue comment):
  - `direnv exec . gh api user --jq .login` must print `MenciusChin`.
  - `git config user.name` must print `MenciusChin`.
  - If either doesn't match, **stop and tell the user**. Don't work around it.
- **Never change the global setup.** No `gh auth switch`, `gh auth login`, `gh auth refresh` or
  edits to `~/.gitconfig*`; those affect the user's work account too. If a token needs a new scope,
  give the user the commands to run.
- **Never print or write the token.** Don't echo `GH_TOKEN`, put it in a file or a remote URL, or
  commit `.envrc`, which lives outside the repo.
- **If direnv reports that `.envrc` is blocked,** ask the user to run `direnv allow`. Don't run it yourself.

## Git and GitHub

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
