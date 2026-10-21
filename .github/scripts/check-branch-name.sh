#!/usr/bin/env bash
# Usage: check-branch-name.sh <branch>
# Enforces the branch naming convention from ROADMAP.md.
set -euo pipefail

branch="${1:?usage: check-branch-name.sh <branch>}"
pattern='^(phase-[0-7]/[0-9]+-[a-z0-9-]+|fix/[0-9]+-[a-z0-9-]+|(chore|docs)/[a-z0-9-]+|dependabot/.+)$'

if [[ "$branch" =~ $pattern ]]; then
  echo "Branch name '$branch' is valid."
  exit 0
fi

cat >&2 <<MSG
::error::Branch name '$branch' doesn't follow the naming convention.

Expected one of:
  phase-<N>/<issue#>-<slug>   e.g. phase-1/13-ranking-metrics
  fix/<issue#>-<slug>         e.g. fix/34-ndcg-tie-handling
  chore/<slug>                e.g. chore/bump-torch
  docs/<slug>                 e.g. docs/phase-1-writeup

Lowercase letters, digits and hyphens only. Pattern: $pattern
Rename with: git branch -m <new-name>
MSG
exit 1
