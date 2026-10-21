import subprocess

import pytest

from generative_recsys.utils.paths import REPO_ROOT

SCRIPT = REPO_ROOT / ".github" / "scripts" / "check-branch-name.sh"


def check(branch: str) -> subprocess.CompletedProcess:
    return subprocess.run([SCRIPT, branch], capture_output=True, text=True)


@pytest.mark.parametrize(
    "branch",
    [
        "phase-0/8-ci",
        "phase-1/13-ranking-metrics",
        "phase-7/120-llm-backbone",
        "fix/34-ndcg-tie-handling",
        "chore/bump-torch",
        "docs/phase-1-writeup",
        "dependabot/uv/torch-2.15.0",
    ],
)
def test_valid_names_pass(branch):
    result = check(branch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "branch",
    [
        "main-fix",  # no prefix
        "patch-1",  # GitHub web editor default
        "phase-0/ci",  # missing issue number
        "phase-8/1-beyond-roadmap",  # no phase 8
        "phase-1/13-Ranking-Metrics",  # uppercase
        "phase-1/13_ranking_metrics",  # underscores
        "fix/ndcg-ties",  # fix needs an issue number
        "feat/13-metrics",  # unknown type
        "chore/",  # empty slug
        "phase-1/13-metrics/extra",  # nested
    ],
)
def test_invalid_names_fail_with_guidance(branch):
    result = check(branch)
    assert result.returncode == 1
    assert f"Branch name '{branch}' doesn't follow" in result.stderr
    assert "phase-<N>/<issue#>-<slug>" in result.stderr


def test_missing_argument_fails():
    result = subprocess.run([SCRIPT], capture_output=True, text=True)
    assert result.returncode != 0
