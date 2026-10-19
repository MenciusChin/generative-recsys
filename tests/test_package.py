import importlib

import pytest

import generative_recsys

SUBPACKAGES = [
    "data",
    "eval",
    "models",
    "models.baselines",
    "models.semantic_ids",
    "models.generative",
    "training",
    "serving",
    "utils",
]


def test_version():
    assert generative_recsys.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(f"generative_recsys.{name}")
    assert module.__doc__
