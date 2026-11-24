"""Engine tests run with an isolated spill directory and from an isolated cwd."""
from __future__ import annotations

import warnings

import pytest

import pycauset as pc


@pytest.fixture(autouse=True, scope="session")
def _isolated_storage(tmp_path_factory):
    root = tmp_path_factory.mktemp("pycauset_storage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pc.PyCausetStorageWarning)
        pc.set_backing_dir(root)
    yield root


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Anything written relative to the cwd lands in a per-test temp dir."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path
