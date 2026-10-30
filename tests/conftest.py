"""Markers: ``slow`` tests (100k-row exactness, minutes) are skipped unless selected with
``-m slow``; ``gpu`` tests skip themselves when no faiss GPU build is present."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
