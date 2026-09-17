"""Shared fixtures for the indicator reference tests.

The OHLCV series lives in ``tests/fixtures/indicator_reference.npz`` (see
``tests/reference/generate_indicator_fixture.py``) so every test compares
against exactly the bars the frozen references were computed from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pytest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "indicator_reference.npz"


def load_fixture() -> Dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def assert_series_match(
    ours: np.ndarray,
    ref: np.ndarray,
    *,
    start: Optional[int] = None,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    name: str = "",
) -> None:
    """Assert two indicator outputs agree.

    By default the NaN warm-up must be identical and every finite value must
    agree within ``rtol`` / ``atol``. With *start*, only indices from
    *start* onward are compared (used where a reference documents a
    different, decaying seed).
    """
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, f"{name}: shape {ours.shape} != {ref.shape}"
    if start is None:
        nan_ours = np.isnan(ours)
        nan_ref = np.isnan(ref)
        mismatch = np.flatnonzero(nan_ours != nan_ref)
        assert mismatch.size == 0, (
            f"{name}: NaN warm-up differs at indices {mismatch[:10].tolist()} "
            f"(ours has {nan_ours.sum()} NaN, reference {nan_ref.sum()})"
        )
        mask = ~nan_ours
    else:
        ours = ours[start:]
        ref = ref[start:]
        assert not np.isnan(ours).any() and not np.isnan(ref).any(), f"{name}: NaN after start"
        mask = np.ones(len(ours), dtype=bool)
    assert mask.sum() > 0, f"{name}: nothing to compare"
    np.testing.assert_allclose(ours[mask], ref[mask], rtol=rtol, atol=atol, err_msg=name)


@pytest.fixture(scope="session")
def ohlcv() -> Dict[str, np.ndarray]:
    """The frozen synthetic OHLCV bars plus frozen pandas-ta references."""
    return load_fixture()


@pytest.fixture(scope="session")
def assert_match() -> Callable[..., None]:
    """The :func:`assert_series_match` helper, injected as a fixture."""
    return assert_series_match
