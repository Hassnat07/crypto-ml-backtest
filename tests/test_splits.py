"""Tests for src/splits.py.

The load-bearing test is test_verify_raises_on_naive_split: it proves the
leakage check actually fires on a split that leaks, so the passing checks
elsewhere in this file mean something.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import splits  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
N_SPLITS = 5
EMBARGO_BARS = 30


def _labeled(symbol: str) -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"labeled_{symbol}.parquet").sort("open_time")


@pytest.fixture(scope="module", params=SYMBOLS)
def symbol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def frames() -> dict[str, pl.DataFrame]:
    return {s: _labeled(s) for s in SYMBOLS}


@pytest.fixture(scope="module")
def df(symbol: str, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return frames[symbol]


@pytest.fixture(scope="module")
def folds(df: pl.DataFrame) -> list[splits.Fold]:
    return splits.build_folds(df, N_SPLITS, EMBARGO_BARS)


# ---------------------------------------------------------------------------
# 1. The leakage check passes on real splits
# ---------------------------------------------------------------------------


def test_verify_passes_on_generated_splits(df: pl.DataFrame) -> None:
    for train_idx, test_idx in splits.purged_walk_forward(df, N_SPLITS, EMBARGO_BARS):
        splits.verify_no_leakage(df, train_idx, test_idx)


# ---------------------------------------------------------------------------
# 2. The leakage check RAISES on an unpurged split (proves it is not vacuous)
# ---------------------------------------------------------------------------


def test_verify_raises_on_naive_split(symbol: str, df: pl.DataFrame) -> None:
    naive = splits.naive_walk_forward(df, N_SPLITS)
    assert len(naive) == N_SPLITS

    for k, (train_idx, test_idx) in enumerate(naive, start=1):
        with pytest.raises(splits.LeakageError, match="resolve inside the test window"):
            splits.verify_no_leakage(df, train_idx, test_idx)


def test_verify_raises_on_overlapping_split(df: pl.DataFrame) -> None:
    train_idx, test_idx = splits.purged_walk_forward(df, N_SPLITS, EMBARGO_BARS)[0]
    with pytest.raises(splits.LeakageError, match="share"):
        splits.verify_no_leakage(df, train_idx + test_idx[:1], test_idx)


def test_verify_raises_on_out_of_order_split(df: pl.DataFrame) -> None:
    train_idx, test_idx = splits.purged_walk_forward(df, N_SPLITS, EMBARGO_BARS)[0]
    later = [i for i in splits.eligible_indices(df) if i > max(test_idx)][:5]
    assert later
    with pytest.raises(splits.LeakageError, match="not before test index"):
        splits.verify_no_leakage(df, train_idx + later, test_idx)


# ---------------------------------------------------------------------------
# 3. Train and test never intersect
# ---------------------------------------------------------------------------


def test_train_and_test_disjoint(symbol: str, folds: list[splits.Fold]) -> None:
    for fold in folds:
        overlap = set(fold.train_idx) & set(fold.test_idx)
        assert not overlap, f"{symbol} fold {fold.index}: {len(overlap)} shared rows"


# ---------------------------------------------------------------------------
# 4. All train indices precede all test indices
# ---------------------------------------------------------------------------


def test_train_precedes_test(symbol: str, folds: list[splits.Fold]) -> None:
    for fold in folds:
        assert fold.train_idx and fold.test_idx
        assert max(fold.train_idx) < min(fold.test_idx), (
            f"{symbol} fold {fold.index}: train runs past test start"
        )


def test_train_expands_across_folds(symbol: str, folds: list[splits.Fold]) -> None:
    """Expanding window, not sliding: each fold trains on at least as much."""
    sizes = [f.n_train for f in folds]
    assert sizes == sorted(sizes), f"{symbol}: train sizes not monotonic: {sizes}"
    assert sizes[0] < sizes[-1]


# ---------------------------------------------------------------------------
# 5. Purging actually removes rows
# ---------------------------------------------------------------------------


def test_purge_removes_rows(symbol: str, folds: list[splits.Fold]) -> None:
    for fold in folds:
        assert fold.n_purged > 0, (
            f"{symbol} fold {fold.index}: purged 0 rows -- t1 is probably "
            "null, constant, or not forward-looking"
        )


def test_purged_train_is_subset_of_naive_train(df: pl.DataFrame) -> None:
    """Purging only ever removes training rows; it never invents them."""
    purged = splits.purged_walk_forward(df, N_SPLITS, EMBARGO_BARS)
    naive = splits.naive_walk_forward(df, N_SPLITS)

    for (p_train, p_test), (n_train, n_test) in zip(purged, naive):
        assert p_test == n_test, "purging must not alter the test block"
        assert set(p_train) < set(n_train)


def test_embargo_widens_the_dropped_band(df: pl.DataFrame) -> None:
    """A larger embargo drops at least as many training rows."""
    small = splits.build_folds(df, N_SPLITS, embargo_bars=0)
    large = splits.build_folds(df, N_SPLITS, embargo_bars=100)
    for a, b in zip(small, large):
        assert b.n_train <= a.n_train
    assert sum(f.n_embargoed for f in large) > sum(f.n_embargoed for f in small)


# ---------------------------------------------------------------------------
# 6. Holdout covers roughly the last 12 months
# ---------------------------------------------------------------------------


def test_holdout_covers_last_12_months(symbol: str, df: pl.DataFrame) -> None:
    dev_idx, holdout_idx = splits.final_holdout_split(df, 12, EMBARGO_BARS)
    assert dev_idx and holdout_idx

    times = df[holdout_idx]["open_time"]
    span_days = (times.max() - times.min()).days
    assert 330 <= span_days <= 375, f"{symbol}: holdout spans {span_days} days"

    eligible = splits.eligible_indices(df)
    assert times.max() == df[eligible]["open_time"].max()
    assert max(dev_idx) < min(holdout_idx)


def test_holdout_is_purged_against_dev(df: pl.DataFrame) -> None:
    dev_idx, holdout_idx = splits.final_holdout_split(df, 12, EMBARGO_BARS)
    splits.verify_no_leakage(df, dev_idx, holdout_idx)


def test_holdout_excluded_from_cv_is_caller_responsibility(df: pl.DataFrame) -> None:
    """CV over the dev slice alone never touches holdout rows."""
    dev_idx, holdout_idx = splits.final_holdout_split(df, 12, EMBARGO_BARS)
    dev_df = df[: max(dev_idx) + 1]
    holdout = set(holdout_idx)
    for train_idx, test_idx in splits.purged_walk_forward(
        dev_df, N_SPLITS, EMBARGO_BARS
    ):
        assert not (set(train_idx) | set(test_idx)) & holdout


# ---------------------------------------------------------------------------
# 7. Folds are ordered in time
# ---------------------------------------------------------------------------


def test_folds_ordered_in_time(symbol: str, df: pl.DataFrame, folds: list[splits.Fold]) -> None:
    for earlier, later in zip(folds, folds[1:]):
        earlier_end = df[earlier.test_idx]["open_time"].max()
        later_start = df[later.test_idx]["open_time"].min()
        assert earlier_end < later_start, (
            f"{symbol}: fold {earlier.index} test block does not precede "
            f"fold {later.index}"
        )


def test_test_blocks_are_contiguous_and_cover_the_tail(
    df: pl.DataFrame, folds: list[splits.Fold]
) -> None:
    """Test blocks tile the eligible rows after the seed training chunk."""
    eligible = splits.eligible_indices(df)
    tiled: list[int] = []
    for fold in folds:
        tiled.extend(fold.test_idx)
    assert tiled == eligible[len(eligible) - len(tiled) :]


def test_fold_count_matches_request(df: pl.DataFrame) -> None:
    for n in (2, 3, 5, 8):
        assert len(splits.purged_walk_forward(df, n, EMBARGO_BARS)) == n


def test_rejects_degenerate_configurations(df: pl.DataFrame) -> None:
    with pytest.raises(ValueError):
        splits.purged_walk_forward(df, 0, EMBARGO_BARS)
    with pytest.raises(ValueError):
        splits.build_folds(df.head(3), N_SPLITS, EMBARGO_BARS)
