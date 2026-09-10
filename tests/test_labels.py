"""Tests for src/labels.py.

The critical property is that a label depends on no bar after its own t1:
test 1 recomputes labels on data truncated at t1 and checks the label is
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import labels  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
HORIZON = labels.DEFAULT_HORIZON


def _features(symbol: str) -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"features_{symbol}.parquet").sort(
        "open_time"
    )


@pytest.fixture(scope="module", params=SYMBOLS)
def symbol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def labeled_frames() -> dict[str, pl.DataFrame]:
    return {
        s: labels.triple_barrier_labels(_features(s)) for s in SYMBOLS
    }


@pytest.fixture(scope="module")
def labeled(symbol: str, labeled_frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return labeled_frames[symbol]


# ---------------------------------------------------------------------------
# 1. No label uses data beyond t1
# ---------------------------------------------------------------------------


def test_no_label_uses_data_beyond_t1(symbol: str, labeled: pl.DataFrame) -> None:
    features = _features(symbol)

    # Sample across every exit type so the truncation check covers early
    # barrier touches (the case where lookahead would actually show up),
    # not just full-horizon expiries.
    indexed = labeled.with_row_index("idx")
    sample_idx: list[int] = []
    for hit in ("upper", "lower", "vertical"):
        rows = indexed.filter(pl.col("barrier_hit") == hit)
        step = max(1, rows.height // 15)
        sample_idx.extend(rows["idx"].gather_every(step).to_list()[:15])
    sample_idx.extend(indexed.filter(pl.col("ambiguous"))["idx"].to_list()[:10])

    assert sample_idx, f"{symbol}: no rows sampled"

    times = labeled["open_time"].to_list()
    time_to_idx = {t: k for k, t in enumerate(times)}

    for i in sample_idx:
        row = labeled.row(i, named=True)
        t1 = row["t1"]
        assert t1 is not None

        truncated = features.head(time_to_idx[t1] + 1)
        recomputed = labels.triple_barrier_labels(truncated).row(i, named=True)

        for col in ("label", "barrier_hit", "bars_held", "ambiguous"):
            assert recomputed[col] == row[col], (
                f"{symbol} row {i} ({col}): full={row[col]!r} "
                f"truncated-at-t1={recomputed[col]!r}"
            )
        assert recomputed["t1"] == t1
        if row["ret"] is None:
            assert recomputed["ret"] is None
        else:
            assert recomputed["ret"] == pytest.approx(row["ret"], rel=1e-12)


# ---------------------------------------------------------------------------
# 2. label null <=> t1 null, except ambiguous rows
# ---------------------------------------------------------------------------


def test_label_and_t1_nullness_agree(symbol: str, labeled: pl.DataFrame) -> None:
    # An ambiguous row has a known exit bar (t1) but an unknowable direction,
    # so it is the one case where label is null while t1 is set.
    bad_label_null = labeled.filter(
        pl.col("label").is_null() & pl.col("t1").is_not_null() & ~pl.col("ambiguous")
    )
    assert bad_label_null.height == 0, (
        f"{symbol}: {bad_label_null.height} rows have null label but non-null t1"
    )

    bad_t1_null = labeled.filter(pl.col("t1").is_null() & pl.col("label").is_not_null())
    assert bad_t1_null.height == 0, (
        f"{symbol}: {bad_t1_null.height} rows have null t1 but non-null label"
    )

    assert labeled.filter(pl.col("ambiguous") & pl.col("label").is_not_null()).height == 0


# ---------------------------------------------------------------------------
# 3. bars_held is between 1 and H
# ---------------------------------------------------------------------------


def test_bars_held_bounds(symbol: str, labeled: pl.DataFrame) -> None:
    out_of_bounds = labeled.filter(
        pl.col("bars_held").is_not_null()
        & ((pl.col("bars_held") < 1) | (pl.col("bars_held") > HORIZON))
    )
    assert out_of_bounds.height == 0, f"{symbol}: bars_held outside [1, {HORIZON}]"


# ---------------------------------------------------------------------------
# 4. Barrier touches have the right return sign
# ---------------------------------------------------------------------------


def test_barrier_hit_return_signs(symbol: str, labeled: pl.DataFrame) -> None:
    upper = labeled.filter(pl.col("barrier_hit") == "upper")
    assert upper.height > 0
    assert upper.filter(pl.col("ret") <= 0).height == 0, (
        f"{symbol}: upper-barrier rows with ret <= 0"
    )

    lower = labeled.filter(pl.col("barrier_hit") == "lower")
    assert lower.height > 0
    assert lower.filter(pl.col("ret") >= 0).height == 0, (
        f"{symbol}: lower-barrier rows with ret >= 0"
    )


# ---------------------------------------------------------------------------
# 5. Row count out == row count in
# ---------------------------------------------------------------------------


def test_row_count_preserved(symbol: str, labeled: pl.DataFrame) -> None:
    assert labeled.height == _features(symbol).height
