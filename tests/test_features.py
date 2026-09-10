"""Tests for src/features.py.

Point-in-time correctness (no lookahead) is the critical property: test 1
recomputes features on a truncated copy of the input and checks that the
last row of the truncated output matches the corresponding row of the full
output exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import features  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# corr_with_btc_90 is null for every row of BTCUSDT by design (a symbol has
# no meaningful correlation with itself as a separate series) — see
# cross_asset_features(). It is excluded from the "not entirely null" check.
EXPECTED_ALL_NULL = {"BTCUSDT": {"corr_with_btc_90"}}


def _raw(symbol: str) -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"{symbol}_4h.parquet").sort("open_time")


@pytest.fixture(scope="module", params=SYMBOLS)
def symbol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def feature_frames() -> dict[str, pl.DataFrame]:
    return {
        s: features.build_features_for_symbol(s, PARQUET_DIR) for s in SYMBOLS
    }


@pytest.fixture(scope="module")
def full_df(symbol: str, feature_frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return feature_frames[symbol]


# ---------------------------------------------------------------------------
# 1. Point-in-time / no-lookahead test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [50, 150, 500, 2000])
def test_point_in_time_no_lookahead(symbol: str, n: int) -> None:
    raw = _raw(symbol)
    assert n < len(raw)

    btc_returns = features._load_btc_returns(PARQUET_DIR)

    full = features.compute_features(raw.clone(), btc_returns, symbol)
    truncated = features.compute_features(raw.head(n).clone(), btc_returns, symbol)

    row_full = full.row(n - 1, named=True)
    row_trunc = truncated.row(n - 1, named=True)

    for col in features.FEATURE_COLUMNS:
        full_val = row_full[col]
        trunc_val = row_trunc[col]
        if full_val is None or trunc_val is None:
            assert full_val is None and trunc_val is None, (
                f"{symbol}/{col}: full={full_val!r} truncated={trunc_val!r}"
            )
        elif isinstance(full_val, bool):
            assert full_val == trunc_val, f"{symbol}/{col} mismatch"
        else:
            assert full_val == pytest.approx(trunc_val, rel=1e-9, abs=1e-12), (
                f"{symbol}/{col}: full={full_val!r} truncated={trunc_val!r}"
            )


# ---------------------------------------------------------------------------
# 2. No feature column is entirely null
# ---------------------------------------------------------------------------


def test_no_column_entirely_null(symbol: str, full_df: pl.DataFrame) -> None:
    exempt = EXPECTED_ALL_NULL.get(symbol, set())
    height = full_df.height
    for col in features.FEATURE_COLUMNS:
        if col in exempt:
            continue
        null_count = full_df.select(pl.col(col).is_null().sum()).item()
        assert null_count < height, f"{symbol}/{col} is entirely null"


# ---------------------------------------------------------------------------
# 3. No feature column has infinite values
# ---------------------------------------------------------------------------


def test_no_infinite_values(symbol: str, full_df: pl.DataFrame) -> None:
    for col in features.FEATURE_COLUMNS:
        if full_df.schema[col] not in (pl.Float32, pl.Float64):
            continue
        inf_count = full_df.select(pl.col(col).is_infinite().sum()).item()
        assert inf_count == 0, f"{symbol}/{col} has {inf_count} infinite values"


# ---------------------------------------------------------------------------
# 4. rsi_14 is between 0 and 100 where not null
# ---------------------------------------------------------------------------


def test_rsi_bounds(symbol: str, full_df: pl.DataFrame) -> None:
    out_of_bounds = full_df.filter(
        pl.col("rsi_14").is_not_null()
        & ((pl.col("rsi_14") < 0) | (pl.col("rsi_14") > 100))
    )
    assert out_of_bounds.height == 0, f"{symbol}: rsi_14 out of [0, 100]"


# ---------------------------------------------------------------------------
# 5. close_position is between 0 and 1 where not null
# ---------------------------------------------------------------------------


def test_close_position_bounds(symbol: str, full_df: pl.DataFrame) -> None:
    out_of_bounds = full_df.filter(
        pl.col("close_position").is_not_null()
        & ((pl.col("close_position") < 0) | (pl.col("close_position") > 1))
    )
    assert out_of_bounds.height == 0, f"{symbol}: close_position out of [0, 1]"


# ---------------------------------------------------------------------------
# 6. Row count out == row count in
# ---------------------------------------------------------------------------


def test_row_count_preserved(symbol: str, full_df: pl.DataFrame) -> None:
    raw = _raw(symbol)
    assert full_df.height == raw.height
