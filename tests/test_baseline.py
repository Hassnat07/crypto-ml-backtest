"""Tests for src/baseline.py."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import baseline  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def _labeled(symbol: str) -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"labeled_{symbol}.parquet").sort("open_time")


@pytest.fixture(scope="module", params=SYMBOLS)
def symbol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def evaluated() -> dict[str, tuple[pl.DataFrame, dict[str, pl.DataFrame]]]:
    return {s: baseline.evaluate_symbol(_labeled(s), s) for s in SYMBOLS}


@pytest.fixture(scope="module")
def results(
    symbol: str, evaluated: dict[str, tuple[pl.DataFrame, dict[str, pl.DataFrame]]]
) -> pl.DataFrame:
    return evaluated[symbol][0]


@pytest.fixture(scope="module")
def trades(
    symbol: str, evaluated: dict[str, tuple[pl.DataFrame, dict[str, pl.DataFrame]]]
) -> dict[str, pl.DataFrame]:
    return evaluated[symbol][1]


# ---------------------------------------------------------------------------
# 1. Costs actually reduce returns
# ---------------------------------------------------------------------------


def test_costs_reduce_returns(symbol: str, results: pl.DataFrame) -> None:
    traded = results.filter(pl.col("trades") > 0)
    assert traded.height > 0

    for row in traded.iter_rows(named=True):
        assert row["mean_ret_net"] < row["mean_ret_gross"], (
            f"{symbol}/{row['strategy']}: net {row['mean_ret_net']} "
            f"not below gross {row['mean_ret_gross']}"
        )


def test_cost_is_the_full_round_trip(symbol: str, trades: dict[str, pl.DataFrame]) -> None:
    """The gap between gross and net is exactly one round trip per trade."""
    for name, frame in trades.items():
        if frame.height == 0:
            continue
        # Compared within tolerance, not bit-exactly: subtracting two floats
        # of differing magnitude perturbs the gap at ~1e-17 per row.
        gap = frame["ret_gross"] - frame["ret_net"]
        assert gap.min() == pytest.approx(-baseline.COST_LOG, rel=1e-9)
        assert gap.max() == pytest.approx(-baseline.COST_LOG, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. random_5pct is reproducible under the same seed
# ---------------------------------------------------------------------------


def test_random_5pct_is_deterministic(symbol: str) -> None:
    df = _labeled(symbol)

    first = baseline.select_random(df, seed=baseline.DEFAULT_SEED)
    second = baseline.select_random(df, seed=baseline.DEFAULT_SEED)
    assert first.equals(second)

    # And the metrics built from it match end to end.
    a = baseline.evaluate_symbol(df, symbol, baseline.DEFAULT_SEED)[0]
    b = baseline.evaluate_symbol(df, symbol, baseline.DEFAULT_SEED)[0]
    assert a.equals(b)


def test_random_5pct_selects_five_percent(symbol: str, trades: dict[str, pl.DataFrame]) -> None:
    eligible = _labeled(symbol).filter(pl.col("label").is_not_null()).height
    assert trades["random_5pct"].height == round(baseline.RANDOM_FRACTION * eligible)


def test_random_5pct_differs_under_a_different_seed(symbol: str) -> None:
    """Guards against the selection ignoring the seed entirely."""
    df = _labeled(symbol)
    a = baseline.select_random(df, seed=baseline.DEFAULT_SEED)
    b = baseline.select_random(df, seed=baseline.DEFAULT_SEED + 1)
    assert not a.equals(b)


# ---------------------------------------------------------------------------
# 3. Win rate is a proportion
# ---------------------------------------------------------------------------


def test_win_rate_in_unit_interval(symbol: str, results: pl.DataFrame) -> None:
    for row in results.iter_rows(named=True):
        win_rate = row["win_rate"]
        if win_rate is None:
            assert row["trades"] == 0
            continue
        assert 0.0 <= win_rate <= 1.0, f"{symbol}/{row['strategy']}: {win_rate}"


# ---------------------------------------------------------------------------
# 4. Profit factor is positive when both wins and losses exist
# ---------------------------------------------------------------------------


def test_profit_factor_positive_with_wins_and_losses(
    symbol: str, results: pl.DataFrame, trades: dict[str, pl.DataFrame]
) -> None:
    checked = 0
    for row in results.iter_rows(named=True):
        frame = trades[row["strategy"]]
        if frame.height == 0:
            continue
        net = frame["ret_net"]
        if net.filter(net > 0).len() == 0 or net.filter(net < 0).len() == 0:
            continue
        checked += 1
        pf = row["profit_factor"]
        assert pf is not None and pf > 0, f"{symbol}/{row['strategy']}: pf={pf}"
    assert checked > 0, f"{symbol}: no strategy had both wins and losses"


def test_profit_factor_null_without_losses(
    symbol: str, results: pl.DataFrame, trades: dict[str, pl.DataFrame]
) -> None:
    for row in results.iter_rows(named=True):
        frame = trades[row["strategy"]]
        if frame.height == 0:
            continue
        net = frame["ret_net"]
        if net.filter(net < 0).len() == 0:
            assert row["profit_factor"] is None


# ---------------------------------------------------------------------------
# 5. always_trade covers exactly the labelled bars
# ---------------------------------------------------------------------------


def test_always_trade_count_equals_non_null_labels(
    symbol: str, results: pl.DataFrame
) -> None:
    expected = _labeled(symbol).filter(pl.col("label").is_not_null()).height
    actual = results.filter(pl.col("strategy") == "always_trade")["trades"][0]
    assert actual == expected


def test_buy_and_hold_is_a_single_trade(symbol: str, results: pl.DataFrame) -> None:
    assert results.filter(pl.col("strategy") == "buy_and_hold")["trades"][0] == 1
