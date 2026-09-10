"""Tests for src/cost_analysis.py."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cost_analysis as ca  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


@pytest.fixture(scope="module", params=SYMBOLS)
def symbol(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def preds(symbol: str) -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"cv_predictions_{symbol}.parquet").sort(
        "open_time"
    )


@pytest.fixture(scope="module")
def marked(symbol: str, preds: pl.DataFrame) -> pl.DataFrame:
    labeled = pl.read_parquet(PARQUET_DIR / f"labeled_{symbol}.parquet")
    return ca.attach_fill_outcome(preds, labeled)


# ---------------------------------------------------------------------------
# 1. Zero cost is a true no-op
# ---------------------------------------------------------------------------


def test_zero_cost_equals_gross_exactly(preds: pl.DataFrame) -> None:
    assert ca.cost_in_log_space(0.0) == 0.0

    ret = preds["ret"].to_numpy()
    np.testing.assert_array_equal(ca.net_returns(ret, 0.0), ret)

    for thr in ca.THRESHOLDS:
        signals = ca.signals_at(preds, thr)
        if len(signals) == 0:
            continue
        assert ca.mean_net(signals, 0.0) == float(signals.mean())


# ---------------------------------------------------------------------------
# 2. Higher cost strictly lowers net return
# ---------------------------------------------------------------------------


def test_net_return_decreases_with_cost(symbol: str, preds: pl.DataFrame) -> None:
    ascending = sorted(ca.COSTS)
    for thr in ca.THRESHOLDS:
        signals = ca.signals_at(preds, thr)
        if len(signals) == 0:
            continue
        values = [ca.mean_net(signals, c) for c in ascending]
        for cheaper, dearer in zip(values, values[1:]):
            assert dearer < cheaper, (
                f"{symbol} thr {thr}: net did not fall as cost rose "
                f"({cheaper} -> {dearer})"
            )


def test_cost_is_a_constant_shift(preds: pl.DataFrame) -> None:
    """Cost is additive in log space, so it cannot reorder thresholds."""
    def ranking(cost: float) -> list[float]:
        scored = []
        for thr in ca.THRESHOLDS:
            signals = ca.signals_at(preds, thr)
            if len(signals) >= ca.MIN_SIGNALS:
                scored.append((ca.mean_net(signals, cost), thr))
        return [thr for _, thr in sorted(scored, reverse=True)]

    assert ranking(0.0) == ranking(ca.TAKER_COST)


# ---------------------------------------------------------------------------
# 3. Break-even cost is never negative
# ---------------------------------------------------------------------------


def test_break_even_non_negative(symbol: str, preds: pl.DataFrame) -> None:
    stats = ca.analyse_costs(symbol, preds)
    assert stats.break_even >= 0.0


def test_break_even_matches_the_sweep(symbol: str, preds: pl.DataFrame) -> None:
    """The closed form must agree with brute force on both sides."""
    stats = ca.analyse_costs(symbol, preds)
    if stats.break_even <= 0.0:
        assert stats.best_mean_gross is None or stats.best_mean_gross <= 0
        return

    signals = ca.signals_at(preds, stats.best_threshold)
    assert ca.mean_net(signals, stats.break_even - 1e-6) > 0
    assert ca.mean_net(signals, stats.break_even + 1e-6) < 0


def test_break_even_of_negative_edge_is_zero() -> None:
    assert ca.break_even_cost(-0.01) == 0.0
    assert ca.break_even_cost(0.0) == 0.0
    assert ca.break_even_cost(None) == 0.0
    assert ca.break_even_cost(0.01) == pytest.approx(1 - np.exp(-0.01))


# ---------------------------------------------------------------------------
# 4 & 5. Fill accounting
# ---------------------------------------------------------------------------


def test_fill_rate_in_unit_interval(symbol: str, marked: pl.DataFrame) -> None:
    for thr in ca.THRESHOLDS:
        stats = ca.fill_stats(marked, thr)
        if stats.fill_rate is None:
            assert stats.n_signals == 0
            continue
        assert 0.0 <= stats.fill_rate <= 1.0, f"{symbol} thr {thr}: {stats.fill_rate}"


def test_filled_plus_missed_equals_signals(symbol: str, marked: pl.DataFrame) -> None:
    for thr in ca.THRESHOLDS:
        stats = ca.fill_stats(marked, thr)
        assert stats.n_filled + stats.n_missed == stats.n_signals


def test_fill_outcome_never_null(marked: pl.DataFrame) -> None:
    assert marked["filled"].null_count() == 0
    assert marked.height > 0


# ---------------------------------------------------------------------------
# 6. Synthetic fills with a known answer
# ---------------------------------------------------------------------------


def _synthetic() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Five bars, all closing at 100, with hand-picked next-bar lows.

    row 0 -> next low  99 <= 100  FILL
    row 1 -> next low 101 >  100  MISS
    row 2 -> next low  98 <= 100  FILL
    row 3 -> next low 102 >  100  MISS
    row 4 -> no next bar          MISS (by rule)
    """
    times = [datetime(2024, 1, 1) + timedelta(hours=4 * i) for i in range(5)]
    labeled = pl.DataFrame(
        {
            "open_time": times,
            "close": [100.0] * 5,
            "low": [999.0, 99.0, 101.0, 98.0, 102.0],
        }
    )
    preds = pl.DataFrame(
        {
            "open_time": times,
            "fold": [1] * 5,
            "y_true": [1, 0, 1, 0, 1],
            "y_pred_proba": [0.9, 0.9, 0.9, 0.9, 0.9],
            "ret": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    return preds, labeled


def test_synthetic_fill_rate_is_exact() -> None:
    preds, labeled = _synthetic()
    marked = ca.attach_fill_outcome(preds, labeled)

    assert marked["filled"].to_list() == [True, False, True, False, False]

    stats = ca.fill_stats(marked, 0.5)
    assert stats.n_signals == 5
    assert stats.n_filled == 2
    assert stats.n_missed == 3
    assert stats.fill_rate == pytest.approx(0.4)

    # Filled rows are 0 and 2 -> mean ret 0.02; missed are 1, 3, 4 -> 0.11/3.
    assert stats.mean_filled == pytest.approx(0.02)
    assert stats.mean_missed == pytest.approx((0.02 + 0.04 + 0.05) / 3)
    assert stats.adverse_gap == pytest.approx(stats.mean_missed - stats.mean_filled)


def test_synthetic_threshold_filters_signals() -> None:
    preds, labeled = _synthetic()
    preds = preds.with_columns(
        pl.Series("y_pred_proba", [0.9, 0.9, 0.9, 0.9, 0.1])
    )
    marked = ca.attach_fill_outcome(preds, labeled)

    stats = ca.fill_stats(marked, 0.5)
    assert stats.n_signals == 4
    assert stats.n_filled == 2
    assert stats.fill_rate == pytest.approx(0.5)


def test_synthetic_offset_reduces_fill_rate() -> None:
    """Posting below the market can only ever fill less often."""
    preds, labeled = _synthetic()
    marked = ca.attach_fill_outcome(preds, labeled)

    rates = []
    for offset in ca.FILL_OFFSETS:
        row = ca.offset_fill_row(marked, 0.5, offset)
        rates.append(row["fill_rate"])
    for wider, narrower in zip(rates, rates[1:]):
        assert narrower <= wider


# ---------------------------------------------------------------------------
# Maker vs taker accounting
# ---------------------------------------------------------------------------


def test_maker_vs_taker_counts_are_consistent(marked: pl.DataFrame) -> None:
    result = ca.maker_vs_taker(marked, 0.5)
    assert result["n_filled"] <= result["n_signals"]
    if result["n_signals"]:
        stats = ca.fill_stats(marked, 0.5)
        assert result["n_filled"] == stats.n_filled
