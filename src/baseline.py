"""Evaluate non-ML baseline strategies on the labelled bars.

These are the bar a model has to clear. Every strategy is scored on the
same triple-barrier trade outcomes, so the comparison isolates signal
selection: which bars a strategy chooses to trade, not how it exits.

Costs are applied in log space (log(1 - round_trip)) rather than by
subtracting the raw fraction, because trade returns are log returns and
are summed to form the equity curve.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_SEED = 42
RANDOM_FRACTION = 0.05

FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)  # 0.30%
COST_LOG = math.log(1.0 - ROUND_TRIP_COST)

BARS_PER_YEAR = 6 * 365  # 4h bars
SHARPE_SCALING = math.sqrt(BARS_PER_YEAR)

STRATEGY_NAMES = [
    "always_trade",
    "buy_and_hold",
    "rsi_oversold",
    "momentum",
    "mean_reversion",
    "random_5pct",
]

METRIC_COLUMNS = [
    "symbol",
    "strategy",
    "trades",
    "win_rate",
    "mean_ret_gross",
    "mean_ret_net",
    "total_ret",
    "ann_ret",
    "sharpe",
    "max_drawdown",
    "profit_factor",
    "mean_bars_held",
]

ASSUMPTIONS = """\
ASSUMPTIONS
  Trades are scored independently and MAY OVERLAP in time. This is a
  measure of signal quality, not a live portfolio -- a real book could not
  hold every one of these positions simultaneously. Position sizing is
  ignored; each trade is one unit.
  Costs: {fee:.2%} fee + {slip:.2%} slippage per side = {rt:.2%} round trip,
  charged once per trade (buy_and_hold is a single trade, so it pays once).
  Sharpe is annualised as mean/std of per-TRADE returns * sqrt(6*365).
  Trades average ~20 bars and overlap, so this overstates a true per-bar
  portfolio Sharpe; use it to rank strategies, not as an achievable figure.
  On a handful of trades it is meaningless -- buy_and_hold's is noise.
  total/ann/maxDD are sums over overlapping unit-size trades. They measure
  aggregate signal edge, NOT an attainable portfolio return path.\
""".format(fee=FEE_PER_SIDE, slip=SLIPPAGE_PER_SIDE, rt=ROUND_TRIP_COST)


def _eligible(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with a resolved triple-barrier outcome."""
    return df.filter(pl.col("label").is_not_null())


def select_always_trade(df: pl.DataFrame) -> pl.DataFrame:
    return _eligible(df)


def select_rsi_oversold(df: pl.DataFrame) -> pl.DataFrame:
    return _eligible(df).filter(pl.col("rsi_14") < 30)


def select_momentum(df: pl.DataFrame) -> pl.DataFrame:
    return _eligible(df).filter(
        (pl.col("close_over_sma_30") > 0) & (pl.col("sma_30_over_sma_90") > 0)
    )


def select_mean_reversion(df: pl.DataFrame) -> pl.DataFrame:
    return _eligible(df).filter(pl.col("close_over_sma_30") < -0.05)


def select_random(df: pl.DataFrame, seed: int = DEFAULT_SEED) -> pl.DataFrame:
    """Uniform 5% sample of eligible bars.

    Uses a local Random instance so the draw depends only on `seed` and the
    eligible row count -- never on global interpreter state or call order.
    """
    eligible = _eligible(df)
    k = round(RANDOM_FRACTION * eligible.height)
    if k == 0:
        return eligible.clear()
    picks = sorted(random.Random(seed).sample(range(eligible.height), k))
    return eligible[picks]


def _trade_frame(rows: pl.DataFrame, symbol: str, strategy: str) -> pl.DataFrame:
    """Normalise selected bars into a uniform per-trade frame."""
    return rows.select(
        pl.lit(symbol).alias("symbol"),
        pl.lit(strategy).alias("strategy"),
        pl.col("open_time"),
        pl.col("ret").alias("ret_gross"),
        (pl.col("ret") + COST_LOG).alias("ret_net"),
        pl.col("label").cast(pl.Int8),
        pl.col("bars_held").cast(pl.Int32),
    )


def buy_and_hold_trade(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """The single hold-the-whole-sample trade.

    Enters at the close of the first bar carrying a resolved label and exits
    at the final bar's close. The triple-barrier label does not apply to a
    trade with no barriers, so `label` is set from the realised sign of the
    after-cost return -- that keeps win_rate meaningful in the shared table.
    """
    indexed = df.with_row_index("_idx")
    eligible = _eligible(indexed)
    empty = _trade_frame(indexed.clear(), symbol, "buy_and_hold")
    if eligible.height == 0:
        return empty

    entry = eligible.row(0, named=True)
    exit_close = df["close"][-1]
    ret_gross = math.log(exit_close / entry["close"])

    return pl.DataFrame(
        {
            "symbol": [symbol],
            "strategy": ["buy_and_hold"],
            "open_time": [entry["open_time"]],
            "ret_gross": [ret_gross],
            "ret_net": [ret_gross + COST_LOG],
            "label": [1 if ret_gross + COST_LOG > 0 else 0],
            "bars_held": [df.height - 1 - entry["_idx"]],
        },
        schema=empty.schema,
    )


def build_trades(
    df: pl.DataFrame, symbol: str, seed: int = DEFAULT_SEED
) -> dict[str, pl.DataFrame]:
    """Per-trade frames for every strategy on one symbol."""
    return {
        "always_trade": _trade_frame(select_always_trade(df), symbol, "always_trade"),
        "buy_and_hold": buy_and_hold_trade(df, symbol),
        "rsi_oversold": _trade_frame(select_rsi_oversold(df), symbol, "rsi_oversold"),
        "momentum": _trade_frame(select_momentum(df), symbol, "momentum"),
        "mean_reversion": _trade_frame(
            select_mean_reversion(df), symbol, "mean_reversion"
        ),
        "random_5pct": _trade_frame(select_random(df, seed), symbol, "random_5pct"),
    }


def _max_drawdown(ret_net: pl.Series) -> float:
    """Peak-to-trough decline of the cumulative curve, in log-return units.

    Deliberately NOT converted to a percentage of capital. The curve is a sum
    of overlapping, unit-size trade log returns rather than a compounding
    equity path, so a 1-exp() conversion would imply compounding that never
    happened -- and saturates at ~100% for any strategy whose curve drops
    more than ~1.0 in log terms, which is most of them.

    The curve is seeded at 0 so a drawdown starting with the first trade is
    measured from inception rather than from that trade's own level.
    """
    peak = 0.0
    worst = 0.0
    running = 0.0
    for r in ret_net.to_list():
        running += r
        peak = max(peak, running)
        worst = min(worst, running - peak)
    return abs(worst)  # abs, not negation: keeps a flat curve at 0.0, not -0.0


def compute_metrics(
    trades: pl.DataFrame, symbol: str, strategy: str, span_bars: int
) -> dict[str, object]:
    """Summary statistics for one strategy's trade set."""
    n = trades.height
    if n == 0:
        return {
            "symbol": symbol,
            "strategy": strategy,
            "trades": 0,
            "win_rate": None,
            "mean_ret_gross": None,
            "mean_ret_net": None,
            "total_ret": 0.0,
            "ann_ret": None,
            "sharpe": None,
            "max_drawdown": None,
            "profit_factor": None,
            "mean_bars_held": None,
        }

    net = trades["ret_net"]
    total_ret = float(net.sum())

    gross_wins = float(net.filter(net > 0).sum())
    gross_losses = float(-net.filter(net < 0).sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else None

    std = net.std()
    sharpe = (
        float(net.mean() / std * SHARPE_SCALING)
        if n > 1 and std is not None and std > 0
        else None
    )

    return {
        "symbol": symbol,
        "strategy": strategy,
        "trades": n,
        "win_rate": float((trades["label"] == 1).mean()),
        "mean_ret_gross": float(trades["ret_gross"].mean()),
        "mean_ret_net": float(net.mean()),
        "total_ret": total_ret,
        "ann_ret": total_ret * BARS_PER_YEAR / span_bars if span_bars else None,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(net),
        "profit_factor": profit_factor,
        "mean_bars_held": float(trades["bars_held"].mean()),
    }


def evaluate_symbol(
    df: pl.DataFrame, symbol: str, seed: int = DEFAULT_SEED
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Metric rows and raw trade frames for one symbol."""
    trades = build_trades(df, symbol, seed)
    rows = [
        compute_metrics(trades[name], symbol, name, df.height)
        for name in STRATEGY_NAMES
    ]
    return pl.DataFrame(rows).select(METRIC_COLUMNS), trades


def _fmt(value: object, width: int, precision: int, scale: float = 1.0) -> str:
    """Right-justified cell; a missing value renders as a padded dash."""
    if value is None:
        return "-".rjust(width)
    return format(value * scale, f">{width}.{precision}f")


def print_table(title: str, results: pl.DataFrame) -> None:
    print(f"\n{title}")
    header = (
        f"{'strategy':<16}{'trades':>8}{'win%':>7}{'gross':>9}{'net':>9}"
        f"{'total':>9}{'ann':>9}{'sharpe':>9}{'maxDD':>8}{'PF':>7}{'bars':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in results.iter_rows(named=True):
        print(
            f"{row['strategy']:<16}"
            f"{row['trades']:>8}"
            f"{_fmt(row['win_rate'], 7, 1, 100)}"
            f"{_fmt(row['mean_ret_gross'], 9, 3, 100)}"
            f"{_fmt(row['mean_ret_net'], 9, 3, 100)}"
            f"{_fmt(row['total_ret'], 9, 2)}"
            f"{_fmt(row['ann_ret'], 9, 2)}"
            f"{_fmt(row['sharpe'], 9, 2)}"
            f"{_fmt(row['max_drawdown'], 8, 2)}"
            f"{_fmt(row['profit_factor'], 7, 2)}"
            f"{_fmt(row['mean_bars_held'], 9, 1)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for random_5pct (default: {DEFAULT_SEED})",
    )
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    args = parser.parse_args()

    print(ASSUMPTIONS)
    print(
        "\nAll returns are LOG returns. gross/net are mean per-trade log return"
        "\nx100 (~= % only while small; buy_and_hold's is large, so read it as log)."
        "\ntotal = sum of net log returns, ann = annualised, maxDD = peak-to-trough"
        "\nof the cumulative curve in log units, PF = profit factor, bars = mean held."
    )

    all_results: list[pl.DataFrame] = []
    pooled: dict[str, list[pl.DataFrame]] = {name: [] for name in STRATEGY_NAMES}
    span_bars = 0

    for symbol in args.symbols:
        df = pl.read_parquet(args.parquet_dir / f"labeled_{symbol}.parquet").sort(
            "open_time"
        )
        results, trades = evaluate_symbol(df, symbol, args.seed)
        all_results.append(results)
        span_bars = max(span_bars, df.height)
        for name in STRATEGY_NAMES:
            pooled[name].append(trades[name])
        print_table(f"=== {symbol} ===", results)

    # Pooled across symbols: the same trades scored as one set. Annualised on
    # the longest single-symbol span, since the symbols cover one shared period.
    combined = pl.DataFrame(
        [
            compute_metrics(pl.concat(pooled[name]), "ALL", name, span_bars)
            for name in STRATEGY_NAMES
        ]
    ).select(METRIC_COLUMNS)
    all_results.append(combined)
    print_table("=== ALL SYMBOLS (pooled trades) ===", combined)

    out = pl.concat(all_results)
    out_path = args.parquet_dir / "baseline_results.parquet"
    out.write_parquet(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
