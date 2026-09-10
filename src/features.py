"""Compute point-in-time model features from 4h kline parquet files.

Reads data/parquet/{SYMBOL}_4h.parquet and writes
data/parquet/features_{SYMBOL}.parquet with the original columns plus
engineered features. Every feature at row i uses only rows <= i (rolling
windows, no negative shifts, no whole-column aggregates).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
BTC_SYMBOL = "BTCUSDT"

EXPECTED_INTERVAL_US = 4 * 60 * 60 * 1_000_000  # 4h, in microseconds


def return_features() -> list[pl.Expr]:
    """Log returns of close over 1, 6, 12, 30, 90 bars back."""
    log_close = pl.col("close").log()
    exprs = []
    for n in (1, 6, 12, 30, 90):
        exprs.append((log_close - log_close.shift(n)).alias(f"ret_{n}"))
    return exprs


def volatility_features() -> list[pl.Expr]:
    """Rolling realized vol of ret_1, vol ratio, and normalised ATR-14."""
    ret_1 = (pl.col("close").log() - pl.col("close").log().shift(1)).alias("ret_1")
    realized_vol_30 = ret_1.rolling_std(window_size=30).alias("realized_vol_30")
    realized_vol_90 = ret_1.rolling_std(window_size=90).alias("realized_vol_90")
    vol_ratio = (realized_vol_30 / realized_vol_90).alias("vol_ratio")

    prev_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    atr_14 = (true_range.rolling_mean(window_size=14) / pl.col("close")).alias(
        "atr_14"
    )

    return [realized_vol_30, realized_vol_90, vol_ratio, atr_14]


def momentum_features() -> list[pl.Expr]:
    """RSI-14, close-vs-SMA ratios, and normalised MACD histogram."""
    delta = pl.col("close").diff(1)
    gain = (
        pl.when(delta.is_null())
        .then(None)
        .otherwise(pl.when(delta > 0).then(delta).otherwise(0.0))
    )
    loss = (
        pl.when(delta.is_null())
        .then(None)
        .otherwise(pl.when(delta < 0).then(-delta).otherwise(0.0))
    )
    avg_gain = gain.ewm_mean(alpha=1.0 / 14, adjust=False, min_samples=14)
    avg_loss = loss.ewm_mean(alpha=1.0 / 14, adjust=False, min_samples=14)
    rs = avg_gain / avg_loss
    rsi_14 = (100.0 - (100.0 / (1.0 + rs))).alias("rsi_14")

    sma_30 = pl.col("close").rolling_mean(window_size=30)
    sma_90 = pl.col("close").rolling_mean(window_size=90)
    close_over_sma_30 = (pl.col("close") / sma_30 - 1.0).alias("close_over_sma_30")
    close_over_sma_90 = (pl.col("close") / sma_90 - 1.0).alias("close_over_sma_90")
    sma_30_over_sma_90 = (sma_30 / sma_90 - 1.0).alias("sma_30_over_sma_90")

    ema_12 = pl.col("close").ewm_mean(span=12, adjust=False, min_samples=12)
    ema_26 = pl.col("close").ewm_mean(span=26, adjust=False, min_samples=26)
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm_mean(span=9, adjust=False, min_samples=9)
    macd_hist = ((macd_line - macd_signal) / pl.col("close")).alias("macd_hist")

    return [
        rsi_14,
        close_over_sma_30,
        close_over_sma_90,
        sma_30_over_sma_90,
        macd_hist,
    ]


def volume_features() -> list[pl.Expr]:
    """Rolling z-scores of volume/trade count and taker-buy ratio features."""
    vol_mean_30 = pl.col("volume").rolling_mean(window_size=30)
    vol_std_30 = pl.col("volume").rolling_std(window_size=30)
    vol_zscore_30 = ((pl.col("volume") - vol_mean_30) / vol_std_30).alias(
        "vol_zscore_30"
    )

    taker_buy_ratio = (pl.col("taker_buy_base") / pl.col("volume")).alias(
        "taker_buy_ratio"
    )
    taker_buy_ratio_ma_30 = taker_buy_ratio.rolling_mean(window_size=30).alias(
        "taker_buy_ratio_ma_30"
    )

    tc_mean_30 = pl.col("trade_count").rolling_mean(window_size=30)
    tc_std_30 = pl.col("trade_count").rolling_std(window_size=30)
    trade_count_zscore_30 = (
        (pl.col("trade_count") - tc_mean_30) / tc_std_30
    ).alias("trade_count_zscore_30")

    return [
        vol_zscore_30,
        taker_buy_ratio,
        taker_buy_ratio_ma_30,
        trade_count_zscore_30,
    ]


def range_features() -> list[pl.Expr]:
    """Candle-shape features: high/low range and wick sizes, normalised by close."""
    high_low_range = ((pl.col("high") - pl.col("low")) / pl.col("close")).alias(
        "high_low_range"
    )
    close_position = (
        pl.when(pl.col("high") != pl.col("low"))
        .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .otherwise(None)
        .alias("close_position")
    )
    upper_wick = (
        (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
        / pl.col("close")
    ).alias("upper_wick")
    lower_wick = (
        (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
        / pl.col("close")
    ).alias("lower_wick")

    return [high_low_range, close_position, upper_wick, lower_wick]


def calendar_features() -> list[pl.Expr]:
    """Sin/cos encodings of hour-of-day and day-of-week (raw ints dropped)."""
    hour = pl.col("open_time").dt.hour()
    dow = pl.col("open_time").dt.weekday() - 1  # polars: 1=Mon..7=Sun -> 0..6

    hour_sin = (2 * 3.141592653589793 * hour / 24).sin().alias("hour_sin")
    hour_cos = (2 * 3.141592653589793 * hour / 24).cos().alias("hour_cos")
    dow_sin = (2 * 3.141592653589793 * dow / 7).sin().alias("dow_sin")
    dow_cos = (2 * 3.141592653589793 * dow / 7).cos().alias("dow_cos")

    return [hour_sin, hour_cos, dow_sin, dow_cos]


def gap_feature() -> list[pl.Expr]:
    """Boolean flag marking bars whose gap from the previous bar exceeds 4h."""
    delta_us = pl.col("open_time").diff(1).dt.total_microseconds()
    after_gap = (delta_us > EXPECTED_INTERVAL_US).fill_null(False).alias("after_gap")
    return [after_gap]


def cross_asset_features(is_btc: bool) -> list[pl.Expr]:
    """Rolling correlation with BTC's ret_1 (null for BTCUSDT itself).

    Assumes btc_ret_1, btc_ret_6, btc_ret_30 have already been joined onto
    the frame from a point-in-time-safe BTC returns table.
    """
    if is_btc:
        corr = pl.lit(None, dtype=pl.Float64).alias("corr_with_btc_90")
    else:
        corr = pl.rolling_corr(
            pl.col("ret_1"), pl.col("btc_ret_1"), window_size=90
        ).alias("corr_with_btc_90")
    return [corr]


def _load_btc_returns(parquet_dir: Path) -> pl.DataFrame:
    """Load BTC bars and compute its own ret_1/ret_6/ret_30 for joining."""
    btc_path = parquet_dir / f"{BTC_SYMBOL}_4h.parquet"
    btc = pl.read_parquet(btc_path).sort("open_time")
    log_close = pl.col("close").log()
    btc = btc.with_columns(
        [
            (log_close - log_close.shift(1)).alias("btc_ret_1"),
            (log_close - log_close.shift(6)).alias("btc_ret_6"),
            (log_close - log_close.shift(30)).alias("btc_ret_30"),
        ]
    )
    return btc.select(["open_time", "btc_ret_1", "btc_ret_6", "btc_ret_30"])


def compute_features(df: pl.DataFrame, btc_returns: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Compute all feature families for one symbol's bars.

    `btc_returns` is the (open_time, btc_ret_1, btc_ret_6, btc_ret_30) frame,
    already point-in-time safe, joined in before the cross-asset and
    correlation features are derived.
    """
    df = df.sort("open_time")

    df = df.with_columns(
        return_features()
        + volatility_features()
        + momentum_features()
        + volume_features()
        + range_features()
        + calendar_features()
        + gap_feature()
    )

    df = df.join(btc_returns, on="open_time", how="left")

    is_btc = symbol == BTC_SYMBOL
    df = df.with_columns(cross_asset_features(is_btc))

    df = df.drop(["btc_ret_1"])

    return df


FEATURE_COLUMNS = [
    "ret_1",
    "ret_6",
    "ret_12",
    "ret_30",
    "ret_90",
    "realized_vol_30",
    "realized_vol_90",
    "vol_ratio",
    "atr_14",
    "rsi_14",
    "close_over_sma_30",
    "close_over_sma_90",
    "sma_30_over_sma_90",
    "macd_hist",
    "vol_zscore_30",
    "taker_buy_ratio",
    "taker_buy_ratio_ma_30",
    "trade_count_zscore_30",
    "high_low_range",
    "close_position",
    "upper_wick",
    "lower_wick",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "after_gap",
    "btc_ret_6",
    "btc_ret_30",
    "corr_with_btc_90",
]


def build_features_for_symbol(symbol: str, parquet_dir: Path) -> pl.DataFrame:
    """Read one symbol's raw parquet and return it with features attached."""
    src_path = parquet_dir / f"{symbol}_4h.parquet"
    df = pl.read_parquet(src_path)
    btc_returns = _load_btc_returns(parquet_dir)
    return compute_features(df, btc_returns, symbol)


def print_summary(symbol: str, df: pl.DataFrame) -> None:
    print(f"\n=== {symbol} ===")
    print(f"rows: {len(df)}  feature columns: {len(FEATURE_COLUMNS)}")

    null_counts = df.select(
        [pl.col(c).is_null().sum().alias(c) for c in FEATURE_COLUMNS]
    ).row(0)
    print("null count per feature:")
    for col, n in zip(FEATURE_COLUMNS, null_counts):
        print(f"  {col}: {n}")

    all_non_null_mask = pl.fold(
        acc=pl.lit(True),
        function=lambda a, b: a & b,
        exprs=[pl.col(c).is_not_null() for c in FEATURE_COLUMNS],
    )
    first_full_row = df.filter(all_non_null_mask).head(1)
    if len(first_full_row) == 0:
        print("first fully non-null row: none found")
    else:
        idx = df.with_row_index().filter(all_non_null_mask).head(1)["index"][0]
        print(f"first fully non-null row: index {idx}, open_time {first_full_row['open_time'][0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to process (default: {DEFAULT_SYMBOLS})",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PARQUET_DIR,
        help="Directory containing input/output parquet files",
    )
    args = parser.parse_args()

    for symbol in args.symbols:
        df = build_features_for_symbol(symbol, args.parquet_dir)
        out_path = args.parquet_dir / f"features_{symbol}.parquet"
        df.write_parquet(out_path)
        print_summary(symbol, df)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
