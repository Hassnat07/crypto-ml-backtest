"""Triple-barrier labelling over the feature parquets.

For each bar i we simulate a long entered at that bar's close and look
forward at most H bars for the first touch of a volatility-scaled profit
target (upper barrier) or stop (lower barrier). If neither is touched the
vertical barrier expires and the trade is labelled a no-trade.

Barrier touches are checked against each forward bar's intrabar high/low,
not its close — using closes only would systematically undercount touches.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_HORIZON = 30
DEFAULT_PT_MULT = 1.0
DEFAULT_SL_MULT = 1.0

LABEL_COLUMNS = ["label", "t1", "ret", "bars_held", "barrier_hit", "ambiguous"]


def triple_barrier_labels(
    df: pl.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
) -> pl.DataFrame:
    """Attach triple-barrier label columns to a feature frame.

    A row is labelled only when its outcome is fully determined by bars at
    or before t1: either a barrier was touched, or the full horizon elapsed.
    A row whose horizon runs past the end of the data without a touch is
    genuinely undetermined and stays null — this is what keeps a label from
    depending on any bar after its own t1.
    """
    df = df.sort("open_time")

    open_time = df["open_time"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()
    sigma = df["realized_vol_30"].to_list()

    n = df.height
    sqrt_h = math.sqrt(horizon)

    label: list[int | None] = [None] * n
    t1: list[datetime | None] = [None] * n
    ret: list[float | None] = [None] * n
    bars_held: list[int | None] = [None] * n
    barrier_hit: list[str | None] = [None] * n
    ambiguous: list[bool] = [False] * n

    for i in range(n):
        sigma_i = sigma[i]
        if sigma_i is None:
            continue

        entry = close[i]
        horizon_vol = sigma_i * sqrt_h
        upper = entry * (1.0 + pt_mult * horizon_vol)
        lower = entry * (1.0 - sl_mult * horizon_vol)

        last_j = min(i + horizon, n - 1)
        touched = False

        for j in range(i + 1, last_j + 1):
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and hit_lower:
                # Both barriers touched inside one bar: OHLC alone cannot say
                # which came first, so the direction stays unknown.
                t1[i] = open_time[j]
                bars_held[i] = j - i
                ambiguous[i] = True
                touched = True
                break
            if hit_upper:
                label[i] = 1
                t1[i] = open_time[j]
                ret[i] = math.log(upper / entry)
                bars_held[i] = j - i
                barrier_hit[i] = "upper"
                touched = True
                break
            if hit_lower:
                label[i] = 0
                t1[i] = open_time[j]
                ret[i] = math.log(lower / entry)
                bars_held[i] = j - i
                barrier_hit[i] = "lower"
                touched = True
                break

        if touched or i + horizon > n - 1:
            continue

        exit_idx = i + horizon
        label[i] = 0
        t1[i] = open_time[exit_idx]
        ret[i] = math.log(close[exit_idx] / entry)
        bars_held[i] = horizon
        barrier_hit[i] = "vertical"

    return df.with_columns(
        [
            pl.Series("label", label, dtype=pl.Int8),
            pl.Series("t1", t1, dtype=pl.Datetime("us")),
            pl.Series("ret", ret, dtype=pl.Float64),
            pl.Series("bars_held", bars_held, dtype=pl.Int32),
            pl.Series("barrier_hit", barrier_hit, dtype=pl.Utf8),
            pl.Series("ambiguous", ambiguous, dtype=pl.Boolean),
        ]
    )


def build_labels_for_symbol(
    symbol: str,
    parquet_dir: Path,
    horizon: int = DEFAULT_HORIZON,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
) -> pl.DataFrame:
    """Read one symbol's feature parquet and return it with labels attached."""
    src_path = parquet_dir / f"features_{symbol}.parquet"
    df = pl.read_parquet(src_path)
    return triple_barrier_labels(df, horizon, pt_mult, sl_mult)


def print_summary(
    symbol: str,
    df: pl.DataFrame,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
) -> None:
    n = df.height
    print(f"\n=== {symbol} ===")
    print(f"rows: {n}  horizon: {horizon}  pt_mult: {pt_mult}  sl_mult: {sl_mult}")

    n_win = df.filter(pl.col("label") == 1).height
    n_loss = df.filter(pl.col("label") == 0).height
    n_null = df.filter(pl.col("label").is_null()).height
    print("label distribution:")
    for name, count in (("1 (win)", n_win), ("0 (loss/no-move)", n_loss), ("null", n_null)):
        print(f"  {name}: {count} ({100.0 * count / n:.2f}%)")

    print("barrier_hit distribution:")
    hit_counts = (
        df.group_by("barrier_hit").len().sort("len", descending=True).rows()
    )
    for hit, count in hit_counts:
        name = hit if hit is not None else "null (unlabelled/ambiguous)"
        print(f"  {name}: {count} ({100.0 * count / n:.2f}%)")

    print(f"ambiguous: {df.filter(pl.col('ambiguous')).height}")

    held = df["bars_held"].drop_nulls()
    print(f"bars_held mean: {held.mean():.2f}  median: {held.median():.1f}")

    width_pct = (
        df["realized_vol_30"].drop_nulls()
        * math.sqrt(horizon)
        * (pt_mult + sl_mult)
        / 2.0
        * 100.0
    )
    print(
        f"barrier width: mean {width_pct.mean():.2f}%  median "
        f"{width_pct.median():.2f}% of price"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to label (default: {DEFAULT_SYMBOLS})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Vertical barrier, in bars (default: {DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--pt-mult",
        type=float,
        default=DEFAULT_PT_MULT,
        help=f"Profit-target multiple of horizon vol (default: {DEFAULT_PT_MULT})",
    )
    parser.add_argument(
        "--sl-mult",
        type=float,
        default=DEFAULT_SL_MULT,
        help=f"Stop-loss multiple of horizon vol (default: {DEFAULT_SL_MULT})",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=PARQUET_DIR,
        help="Directory containing input/output parquet files",
    )
    args = parser.parse_args()

    for symbol in args.symbols:
        df = build_labels_for_symbol(
            symbol, args.parquet_dir, args.horizon, args.pt_mult, args.sl_mult
        )
        out_path = args.parquet_dir / f"labeled_{symbol}.parquet"
        df.write_parquet(out_path)
        print_summary(symbol, df, args.horizon, args.pt_mult, args.sl_mult)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
