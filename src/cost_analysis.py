"""Pre-registered question: at what round-trip cost does the model become
profitable, and is that cost reachable given limit-order fill risk?

No retraining happens here. The CV predictions are read back as-is and only
the cost assumption is varied, so nothing in this file can flatter the model
by refitting it.

Two facts shape the whole analysis:

  1. Cost enters as log(1 - c), an ADDITIVE constant on a log return. So the
     ranking of thresholds is identical at every cost level, and break-even
     has a closed form -- c* = 1 - exp(-mean_gross) -- rather than being
     read off the sweep grid at its resolution.

  2. Cheaper execution means posting a limit order, which only fills if the
     market comes to you. Part 2 tests whether the bars that fill are the
     ones worth having. A cost saving that only lands on the losing half of
     the distribution is not a saving.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

COSTS = [0.0030, 0.0025, 0.0020, 0.0015, 0.0010, 0.0005, 0.0000]
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

TAKER_COST = 0.0030
MAKER_COST = 0.0015

# A threshold that fires a handful of times can post a spectacular mean by
# luck. Thresholds below this many signals are excluded from "best threshold"
# selection, though they still appear in the grid.
MIN_SIGNALS = 30

# Adverse selection is not claimed from a handful of missed trades.
MIN_MISSED_FOR_CLAIM = 30

# Supplementary to the specified fill rule. Posting AT close_i fills ~99.9% of
# the time on continuous 4h bars (the next bar opens at the prior close, so its
# low is at or below it by construction), which cannot discriminate. Posting
# BELOW the market is what actually trades fill probability against price
# improvement, so these offsets are swept to answer the underlying question.
FILL_OFFSETS = [0.0000, 0.0005, 0.0010, 0.0025, 0.0050]

SUCCESS_CRITERIA = """\
SUCCESS CRITERIA (pre-registered)
  (a) pooled mean net return > 0 at the best threshold
  (b) mean net return positive in >= 4 of 5 folds
  (c) profitable at a reachable cost: break-even round trip >= 0.15% (maker)\
"""


def cost_in_log_space(cost: float) -> float:
    """log(1 - c). Exactly 0.0 at c = 0, so zero cost is a true no-op."""
    return math.log(1.0 - cost)


def net_returns(ret: np.ndarray, cost: float) -> np.ndarray:
    return ret + cost_in_log_space(cost)


def mean_net(ret: np.ndarray, cost: float) -> float | None:
    if len(ret) == 0:
        return None
    return float(net_returns(ret, cost).mean())


@dataclass
class SymbolCosts:
    symbol: str
    best_threshold: float | None
    best_mean_gross: float | None
    n_best_signals: int
    break_even: float
    pooled_mean_net_taker: float | None
    folds_positive_taker: int
    n_folds: int


def signals_at(preds: pl.DataFrame, threshold: float) -> np.ndarray:
    return preds.filter(pl.col("y_pred_proba") >= threshold)["ret"].to_numpy()


def best_threshold(preds: pl.DataFrame) -> tuple[float | None, float | None, int]:
    """Threshold with the highest mean GROSS return, subject to MIN_SIGNALS.

    Gross is the right ranking key because cost is additive in log space, so
    this choice does not depend on which cost level we later assume.
    """
    best: tuple[float | None, float | None, int] = (None, None, 0)
    for thr in THRESHOLDS:
        ret = signals_at(preds, thr)
        if len(ret) < MIN_SIGNALS:
            continue
        mean_gross = float(ret.mean())
        if best[1] is None or mean_gross > best[1]:
            best = (thr, mean_gross, len(ret))
    return best


def break_even_cost(mean_gross: float | None) -> float:
    """Highest round trip still leaving a positive mean, clamped at zero.

    mean_gross + log(1 - c) > 0  =>  c < 1 - exp(-mean_gross).
    A non-positive mean_gross loses money before any cost at all, so the
    reachable break-even is 0.0 -- there is no cost level that rescues it.
    """
    if mean_gross is None or mean_gross <= 0:
        return 0.0
    return 1.0 - math.exp(-mean_gross)


def print_cost_grid(symbol: str, preds: pl.DataFrame) -> None:
    print(f"\n  {symbol}: mean net return per trade (%), thresholds x round-trip cost")
    header = "    thr   n_sig" + "".join(f"{c * 100:>9.2f}%" for c in COSTS)
    print(header)
    print("    " + "-" * (len(header) - 4))

    for thr in THRESHOLDS:
        ret = signals_at(preds, thr)
        cells = ""
        for cost in COSTS:
            value = mean_net(ret, cost)
            if value is None:
                cells += f"{'-':>10}"
            else:
                # Trailing * marks a profitable cell.
                cells += f"{value * 100:>9.3f}" + ("*" if value > 0 else " ")
        flag = " " if len(ret) >= MIN_SIGNALS else "~"
        print(f"    {thr:>4.2f}{len(ret):>8}{flag}{cells}")

    print(f"    (* = mean_net > 0; ~ = fewer than {MIN_SIGNALS} signals, excluded from 'best')")


def print_fold_breakdown(preds: pl.DataFrame, threshold: float) -> None:
    folds = sorted(preds["fold"].unique().to_list())
    print(f"\n  per-fold mean net return (%) at threshold {threshold:.2f}")
    header = "    fold   n_sig" + "".join(f"{c * 100:>9.2f}%" for c in COSTS)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for fold in folds:
        ret = signals_at(preds.filter(pl.col("fold") == fold), threshold)
        cells = ""
        for cost in COSTS:
            value = mean_net(ret, cost)
            cells += (
                f"{'-':>10}"
                if value is None
                else f"{value * 100:>9.3f}" + ("*" if value > 0 else " ")
            )
        print(f"    {fold:>4}{len(ret):>8} {cells}")


def folds_positive(preds: pl.DataFrame, threshold: float, cost: float) -> tuple[int, int]:
    folds = sorted(preds["fold"].unique().to_list())
    positive = 0
    for fold in folds:
        ret = signals_at(preds.filter(pl.col("fold") == fold), threshold)
        value = mean_net(ret, cost)
        if value is not None and value > 0:
            positive += 1
    return positive, len(folds)


def analyse_costs(symbol: str, preds: pl.DataFrame) -> SymbolCosts:
    thr, mean_gross, n_signals = best_threshold(preds)
    pooled = mean_net(signals_at(preds, thr), TAKER_COST) if thr is not None else None
    positive, n_folds = (
        folds_positive(preds, thr, TAKER_COST) if thr is not None else (0, 0)
    )
    return SymbolCosts(
        symbol=symbol,
        best_threshold=thr,
        best_mean_gross=mean_gross,
        n_best_signals=n_signals,
        break_even=break_even_cost(mean_gross),
        pooled_mean_net_taker=pooled,
        folds_positive_taker=positive,
        n_folds=n_folds,
    )


# ---------------------------------------------------------------------------
# Part 2: limit-order fill realism
# ---------------------------------------------------------------------------


def attach_fill_outcome(preds: pl.DataFrame, labeled: pl.DataFrame) -> pl.DataFrame:
    """Mark each signal FILLED if the next bar trades down to the entry price.

    A limit buy resting at close_i fills only if bar i+1 dips to it, so this
    needs the NEXT bar's low. That is a deliberate forward look and is sound
    here -- the entry decision was already made at bar i using past data only,
    and the fill is part of the trade's forward window. It must never become
    a model feature.

    A signal on the final bar has no following bar to fill against; it is
    counted as MISSED so filled + missed always equals the signal count.
    """
    execution = labeled.sort("open_time").select(
        pl.col("open_time"),
        pl.col("close"),
        pl.col("low").shift(-1).alias("next_low"),
    )
    joined = preds.join(execution, on="open_time", how="left")
    return joined.with_columns(
        (
            pl.col("next_low").is_not_null() & (pl.col("next_low") <= pl.col("close"))
        ).alias("filled")
    )


@dataclass
class FillStats:
    threshold: float
    n_signals: int
    n_filled: int
    n_missed: int
    fill_rate: float | None
    mean_filled: float | None
    mean_missed: float | None
    adverse_gap: float | None


def fill_stats(marked: pl.DataFrame, threshold: float) -> FillStats:
    at = marked.filter(pl.col("y_pred_proba") >= threshold)
    filled = at.filter(pl.col("filled"))["ret"].to_numpy()
    missed = at.filter(~pl.col("filled"))["ret"].to_numpy()

    mean_f = float(filled.mean()) if len(filled) else None
    mean_m = float(missed.mean()) if len(missed) else None
    gap = None if mean_f is None or mean_m is None else mean_m - mean_f

    return FillStats(
        threshold=threshold,
        n_signals=at.height,
        n_filled=len(filled),
        n_missed=len(missed),
        fill_rate=len(filled) / at.height if at.height else None,
        mean_filled=mean_f,
        mean_missed=mean_m,
        adverse_gap=gap,
    )


def _fmt(value: float | None, width: int, precision: int, scale: float = 1.0) -> str:
    if value is None:
        return "-".rjust(width)
    return format(value * scale, f">{width}.{precision}f")


def print_fill_analysis(symbol: str, marked: pl.DataFrame) -> list[FillStats]:
    print(f"\n  {symbol}: limit-order fill analysis (gross returns, %)")
    header = (
        f"    {'thr':>5}{'signals':>9}{'filled':>8}{'missed':>8}"
        f"{'fill%':>8}{'mean_fill':>11}{'mean_miss':>11}{'adverse':>10}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))

    rows = []
    for thr in THRESHOLDS:
        s = fill_stats(marked, thr)
        rows.append(s)
        print(
            f"    {s.threshold:>5.2f}{s.n_signals:>9}{s.n_filled:>8}{s.n_missed:>8}"
            f"{_fmt(s.fill_rate, 8, 1, 100)}"
            f"{_fmt(s.mean_filled, 11, 3, 100)}"
            f"{_fmt(s.mean_missed, 11, 3, 100)}"
            f"{_fmt(s.adverse_gap, 10, 3, 100)}"
        )

    total_missed = sum(r.n_missed for r in rows)
    if total_missed < MIN_MISSED_FOR_CLAIM:
        print(
            f"    DEGENERATE: only {total_missed} missed fills across all thresholds."
            "\n    Posting at close_i is not a real test on continuous 4h bars -- the"
            "\n    next bar opens at the prior close, so its low is at or below the"
            "\n    entry by construction and the order almost always fills. No"
            "\n    adverse selection can be measured from this. See the offset sweep."
        )
    return rows


def offset_fill_row(
    marked: pl.DataFrame, threshold: float, offset: float
) -> dict[str, object]:
    """Fill outcome for a limit posted `offset` BELOW the signal bar's close.

    Filling below the market improves the entry by -log(1 - offset), which is
    added to the filled trades' returns. The trades that do not fill are the
    ones that never traded down -- so this is where adverse selection, if it
    exists, becomes visible.
    """
    at = marked.filter(pl.col("y_pred_proba") >= threshold)
    if at.height == 0:
        return {"offset": offset, "n": 0}

    improvement = -math.log(1.0 - offset)
    priced = at.with_columns(
        (pl.col("close") * (1.0 - offset)).alias("limit_price")
    ).with_columns(
        (
            pl.col("next_low").is_not_null()
            & (pl.col("next_low") <= pl.col("limit_price"))
        ).alias("hit")
    )

    filled = priced.filter(pl.col("hit"))["ret"].to_numpy() + improvement
    missed = priced.filter(~pl.col("hit"))["ret"].to_numpy()

    maker_net = net_returns(filled, MAKER_COST) if len(filled) else np.array([])
    taker_net = net_returns(at["ret"].to_numpy(), TAKER_COST)

    return {
        "offset": offset,
        "n": at.height,
        "n_filled": len(filled),
        "fill_rate": len(filled) / at.height,
        "mean_filled": float(filled.mean()) if len(filled) else None,
        "mean_missed": float(missed.mean()) if len(missed) else None,
        "adverse": (
            float(missed.mean()) - float(filled.mean())
            if len(filled) and len(missed)
            else None
        ),
        "maker_mean": float(maker_net.mean()) if len(maker_net) else None,
        "maker_total": float(maker_net.sum()),
        "taker_total": float(taker_net.sum()),
    }


def print_offset_sweep(symbol: str, marked: pl.DataFrame, threshold: float) -> None:
    """Supplementary: trade fill probability against price improvement."""
    print(
        f"\n  {symbol}: SUPPLEMENTARY offset sweep at threshold {threshold:.2f}"
        "\n  (limit posted below close_i; filled trades gain the price improvement)"
    )
    header = (
        f"    {'offset':>8}{'filled':>8}{'fill%':>8}{'mean_fill':>11}"
        f"{'mean_miss':>11}{'adverse':>10}{'maker_tot':>11}{'taker_tot':>11}"
    )
    print(header)
    print("    " + "-" * (len(header) - 4))
    for offset in FILL_OFFSETS:
        row = offset_fill_row(marked, threshold, offset)
        if row.get("n", 0) == 0:
            continue
        print(
            f"    {offset * 100:>7.2f}%{row['n_filled']:>8}"
            f"{_fmt(row['fill_rate'], 8, 1, 100)}"
            f"{_fmt(row['mean_filled'], 11, 3, 100)}"
            f"{_fmt(row['mean_missed'], 11, 3, 100)}"
            f"{_fmt(row['adverse'], 10, 3, 100)}"
            f"{_fmt(row['maker_total'], 11, 3)}"
            f"{_fmt(row['taker_total'], 11, 3)}"
        )


# ---------------------------------------------------------------------------
# Part 3: maker vs taker for BTC
# ---------------------------------------------------------------------------


def maker_vs_taker(marked: pl.DataFrame, threshold: float) -> dict[str, object]:
    """Maker fills a subset at 0.15%; taker fills everything at 0.30%.

    Reported per trade AND in total. Per-trade flatters the maker route when
    it declines trades; total is the honest comparison over one signal set,
    since a missed signal earns nothing rather than being reinvested.
    """
    at = marked.filter(pl.col("y_pred_proba") >= threshold)
    all_ret = at["ret"].to_numpy()
    filled_ret = at.filter(pl.col("filled"))["ret"].to_numpy()

    maker_net = net_returns(filled_ret, MAKER_COST) if len(filled_ret) else np.array([])
    taker_net = net_returns(all_ret, TAKER_COST) if len(all_ret) else np.array([])

    return {
        "n_signals": at.height,
        "n_filled": len(filled_ret),
        "maker_mean": float(maker_net.mean()) if len(maker_net) else None,
        "maker_total": float(maker_net.sum()),
        "taker_mean": float(taker_net.mean()) if len(taker_net) else None,
        "taker_total": float(taker_net.sum()),
    }


def print_combined_verdict(symbol: str, marked: pl.DataFrame, threshold: float) -> None:
    print(f"\n{'=' * 78}")
    print(f"PART 3 - COMBINED VERDICT: {symbol} at threshold {threshold:.2f}")
    print("=" * 78)

    result = maker_vs_taker(marked, threshold)
    print(
        f"\n  signals {result['n_signals']}, filled {result['n_filled']} "
        f"({result['n_filled'] / result['n_signals']:.1%})"
    )
    print(
        f"\n  {'route':<28}{'trades':>9}{'mean_net':>11}{'total_net':>12}"
    )
    print("  " + "-" * 58)
    print(
        f"  {'maker 0.15%, filled only':<28}{result['n_filled']:>9}"
        f"{_fmt(result['maker_mean'], 11, 3, 100)}{_fmt(result['maker_total'], 12, 3)}"
    )
    print(
        f"  {'taker 0.30%, all fill':<28}{result['n_signals']:>9}"
        f"{_fmt(result['taker_mean'], 11, 3, 100)}{_fmt(result['taker_total'], 12, 3)}"
    )

    if result["maker_mean"] is not None and result["taker_mean"] is not None:
        d_mean = (result["maker_mean"] - result["taker_mean"]) * 100
        d_total = result["maker_total"] - result["taker_total"]
        better = "maker" if d_total > 0 else "taker"
        print(
            f"\n  Better on total net: {better.upper()} "
            f"(maker - taker = {d_total:+.3f} log, {d_mean:+.3f}pp per trade)"
        )

    maker_folds = 0
    folds = sorted(marked["fold"].unique().to_list())
    print(f"\n  per-fold, maker route (0.15%, filled only) at threshold {threshold:.2f}:")
    for fold in folds:
        at = marked.filter(
            (pl.col("fold") == fold)
            & (pl.col("y_pred_proba") >= threshold)
            & pl.col("filled")
        )["ret"].to_numpy()
        value = mean_net(at, MAKER_COST)
        if value is not None and value > 0:
            maker_folds += 1
        mark = "*" if value is not None and value > 0 else " "
        print(f"    fold {fold}: n={len(at):>5}  mean_net {_fmt(value, 8, 3, 100)}%{mark}")
    print(f"  folds positive (maker): {maker_folds} of {len(folds)}")


def print_symbol_verdict(stats: SymbolCosts) -> None:
    a = stats.pooled_mean_net_taker is not None and stats.pooled_mean_net_taker > 0
    b = stats.folds_positive_taker >= 4
    c = stats.break_even >= MAKER_COST

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    verdict = "GO" if (a and b and c) else "NO-GO"
    print(
        f"  {stats.symbol:<9} break-even {stats.break_even * 100:>5.3f}%  "
        f"(a) {mark(a)}  (b) {mark(b)} [{stats.folds_positive_taker}/{stats.n_folds}]  "
        f"(c) {mark(c)}  -> {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--verdict-symbol", default="BTCUSDT")
    args = parser.parse_args()

    print(SUCCESS_CRITERIA)
    print(
        f"\nNo retraining: CV predictions are read back and only the cost "
        f"assumption varies.\nCost enters as log(1-c), so threshold ranking is "
        f"identical at every cost level."
    )

    all_stats: list[SymbolCosts] = []
    marked_by_symbol: dict[str, pl.DataFrame] = {}

    print(f"\n{'=' * 78}")
    print("PART 1 - COST SWEEP")
    print("=" * 78)

    for symbol in args.symbols:
        preds = pl.read_parquet(
            args.parquet_dir / f"cv_predictions_{symbol}.parquet"
        ).sort("open_time")
        stats = analyse_costs(symbol, preds)
        all_stats.append(stats)

        print_cost_grid(symbol, preds)
        if stats.best_threshold is not None:
            print_fold_breakdown(preds, stats.best_threshold)
            print(
                f"\n  best threshold {stats.best_threshold:.2f} "
                f"({stats.n_best_signals} signals), mean gross "
                f"{stats.best_mean_gross * 100:+.3f}%"
            )
            print(
                f"  BREAK-EVEN ROUND TRIP: {stats.break_even * 100:.3f}%"
                + (
                    "  (never profitable, even at zero cost)"
                    if stats.break_even <= 0
                    else ""
                )
            )

        labeled = pl.read_parquet(args.parquet_dir / f"labeled_{symbol}.parquet")
        marked_by_symbol[symbol] = attach_fill_outcome(preds, labeled)

    print(f"\n{'=' * 78}")
    print("PART 2 - FILL REALISM (does the maker route get the good trades?)")
    print("=" * 78)

    for symbol in args.symbols:
        rows = print_fill_analysis(symbol, marked_by_symbol[symbol])

        # Only claim adverse selection when enough trades actually missed.
        gaps = [
            r.adverse_gap
            for r in rows
            if r.adverse_gap is not None and r.n_missed >= MIN_MISSED_FOR_CLAIM
        ]
        if gaps and sum(g > 0 for g in gaps) > len(gaps) / 2:
            print(
                f"    ADVERSE SELECTION: missed trades out-return filled ones at "
                f"{sum(g > 0 for g in gaps)}/{len(gaps)} thresholds "
                f"(up to {max(gaps) * 100:+.3f}pp). The limit order is "
                f"systematically filled by the worse half."
            )

        stats = next(s for s in all_stats if s.symbol == symbol)
        if stats.best_threshold is not None:
            print_offset_sweep(symbol, marked_by_symbol[symbol], stats.best_threshold)

    verdict_symbol = args.verdict_symbol
    if verdict_symbol in marked_by_symbol:
        stats = next(s for s in all_stats if s.symbol == verdict_symbol)
        if stats.best_threshold is not None:
            print_combined_verdict(
                verdict_symbol, marked_by_symbol[verdict_symbol], stats.best_threshold
            )

    print(f"\n{'=' * 78}")
    print("VERDICT vs SUCCESS CRITERIA")
    print("=" * 78)
    print(
        f"  (a) pooled mean_net > 0 at {TAKER_COST:.2%}   "
        f"(b) >= 4/5 folds positive   (c) break-even >= {MAKER_COST:.2%}"
    )
    for stats in all_stats:
        print_symbol_verdict(stats)


if __name__ == "__main__":
    main()
