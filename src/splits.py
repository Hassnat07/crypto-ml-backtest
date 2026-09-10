"""Purged walk-forward cross-validation splits.

Triple-barrier labels overlap: the label at bar i is only resolved at t1,
up to H bars later. A naive contiguous split therefore leaks -- a training
row near the boundary carries an outcome that was determined inside the
test window. Random k-fold on this data leaks far worse still.

Two defences, both applied to the training side (the test block is never
touched, so test-set statistics stay honest):

  PURGE    drop training rows whose label window reaches into the test
           window, i.e. t1 >= test_start.
  EMBARGO  drop training rows in the `embargo_bars` immediately before
           test_start. Their labels resolve before the test window, but
           serial correlation in price and features still makes them
           near-duplicates of the earliest test rows.

Splits are expanding-window: fold k trains on everything before its test
block, so fold 1 sees the least data and fold n the most.

Only rows carrying a resolved label are eligible as samples. Unlabelled
rows (warm-up bars, the trailing horizon, ambiguous bars) are not dropped
from the frame -- they are simply never emitted as train or test indices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_N_SPLITS = 5
DEFAULT_EMBARGO_BARS = 30
DEFAULT_HOLDOUT_MONTHS = 12


class LeakageError(AssertionError):
    """Raised when a split would let test-period information into training."""


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, with the bookkeeping needed to audit it."""

    index: int
    train_idx: list[int]
    test_idx: list[int]
    n_purged: int
    n_embargoed: int

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def eligible_indices(df: pl.DataFrame) -> list[int]:
    """Row indices of samples usable for supervised learning.

    A resolved label implies a resolved t1 (verified in the labelling
    stage), which is what makes the purge comparison well defined.
    """
    return (
        df.with_row_index("_idx")
        .filter(pl.col("label").is_not_null())
        .get_column("_idx")
        .cast(pl.Int64)
        .to_list()
    )


def _chunk_bounds(n: int, n_splits: int) -> list[int]:
    """Boundaries splitting n samples into n_splits + 1 contiguous chunks.

    The extra chunk is the seed training block: fold 1 trains on chunk 0 and
    tests on chunk 1, so every fold has a non-empty training set.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    n_chunks = n_splits + 1
    if n < n_chunks:
        raise ValueError(f"need at least {n_chunks} labelled rows, got {n}")
    return [j * n // n_chunks for j in range(n_chunks)] + [n]


def verify_no_leakage(
    df: pl.DataFrame, train_idx: list[int], test_idx: list[int]
) -> None:
    """Raise LeakageError if this split leaks test information into training.

    Checks, in order: index overlap, temporal ordering, and label-window
    containment (the purge condition). Passing all three means no training
    row's outcome could have been influenced by the test window.
    """
    if not test_idx:
        raise ValueError("test split is empty")
    if not train_idx:
        return

    overlap = set(train_idx) & set(test_idx)
    if overlap:
        raise LeakageError(
            f"train and test share {len(overlap)} rows "
            f"(e.g. index {sorted(overlap)[0]})"
        )

    max_train, min_test = max(train_idx), min(test_idx)
    if max_train >= min_test:
        raise LeakageError(
            f"train index {max_train} is not before test index {min_test}; "
            "walk-forward requires all training rows to precede the test block"
        )

    test_start: datetime = df["open_time"][min_test]
    train_t1 = df[train_idx]["t1"]

    n_null = train_t1.null_count()
    if n_null:
        raise LeakageError(
            f"{n_null} training rows have a null t1; their forward window is "
            "unknown so purging cannot be verified"
        )

    n_late = int((train_t1 >= test_start).sum())
    if n_late:
        latest = train_t1.max()
        raise LeakageError(
            f"{n_late} training rows have t1 >= test start {test_start} "
            f"(latest t1 {latest}); these labels resolve inside the test window"
        )


def _build_fold(
    df: pl.DataFrame,
    eligible: list[int],
    lo: int,
    hi: int,
    fold_index: int,
    embargo_bars: int,
    purge: bool,
) -> Fold:
    """Assemble one fold from eligible-sample positions [lo, hi) as test."""
    test_idx = eligible[lo:hi]
    candidate_train = eligible[:lo]

    if not purge or not candidate_train:
        return Fold(fold_index, candidate_train, test_idx, 0, 0)

    test_start_row = test_idx[0]
    test_start_time: datetime = df["open_time"][test_start_row]
    embargo_floor = test_start_row - embargo_bars

    t1 = df[candidate_train]["t1"].to_list()

    train_idx: list[int] = []
    n_purged = 0
    n_embargoed = 0
    for row, row_t1 in zip(candidate_train, t1):
        # Purge takes precedence so the two counts stay additive.
        if row_t1 is None or row_t1 >= test_start_time:
            n_purged += 1
        elif row >= embargo_floor:
            n_embargoed += 1
        else:
            train_idx.append(row)

    return Fold(fold_index, train_idx, test_idx, n_purged, n_embargoed)


def build_folds(
    df: pl.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
) -> list[Fold]:
    """Purged, embargoed, expanding-window folds with audit counts."""
    eligible = eligible_indices(df)
    bounds = _chunk_bounds(len(eligible), n_splits)

    folds = [
        _build_fold(
            df, eligible, bounds[k], bounds[k + 1], k, embargo_bars, purge=True
        )
        for k in range(1, n_splits + 1)
    ]
    for fold in folds:
        verify_no_leakage(df, fold.train_idx, fold.test_idx)
    return folds


def purged_walk_forward(
    df: pl.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
) -> list[tuple[list[int], list[int]]]:
    """(train_indices, test_indices) per fold, verified leak-free.

    Every fold is passed through verify_no_leakage before returning, so a
    caller cannot receive a leaking split from this function.
    """
    return [(f.train_idx, f.test_idx) for f in build_folds(df, n_splits, embargo_bars)]


def naive_walk_forward(
    df: pl.DataFrame, n_splits: int = DEFAULT_N_SPLITS
) -> list[tuple[list[int], list[int]]]:
    """Contiguous walk-forward with NO purge and NO embargo.

    The leaking baseline this module exists to replace. Provided so the
    leakage check can be shown to fail on it -- proving the check has teeth
    -- and so the cost of purging can be quantified. Do not train on these.
    """
    eligible = eligible_indices(df)
    bounds = _chunk_bounds(len(eligible), n_splits)
    return [
        (eligible[: bounds[k]], eligible[bounds[k] : bounds[k + 1]])
        for k in range(1, n_splits + 1)
    ]


def final_holdout_split(
    df: pl.DataFrame,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    embargo_bars: int = DEFAULT_EMBARGO_BARS,
) -> tuple[list[int], list[int]]:
    """Split off the final `holdout_months` as an untouched holdout.

    Look at this ONCE, at the very end, for the go/no-go call. The dev side
    is purged and embargoed against the holdout boundary on the same terms
    as a CV fold -- without that, dev labels spanning the boundary would
    contaminate the one set that is supposed to be clean.
    """
    eligible = eligible_indices(df)
    if not eligible:
        return [], []

    last: datetime = df["open_time"][eligible[-1]]
    cutoff: datetime = (
        pl.Series([last]).dt.offset_by(f"-{holdout_months}mo").item()
    )

    holdout = [i for i in eligible if df["open_time"][i] >= cutoff]
    if not holdout:
        return eligible, []

    dev_candidates = [i for i in eligible if df["open_time"][i] < cutoff]
    fold = _build_fold(
        df,
        dev_candidates + holdout,
        len(dev_candidates),
        len(dev_candidates) + len(holdout),
        0,
        embargo_bars,
        purge=True,
    )
    verify_no_leakage(df, fold.train_idx, fold.test_idx)
    return fold.train_idx, fold.test_idx


def _label_balance(df: pl.DataFrame, idx: list[int]) -> str:
    if not idx:
        return "n/a"
    labels = df[idx]["label"]
    wins = int((labels == 1).sum())
    return f"{wins / len(idx):.1%} win ({wins}/{len(idx)})"


def _span(df: pl.DataFrame, idx: list[int]) -> str:
    if not idx:
        return "n/a"
    times = df[idx]["open_time"]
    return f"{times.min():%Y-%m-%d} -> {times.max():%Y-%m-%d}"


def print_summary(
    symbol: str,
    df: pl.DataFrame,
    folds: list[Fold],
    n_splits: int,
    embargo_bars: int,
) -> None:
    eligible = eligible_indices(df)
    print(f"\n=== {symbol} ===")
    print(
        f"rows {df.height}, labelled {len(eligible)}, "
        f"n_splits {n_splits}, embargo {embargo_bars} bars"
    )

    for fold in folds:
        dropped = fold.n_purged + fold.n_embargoed
        print(
            f"\n  fold {fold.index}: train {fold.n_train:>6}  test {fold.n_test:>6}"
            f"  purged {fold.n_purged:>3}  embargoed {fold.n_embargoed:>3}"
            f"  (dropped {dropped})"
        )
        print(f"    train  {_span(df, fold.train_idx):<26} {_label_balance(df, fold.train_idx)}")
        print(f"    test   {_span(df, fold.test_idx):<26} {_label_balance(df, fold.test_idx)}")

    dev_idx, holdout_idx = final_holdout_split(df, DEFAULT_HOLDOUT_MONTHS, embargo_bars)
    print(f"\n  final holdout ({DEFAULT_HOLDOUT_MONTHS} months, look once):")
    print(f"    dev      {_span(df, dev_idx):<26} {_label_balance(df, dev_idx)}")
    print(f"    holdout  {_span(df, holdout_idx):<26} {_label_balance(df, holdout_idx)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--embargo-bars", type=int, default=DEFAULT_EMBARGO_BARS)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    args = parser.parse_args()

    for symbol in args.symbols:
        df = pl.read_parquet(args.parquet_dir / f"labeled_{symbol}.parquet").sort(
            "open_time"
        )
        folds = build_folds(df, args.n_splits, args.embargo_bars)
        print_summary(symbol, df, folds, args.n_splits, args.embargo_bars)


if __name__ == "__main__":
    main()
