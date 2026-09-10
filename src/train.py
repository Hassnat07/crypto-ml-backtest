"""LightGBM classifier trained under purged walk-forward CV.

The holdout is never read here. final_holdout_split() runs first, CV is
confined to the dev slice, and every fold index is asserted to fall outside
the holdout before any model is fitted.

What matters in the output is not AUC but the threshold table: a classifier
that is barely better than chance overall can still be useful if its
high-confidence bucket is genuinely more precise than the base rate. The
lift column is where that shows up, and mean net return is where it either
survives the 0.30% round trip or does not.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import baseline
import features
import splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
DEFAULT_SEED = 42
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# after_gap is a data-quality flag, not a signal; it marks the Binance
# outage bar and would let the model key on a single calendar date.
EXCLUDED_FEATURES = {"after_gap"}


class HoldoutContaminationError(AssertionError):
    """Raised if a CV fold reaches into the final holdout."""


@dataclass
class FoldResult:
    fold: int
    open_time: pl.Series
    y_true: np.ndarray
    proba: np.ndarray
    ret: np.ndarray
    auc: float | None
    n_train: int
    n_train_dropped: int
    importance: dict[str, float]


def usable_feature_columns(df: pl.DataFrame) -> list[str]:
    """FEATURE_COLUMNS minus the excluded flag and any all-null column.

    corr_with_btc_90 is all-null for BTCUSDT by construction, so the matrix
    is narrower for that symbol than for the others.
    """
    cols = []
    for col in features.FEATURE_COLUMNS:
        if col in EXCLUDED_FEATURES:
            continue
        if df[col].null_count() == df.height:
            continue
        cols.append(col)
    return cols


def build_matrix(
    df: pl.DataFrame, idx: list[int], cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y, ret for the given rows. Nulls survive as NaN for LightGBM."""
    rows = df[idx]
    X = rows.select(cols).to_numpy().astype(np.float64)
    y = rows["label"].to_numpy().astype(np.int8)
    ret = rows["ret"].to_numpy().astype(np.float64)
    return X, y, ret


def _complete_rows(X: np.ndarray) -> np.ndarray:
    return ~np.isnan(X).any(axis=1)


def make_model(params: dict[str, object], seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
        **params,
    )


def fit_fold(
    df: pl.DataFrame,
    fold: splits.Fold,
    cols: list[str],
    params: dict[str, object],
    seed: int,
    drop_null_rows: bool,
) -> tuple[FoldResult, lgb.LGBMClassifier]:
    X_train, y_train, _ = build_matrix(df, fold.train_idx, cols)
    X_test, y_test, ret_test = build_matrix(df, fold.test_idx, cols)

    incomplete = int((~_complete_rows(X_train)).sum())
    if drop_null_rows:
        keep = _complete_rows(X_train)
        X_train, y_train = X_train[keep], y_train[keep]

    model = make_model(params, seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = (
        float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None
    )

    gains = model.booster_.feature_importance(importance_type="gain")
    importance = {c: float(g) for c, g in zip(cols, gains)}

    result = FoldResult(
        fold=fold.index,
        open_time=df[fold.test_idx]["open_time"],
        y_true=y_test,
        proba=proba,
        ret=ret_test,
        auc=auc,
        n_train=len(y_train),
        n_train_dropped=incomplete,
        importance=importance,
    )
    return result, model


def threshold_rows(
    y_true: np.ndarray, proba: np.ndarray, ret: np.ndarray
) -> list[dict[str, object]]:
    """Signal count, precision and net P&L at each confidence threshold."""
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    rows = []
    for thr in THRESHOLDS:
        mask = proba >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "threshold": thr,
                    "n_signals": 0,
                    "coverage": 0.0,
                    "precision": None,
                    "lift": None,
                    "mean_net": None,
                    "total_net": 0.0,
                }
            )
            continue
        precision = float(y_true[mask].mean())
        net = ret[mask] + baseline.COST_LOG
        rows.append(
            {
                "threshold": thr,
                "n_signals": n,
                "coverage": n / len(y_true),
                "precision": precision,
                "lift": precision - base_rate,
                "mean_net": float(net.mean()),
                "total_net": float(net.sum()),
            }
        )
    return rows


def _fmt(value: object, width: int, precision: int, scale: float = 1.0) -> str:
    if value is None:
        return "-".rjust(width)
    return format(value * scale, f">{width}.{precision}f")


def print_threshold_table(
    title: str, y_true: np.ndarray, proba: np.ndarray, ret: np.ndarray, auc: float | None
) -> None:
    base_rate = float(y_true.mean())
    always_net = float((ret + baseline.COST_LOG).mean())

    print(f"\n{title}")
    print(
        f"  n_test {len(y_true)}   base rate {base_rate:.1%}   "
        f"AUC {'-' if auc is None else f'{auc:.4f}'}   "
        f"always_trade net {always_net * 100:+.3f}%"
    )
    header = (
        f"  {'thr':>5}{'signals':>9}{'cover':>8}{'prec':>8}"
        f"{'lift':>8}{'mean_net':>10}{'total_net':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in threshold_rows(y_true, proba, ret):
        print(
            f"  {row['threshold']:>5.2f}"
            f"{row['n_signals']:>9}"
            f"{_fmt(row['coverage'], 8, 1, 100)}"
            f"{_fmt(row['precision'], 8, 1, 100)}"
            f"{_fmt(row['lift'], 8, 1, 100)}"
            f"{_fmt(row['mean_net'], 10, 3, 100)}"
            f"{_fmt(row['total_net'], 11, 2)}"
        )


def permutation_check(
    df: pl.DataFrame,
    fold: splits.Fold,
    cols: list[str],
    params: dict[str, object],
    seed: int,
    repeats: int = 1,
) -> list[float]:
    """Retrain one fold on shuffled labels; AUC should collapse to ~0.50.

    Shuffling breaks the feature-label link while leaving the feature matrix,
    the split geometry and the class balance untouched. An AUC that stays
    high afterwards means the model is reading something structural -- row
    order, a leaked column, an artefact of the split -- rather than signal.

    A single shuffle is a noisy estimate (empirically +-0.04 here), so
    `repeats` draws several and the caller reports the spread. One draw
    landing at 0.56 is not evidence of leakage on its own.
    """
    X_train, y_train, _ = build_matrix(df, fold.train_idx, cols)
    X_test, y_test, _ = build_matrix(df, fold.test_idx, cols)
    if len(np.unique(y_test)) < 2:
        return []

    aucs = []
    for r in range(repeats):
        shuffled = y_train.copy()
        np.random.default_rng(seed + r).shuffle(shuffled)
        model = make_model(params, seed)
        model.fit(X_train, shuffled)
        aucs.append(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])))
    return aucs


def assert_folds_avoid_holdout(
    symbol: str, folds: list[splits.Fold], holdout_idx: list[int]
) -> None:
    """Raise unless every fold stays strictly inside the dev slice.

    Cheap to run and the single thing standing between an honest final
    evaluation and a quietly burned holdout, so it runs before any fit.
    """
    holdout = set(holdout_idx)
    for fold in folds:
        touched = holdout & (set(fold.train_idx) | set(fold.test_idx))
        if touched:
            raise HoldoutContaminationError(
                f"{symbol} fold {fold.index} contains {len(touched)} holdout rows "
                f"(e.g. index {sorted(touched)[0]}); CV must stay inside the dev slice"
            )


def run_symbol(
    symbol: str,
    df: pl.DataFrame,
    params: dict[str, object],
    n_splits: int,
    embargo_bars: int,
    holdout_months: int,
    seed: int,
    drop_null_rows: bool,
    model_dir: Path,
    permutation_repeats: int = 1,
) -> tuple[list[FoldResult], pl.DataFrame]:
    # Holdout comes off first and is never read again in this function.
    dev_idx, holdout_idx = splits.final_holdout_split(df, holdout_months, embargo_bars)
    if not dev_idx:
        raise ValueError(f"{symbol}: empty dev slice")

    dev_df = df[: max(dev_idx) + 1]
    folds = splits.build_folds(dev_df, n_splits, embargo_bars)
    assert_folds_avoid_holdout(symbol, folds, holdout_idx)

    cols = usable_feature_columns(dev_df)
    dropped_cols = [c for c in features.FEATURE_COLUMNS if c not in cols]

    print(f"\n{'=' * 78}")
    print(f"=== {symbol} ===")
    print(
        f"rows {df.height}, dev {len(dev_idx)}, holdout {len(holdout_idx)} "
        f"(untouched), folds {len(folds)}"
    )
    print(f"features {len(cols)} of {len(features.FEATURE_COLUMNS)}; dropped: {dropped_cols}")
    print(f"null handling: {'drop rows with any null' if drop_null_rows else 'kept (LightGBM native)'}")

    model_dir.mkdir(parents=True, exist_ok=True)
    results: list[FoldResult] = []
    for fold in folds:
        result, model = fit_fold(df, fold, cols, params, seed, drop_null_rows)
        results.append(result)
        model.booster_.save_model(str(model_dir / f"{symbol}_fold{fold.index}.txt"))

        print_threshold_table(
            f"--- fold {fold.index}  (train {result.n_train}, "
            f"{result.n_train_dropped} train rows carry a null feature) ---",
            result.y_true,
            result.proba,
            result.ret,
            result.auc,
        )

    y_all = np.concatenate([r.y_true for r in results])
    p_all = np.concatenate([r.proba for r in results])
    ret_all = np.concatenate([r.ret for r in results])
    pooled_auc = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) > 1 else None
    print_threshold_table(f"--- {symbol} POOLED across folds ---", y_all, p_all, ret_all, pooled_auc)

    aucs = [r.auc for r in results if r.auc is not None]
    print(f"\n  AUC per fold: {', '.join(f'{a:.4f}' for a in aucs)}")
    print(f"  AUC mean {np.mean(aucs):.4f}  pooled {pooled_auc:.4f}")

    print("\n  Top 15 features by gain (mean across folds):")
    mean_gain = {
        c: float(np.mean([r.importance.get(c, 0.0) for r in results])) for c in cols
    }
    total_gain = sum(mean_gain.values()) or 1.0
    for rank, (col, gain) in enumerate(
        sorted(mean_gain.items(), key=lambda kv: kv[1], reverse=True)[:15], start=1
    ):
        print(f"    {rank:>2}. {col:<24}{gain:>12.1f}  {gain / total_gain:>6.1%}")

    perm_aucs = permutation_check(df, folds[-1], cols, params, seed, permutation_repeats)
    if perm_aucs:
        perm_mean = float(np.mean(perm_aucs))
        verdict = (
            "OK, near chance"
            if 0.42 <= perm_mean <= 0.58
            else "SUSPECT - investigate leakage"
        )
        spread = (
            ""
            if len(perm_aucs) == 1
            else f" over {len(perm_aucs)} shuffles [{min(perm_aucs):.4f}, {max(perm_aucs):.4f}]"
        )
        print(
            f"\n  Permutation check (fold {folds[-1].index}, shuffled y): "
            f"AUC {perm_mean:.4f}{spread}  [{verdict}]"
        )
        if pooled_auc is not None and perm_mean >= pooled_auc:
            print(
                "    NOTE: shuffled-label AUC is >= the real pooled AUC, i.e. this "
                "symbol carries no usable signal at all."
            )

    predictions = pl.concat(
        [
            pl.DataFrame(
                {
                    "open_time": r.open_time,
                    "fold": pl.Series([r.fold] * len(r.y_true), dtype=pl.Int32),
                    "y_true": pl.Series(r.y_true, dtype=pl.Int8),
                    "y_pred_proba": pl.Series(r.proba, dtype=pl.Float64),
                    "ret": pl.Series(r.ret, dtype=pl.Float64),
                }
            )
            for r in results
        ]
    )
    return results, predictions


def build_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        # Without a non-zero subsample_freq LightGBM ignores subsample entirely.
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--n-splits", type=int, default=splits.DEFAULT_N_SPLITS)
    parser.add_argument("--embargo-bars", type=int, default=splits.DEFAULT_EMBARGO_BARS)
    parser.add_argument("--holdout-months", type=int, default=splits.DEFAULT_HOLDOUT_MONTHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=1,
        help="Shuffled-label refits for the leakage check; >1 averages out its noise",
    )
    parser.add_argument(
        "--drop-null-rows",
        action="store_true",
        help="Drop training rows with any null feature (default: keep, LightGBM handles them)",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=0.1)
    args = parser.parse_args()

    params = build_params(args)

    for symbol in args.symbols:
        df = pl.read_parquet(args.parquet_dir / f"labeled_{symbol}.parquet").sort(
            "open_time"
        )
        _, predictions = run_symbol(
            symbol,
            df,
            params,
            args.n_splits,
            args.embargo_bars,
            args.holdout_months,
            args.seed,
            args.drop_null_rows,
            args.model_dir,
            args.permutation_repeats,
        )
        out_path = args.parquet_dir / f"cv_predictions_{symbol}.parquet"
        predictions.write_parquet(out_path)
        print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
