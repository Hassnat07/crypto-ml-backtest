"""Tests for src/train.py.

Training runs use a reduced n_estimators so the suite stays fast; the
permutation check is the one place where the result itself is asserted, and
it is stable under that reduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import features  # noqa: E402
import splits  # noqa: E402
import train  # noqa: E402

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
SYMBOL = "BTCUSDT"
SEED = 42

FAST_PARAMS: dict[str, object] = {
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.read_parquet(PARQUET_DIR / f"labeled_{SYMBOL}.parquet").sort("open_time")


@pytest.fixture(scope="module")
def holdout_and_folds(df: pl.DataFrame) -> tuple[list[int], list[splits.Fold]]:
    dev_idx, holdout_idx = splits.final_holdout_split(
        df, splits.DEFAULT_HOLDOUT_MONTHS, splits.DEFAULT_EMBARGO_BARS
    )
    dev_df = df[: max(dev_idx) + 1]
    folds = splits.build_folds(
        dev_df, splits.DEFAULT_N_SPLITS, splits.DEFAULT_EMBARGO_BARS
    )
    return holdout_idx, folds


@pytest.fixture(scope="module")
def trained(
    df: pl.DataFrame, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[train.FoldResult], pl.DataFrame, Path]:
    model_dir = tmp_path_factory.mktemp("models")
    results, predictions = train.run_symbol(
        SYMBOL,
        df,
        FAST_PARAMS,
        splits.DEFAULT_N_SPLITS,
        splits.DEFAULT_EMBARGO_BARS,
        splits.DEFAULT_HOLDOUT_MONTHS,
        SEED,
        drop_null_rows=False,
        model_dir=model_dir,
    )
    return results, predictions, model_dir


# ---------------------------------------------------------------------------
# 1. No holdout row appears in any CV fold
# ---------------------------------------------------------------------------


def test_no_holdout_row_in_any_fold(
    holdout_and_folds: tuple[list[int], list[splits.Fold]]
) -> None:
    holdout_idx, folds = holdout_and_folds
    holdout = set(holdout_idx)
    assert holdout

    for fold in folds:
        assert not holdout & set(fold.train_idx)
        assert not holdout & set(fold.test_idx)

    train.assert_folds_avoid_holdout(SYMBOL, folds, holdout_idx)


def test_holdout_guard_raises_on_contaminated_folds(df: pl.DataFrame) -> None:
    """Proves the guard is not vacuous.

    Building folds on the FULL frame -- the exact mistake the guard exists to
    catch -- puts holdout rows in the final fold's test block.
    """
    _, holdout_idx = splits.final_holdout_split(
        df, splits.DEFAULT_HOLDOUT_MONTHS, splits.DEFAULT_EMBARGO_BARS
    )
    contaminated = splits.build_folds(
        df, splits.DEFAULT_N_SPLITS, splits.DEFAULT_EMBARGO_BARS
    )
    with pytest.raises(train.HoldoutContaminationError, match="holdout rows"):
        train.assert_folds_avoid_holdout(SYMBOL, contaminated, holdout_idx)


def test_holdout_rows_absent_from_predictions(
    df: pl.DataFrame, trained: tuple[list[train.FoldResult], pl.DataFrame, Path]
) -> None:
    _, predictions, _ = trained
    _, holdout_idx = splits.final_holdout_split(
        df, splits.DEFAULT_HOLDOUT_MONTHS, splits.DEFAULT_EMBARGO_BARS
    )
    holdout_times = set(df[holdout_idx]["open_time"].to_list())
    predicted_times = set(predictions["open_time"].to_list())
    assert not holdout_times & predicted_times


# ---------------------------------------------------------------------------
# 2. Predicted probabilities are in [0, 1]
# ---------------------------------------------------------------------------


def test_probabilities_in_unit_interval(
    trained: tuple[list[train.FoldResult], pl.DataFrame, Path]
) -> None:
    results, predictions, _ = trained
    for result in results:
        assert np.isfinite(result.proba).all()
        assert result.proba.min() >= 0.0
        assert result.proba.max() <= 1.0

    proba = predictions["y_pred_proba"]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert proba.null_count() == 0


# ---------------------------------------------------------------------------
# 3. Prediction count equals test set size for every fold
# ---------------------------------------------------------------------------


def test_prediction_count_matches_test_size(
    trained: tuple[list[train.FoldResult], pl.DataFrame, Path],
    holdout_and_folds: tuple[list[int], list[splits.Fold]],
) -> None:
    results, predictions, _ = trained
    _, folds = holdout_and_folds
    assert len(results) == len(folds)

    for result, fold in zip(results, folds):
        assert len(result.proba) == fold.n_test
        assert len(result.y_true) == fold.n_test
        assert len(result.ret) == fold.n_test
        assert len(result.open_time) == fold.n_test

    assert predictions.height == sum(f.n_test for f in folds)


def test_null_rows_are_not_dropped_from_test(
    df: pl.DataFrame, holdout_and_folds: tuple[list[int], list[splits.Fold]]
) -> None:
    """Dropping nulls affects training only; test coverage must stay whole."""
    _, folds = holdout_and_folds
    cols = train.usable_feature_columns(df)
    fold = folds[0]

    kept, _ = train.fit_fold(df, fold, cols, FAST_PARAMS, SEED, drop_null_rows=False)
    dropped, _ = train.fit_fold(df, fold, cols, FAST_PARAMS, SEED, drop_null_rows=True)

    assert len(kept.proba) == len(dropped.proba) == fold.n_test
    assert dropped.n_train < kept.n_train
    assert kept.n_train_dropped > 0


# ---------------------------------------------------------------------------
# 4. Permutation check lands near chance
# ---------------------------------------------------------------------------


def test_permutation_auc_near_chance(
    df: pl.DataFrame, holdout_and_folds: tuple[list[int], list[splits.Fold]]
) -> None:
    _, folds = holdout_and_folds
    cols = train.usable_feature_columns(df)
    aucs = train.permutation_check(df, folds[-1], cols, FAST_PARAMS, SEED, repeats=3)
    assert aucs

    auc = float(np.mean(aucs))
    assert 0.42 <= auc <= 0.58, (
        f"shuffled-label AUC {auc:.4f} (from {aucs}) is not near chance -- the "
        "model is reading something other than the feature-label relationship"
    )


# ---------------------------------------------------------------------------
# 5. Feature matrix width matches the feature list
# ---------------------------------------------------------------------------


def test_matrix_width_matches_feature_list(
    df: pl.DataFrame, holdout_and_folds: tuple[list[int], list[splits.Fold]]
) -> None:
    _, folds = holdout_and_folds
    cols = train.usable_feature_columns(df)
    X, y, ret = train.build_matrix(df, folds[0].train_idx, cols)

    assert X.shape[1] == len(cols)
    assert X.shape[0] == len(y) == len(ret) == folds[0].n_train


def test_excluded_columns_are_absent(df: pl.DataFrame) -> None:
    cols = train.usable_feature_columns(df)
    assert "after_gap" not in cols
    # corr_with_btc_90 is all-null for BTC, so it must drop out for this symbol.
    assert "corr_with_btc_90" not in cols
    assert set(cols) < set(features.FEATURE_COLUMNS)
    for col in cols:
        assert df[col].null_count() < df.height


def test_non_btc_keeps_btc_correlation() -> None:
    """The all-null drop is per symbol, not a blanket removal."""
    eth = pl.read_parquet(PARQUET_DIR / "labeled_ETHUSDT.parquet").sort("open_time")
    assert "corr_with_btc_90" in train.usable_feature_columns(eth)


# ---------------------------------------------------------------------------
# 6. Model files are written and reloadable
# ---------------------------------------------------------------------------


def test_models_written_and_reloadable(
    df: pl.DataFrame,
    trained: tuple[list[train.FoldResult], pl.DataFrame, Path],
    holdout_and_folds: tuple[list[int], list[splits.Fold]],
) -> None:
    results, _, model_dir = trained
    _, folds = holdout_and_folds
    cols = train.usable_feature_columns(df)

    for result, fold in zip(results, folds):
        path = model_dir / f"{SYMBOL}_fold{fold.index}.txt"
        assert path.exists() and path.stat().st_size > 0

        booster = lgb.Booster(model_file=str(path))
        assert booster.num_feature() == len(cols)

        X_test, _, _ = train.build_matrix(df, fold.test_idx, cols)
        reloaded = booster.predict(X_test)
        np.testing.assert_allclose(reloaded, result.proba, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Threshold table sanity
# ---------------------------------------------------------------------------


def test_threshold_rows_are_monotonic_in_signal_count(
    trained: tuple[list[train.FoldResult], pl.DataFrame, Path]
) -> None:
    results, _, _ = trained
    for result in results:
        rows = train.threshold_rows(result.y_true, result.proba, result.ret)
        counts = [r["n_signals"] for r in rows]
        assert counts == sorted(counts, reverse=True)
        assert all(0.0 <= r["coverage"] <= 1.0 for r in rows)
        for row in rows:
            if row["precision"] is not None:
                assert 0.0 <= row["precision"] <= 1.0
