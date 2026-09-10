# btc-signals

An end-to-end machine learning pipeline testing whether gradient-boosted trees can
find tradeable edge in 4-hour cryptocurrency bars after realistic transaction costs.

**Result: no.** The model finds real, statistically detectable signal — BTC precision
lifts 8.3 percentage points above base rate — but the edge is smaller than the 0.30%
round-trip cost of trading it. A pre-registered go/no-go test returned NO-GO on all
four symbols.

This repository is built around the checks that make that conclusion trustworthy.
216 tests, several of them verified non-vacuous by deliberate bug injection.

---

## Why a negative result

Most retail ML-trading projects report a profitable backtest. Almost all of them are
wrong, for the same handful of reasons: lookahead bias in features, label leakage
across train/test boundaries, transaction costs added as an afterthought, and
strategy selection over dozens of untracked variants.

This project was built to make those failures impossible to hide:

- Every feature is verified point-in-time by recomputing on truncated data
- Cross-validation is purged and embargoed against overlapping label windows
- Transaction costs are modelled from the first line of the backtest, not the last
- A permutation test confirms the model reads signal, not structure
- 12 months of data were held out and never examined
- Success criteria were written down before the final test was run

When the answer came back negative, the criteria were not revised.

---

## Results

### Baseline: the cost wall

Before any model, six rule-based strategies were evaluated on 6.5 years of data.

| Strategy | Trades | Gross/trade | Net/trade |
|---|---|---|---|
| `always_trade` | 14,368 | +0.251% | **−0.049%** |
| `random_5pct` | 718 | +0.254% | **−0.046%** |
| `momentum` | 4,851 | +0.303% | +0.002% |
| `rsi_oversold` | 645 | −0.407% | −0.708% |
| `mean_reversion` | 984 | −0.330% | −0.631% |

BTCUSDT, costs at 0.30% round trip (0.1% fee + 0.05% slippage per side).

The raw signal available to any strategy is roughly 0.25% per trade. The cost of
capturing it is 0.30%. Randomly selected trades perform indistinguishably from
"smart" rules, which is the signature of no exploitable structure at this horizon.

### Model: real lift, insufficient magnitude

LightGBM classifier, 29 engineered features, triple-barrier labels, purged
walk-forward CV over 5 expanding folds.

| Symbol | Pooled AUC | Best precision lift | Best net/trade |
|---|---|---|---|
| BTCUSDT | 0.554 | +8.3pp | −0.021% |
| ETHUSDT | 0.518 | +4.2pp | −0.989% |
| SOLUSDT | 0.577 | +13.9pp | +2.522% |
| BNBUSDT | 0.492 | +1.7pp | −0.781% |

BTC precision rises from a 32.5% base rate to 40.8% at confidence threshold 0.65.
The lift is real and survives permutation testing. It is also about one tenth of
what is needed to clear costs.

SOLUSDT's apparent success is not treated as evidence. The same symbol was the
outlier in the rule-based baseline, its result concentrates in the 2021 trend
regime, and at threshold 0.80 two of five folds fire fewer than 10 times. Two
methods agreeing on one asset during one directional run is a single observation.

BNBUSDT's real AUC (0.4915) falls *below* its shuffled-label AUC (0.5039 over
seven permutations). Precision lift goes to −8.2pp at the highest confidence
threshold: the model is most reliably wrong when most certain.

### Regime dependence

Pooled AUC conceals the most important pattern in the results.

| Fold | Period | BTC | ETH | SOL | BNB |
|---|---|---|---|---|---|
| 2 | 2022 bear | 0.416 | 0.425 | 0.563 | 0.483 |
| 4 | 2024 recovery | 0.625 | 0.585 | 0.614 | 0.502 |

Three of four symbols are at or below chance during the 2022 drawdown. The model
is a bull-market detector, not a market model — a distinction invisible in any
single aggregate metric.

### Pre-registered final test

Question: at what round-trip cost does the model become profitable, and is that
cost reachable given limit-order fill risk?

Criteria, fixed before the test:
- (a) pooled mean net return > 0
- (b) positive in at least 4 of 5 folds
- (c) break-even cost at or above 0.15% (reachable as a maker)

| Symbol | Break-even | (a) | (b) | (c) | Verdict |
|---|---|---|---|---|---|
| BTCUSDT | 0.280% | FAIL | FAIL (3/5) | PASS | **NO-GO** |
| ETHUSDT | 0.000% | FAIL | FAIL (0/5) | FAIL | **NO-GO** |
| SOLUSDT | 2.783% | PASS | FAIL (2/5) | PASS | **NO-GO** |
| BNBUSDT | 0.000% | FAIL | FAIL (1/5) | FAIL | **NO-GO** |

BTC misses by 0.02 percentage points — break-even 0.280% against a 0.300% taker
cost. Switching to maker execution at 0.15% does turn it positive (+0.130% per
trade), but the profit concentrates in 70 of 429 trades across two folds. Criterion
(b) exists to catch exactly that, and it did.

### The adverse selection finding

The most interesting result was not the one the test was designed to find.

Cheaper execution requires posting a limit order rather than crossing the spread.
Sweeping the limit price below the entry:

| Offset below close | Fill rate | Missed − filled return | Maker total |
|---|---|---|---|
| 0.00% | 100.0% | — | +0.557 |
| 0.05% | 95.6% | +0.952pp | +0.555 |
| 0.10% | 89.7% | +1.362pp | +0.308 |
| 0.50% | 48.5% | +0.891pp | **−0.179** |

Missed trades systematically outperform filled ones, and the maker route turns
negative by 0.50%. The mechanism is direct: if price never returns to your bid, it
moved up — and that was the trade worth having. Chasing price improvement destroys
more edge than it saves in fees. ETH and BNB show the same pattern (+1.0 to +1.6pp).

---

## Method

**Data.** Binance public archive (`data.binance.vision`), no API key required.
BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT at 4h resolution, January 2020 to July 2026.
14,423 bars per symbol (13,085 for SOL, listed August 2020). One 8-hour gap on
2020-02-19 from an exchange outage, flagged rather than interpolated.

**Features.** 29 columns across returns, realized volatility, momentum, volume
imbalance, candle geometry, cyclical calendar encodings, and BTC cross-asset
correlation. Every feature at row *i* uses only rows ≤ *i*, verified mechanically.

**Labels.** Triple-barrier (López de Prado). Volatility-scaled profit target and
stop loss, 30-bar horizon, intrabar high/low used for barrier touches. Bars where
both barriers fall inside one candle are labelled null rather than guessed.

**Validation.** Purged walk-forward with expanding windows and a 30-bar embargo.
Training rows whose label window reaches into the test block are removed. The last
12 months are split off as an untouched holdout.

**Model.** LightGBM binary classifier with conservative regularisation. Evaluated
on precision at confidence thresholds rather than accuracy, because the strategy
only trades its highest-confidence signals.

---

## Testing

```
216 passed
```

Passing tests prove little unless they can fail. Four were verified by breaking the
code on purpose:

| Test | Injected bug | Caught |
|---|---|---|
| Point-in-time features | `shift(-1)` on `ret_1` | Yes, all 4 symbols |
| Label horizon | Off-by-one peek at `high[j+1]` | Yes, all 4 symbols |
| Cost accounting | `COST_LOG = 0.0` | Yes, all 4 symbols |
| Holdout guard | CV built on full frame | Yes |

The purge check was also quantified rather than asserted: on naive unpurged splits
it identifies 16, 27, 19, 2 and 9 contaminated training rows across the five BTC
folds — real rows whose labels resolve after the test window opens.

---

## Repository layout

```
src/
  download.py       Binance archive → validated Parquet
  features.py       29 point-in-time features
  labels.py         Triple-barrier labelling
  baseline.py       Rule-based strategies (the bar to clear)
  splits.py         Purged walk-forward CV
  train.py          LightGBM + threshold evaluation
  cost_analysis.py  Cost sweep and fill realism
tests/              216 tests
models/             20 boosters (4 symbols x 5 folds)
data/parquet/       Generated, not committed
```

---

## Running it

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/download.py            # ~400 MB, several minutes
python src/features.py
python src/labels.py
python src/baseline.py
python src/splits.py
python src/train.py
python src/cost_analysis.py

pytest tests/
```

Every script takes `--symbols` and prints a validation summary.

---

## The holdout

2,175 BTCUSDT bars, 2025-07-30 to 2026-07-30, have never been examined.

That is the intended outcome, not an unfinished task. A holdout is spent once, on a
strategy that has already earned the right to be tested. Nothing here earned it.
It remains clean for any future work on this dataset.

---

## What this does not claim

- That crypto markets are efficient. Only that these features, at this horizon,
  do not beat these costs.
- That no ML approach can work. Higher-frequency microstructure, order book data,
  and funding rates were not tested.
- That the negative result generalises to other assets, timeframes, or periods.

## Further reading

- López de Prado, *Advances in Financial Machine Learning* — triple-barrier
  labelling, purged CV, and deflated Sharpe ratios
- Jansen, *Machine Learning for Algorithmic Trading*

See [FINDINGS.md](FINDINGS.md) for the detailed research writeup.
