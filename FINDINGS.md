# Findings

Detailed writeup of the research, the decisions behind it, and the reasoning that
led to a NO-GO conclusion.

---

## 1. The question

Can a gradient-boosted classifier, trained on standard technical and microstructure
features, identify 4-hour cryptocurrency bars where a long position is profitable
after realistic transaction costs?

Framed as a decision rather than a prediction: the system should stay silent most of
the time and signal only when confidence is high enough that the expected return
exceeds the cost of trading.

---

## 2. Establishing the bar first

The most useful decision in the project was measuring the cost wall before building
any model.

Six rule-based strategies were evaluated across 6.5 years and four symbols. On
BTCUSDT, taking every labelled bar yields **+0.251% gross per trade**. A round trip
of 0.1% fee and 0.05% slippage per side costs **0.30%**. Net: **−0.049%**.

Two comparisons made the picture unambiguous.

**Random selection.** Choosing 5% of bars at random returned −0.046% — statistically
indistinguishable from trading everything (−0.049%). If a random selector matches
your "smart" rules, the rules carry no information.

**Cross-symbol consistency.** SOL momentum returned +0.810% net with a profit factor
of 1.20. The identical rule returned −0.146% on BNB, +0.002% on BTC, −0.034% on ETH.
One symbol out of four, during Solana's 2021 run from roughly $1 to $260. Testing
six strategies on four symbols is 24 combinations; roughly one will look excellent
by chance. This was that one.

The bar was therefore explicit before modelling began: **beat 0.30% per trade.**

---

## 3. Label construction

Target definition mattered more than model choice.

Triple-barrier labelling was used: for each bar, simulate a long entry at the close
and record which of three events occurs first — profit target, stop loss, or horizon
expiry at 30 bars (5 days).

Three details that materially affect the result:

**Volatility-scaled barriers.** Fixed percentage barriers mean something different in
calm and turbulent markets. Barriers are set at `close × (1 ± σ√H)` where σ is the
30-bar realized volatility available at that point in time. Resulting widths: 6.18%
mean on BTC, 12.00% on SOL — tracking each asset's actual volatility.

**Intrabar highs and lows.** Checking only closes systematically undercounts barrier
touches, since a barrier can be hit and retraced within a single candle.

**`t1`, the label resolution time.** Recorded for every bar. Without it, purged
cross-validation is impossible, because there is no way to know which training rows
have labels that resolve inside a test window.

Bars where both barriers fall within the same candle are labelled null and flagged
`ambiguous` — OHLC data cannot resolve the ordering, and guessing would introduce
noise correlated with volatility. This affected 2–12 bars per symbol.

**Base rate: 32.5% wins on BTC.** Of trades that resolved at a barrier rather than
expiring, 52.8% were upward — a mild bullish tilt consistent with the period, and
evidence the labelling carries no directional bug.

---

## 4. Preventing leakage

Financial ML fails silently. A leaking model produces a plausible number, not an
error. Three defences were built and each was verified capable of failing.

**Point-in-time features.** Every feature is computed twice — once on the full
dataset, once on data truncated at row N — and the value at row N−1 must match
exactly. Verified by injecting `shift(-1)` into `ret_1`; the test failed immediately
on all four symbols with a clear diff.

This surfaced a real trap: Polars' `when(null_condition).then(...)` falls through to
`otherwise`, which would have silently fabricated a `0.0` gain at RSI's first bar
instead of preserving null.

**Purged, embargoed cross-validation.** Labels overlap by up to 30 bars. Training
rows whose `t1` falls inside the test window are purged; rows within 30 bars of the
test boundary are embargoed for serial correlation.

The check was quantified rather than assumed: on naive unpurged splits it identifies
**16, 27, 19, 2 and 9** contaminated training rows across the five BTC folds. The
cost of purging is about 30 rows out of training sets ranging from 2,400 to 12,000 —
effectively free.

**Permutation testing.** Labels are shuffled within the training set and the model
retrained. AUC should collapse to 0.50. BTC returned 0.4930.

A single shuffle proved too noisy — BNB's first draw read 0.5650, approaching the
0.58 alarm threshold. Seven shuffles gave 0.5039 with range [0.4397, 0.5650],
confirming noise rather than leakage. The check was made repeatable as a result.

---

## 5. Model results

LightGBM, 29 features (28 for BTC, where cross-correlation with itself is undefined),
conservative regularisation, 5 expanding purged folds on the dev slice only.

| Symbol | Pooled AUC | Fold range | Best lift | Best net/trade |
|---|---|---|---|---|
| BTCUSDT | 0.554 | 0.416 – 0.625 | +8.3pp | −0.021% |
| ETHUSDT | 0.518 | 0.425 – 0.600 | +4.2pp | −0.989% |
| SOLUSDT | 0.577 | 0.534 – 0.614 | +13.9pp | +2.522% |
| BNBUSDT | 0.492 | 0.429 – 0.576 | +1.7pp | −0.781% |

### BTC: the informative case

Precision rises from a 32.5% base rate to **40.8%** at confidence threshold 0.65 —
a genuine 8.3 point lift, no leakage detected, exactly the effect the threshold
design was built to find.

It still loses money at every threshold.

Working backwards from the best result: −0.021% net at threshold 0.70 implies 0.280%
gross, against `always_trade`'s 0.251%. **The model improved raw per-trade signal by
about 11%. The cost wall is 0.30%.** The improvement is real, measured, and roughly
a tenth of what is required.

### BNB: negative information

Real pooled AUC 0.4915, below its shuffled-label mean of 0.5039. Precision lift
reaches −8.2pp at threshold 0.80. The model is most reliably wrong when most
confident — worse than useless, since a naive user would size up on those signals.

### Regime dependence

The finding with the most general value.

| Fold | Period | BTC AUC | Base rate |
|---|---|---|---|
| 1 | 2021 | 0.531 | 34.5% |
| 2 | 2022 bear | **0.416** | 22.4% |
| 3 | 2023 | 0.561 | 32.9% |
| 4 | 2024 | 0.625 | 36.7% |
| 5 | 2025–26 | 0.606 | 36.0% |

Fold 2 is at or below chance on three of four symbols (0.416 / 0.425 / 0.483). The
model performs well in rising markets and fails in falling ones — it has learned a
bull-market proxy, not a market model.

A single pooled AUC of 0.554 conceals this entirely. Any evaluation reporting one
aggregate number would have missed it.

---

## 6. The pre-registered cost test

BTC's near-miss justified one further test. Criteria were fixed before running it:

- (a) pooled mean net return > 0 at the best threshold
- (b) positive in at least 4 of 5 folds
- (c) break-even round-trip cost ≥ 0.15%, i.e. reachable via maker execution

Criterion (b) was included specifically to prevent a repeat of the SOL pattern —
strong aggregate performance concentrated in one regime.

### Break-even costs

Because cost enters as `log(1−c)`, an additive constant on log returns, threshold
ranking is invariant to cost level and break-even has a closed form:
`c* = 1 − exp(−mean_gross)`.

| Symbol | Best threshold | Signals | Mean gross | Break-even |
|---|---|---|---|---|
| BTCUSDT | 0.70 | 429 | +0.280% | **0.280%** |
| ETHUSDT | 0.50 | 2,516 | −0.689% | 0.000% |
| SOLUSDT | 0.80 | 170 | +2.822% | 2.783% |
| BNBUSDT | 0.50 | 2,477 | −0.481% | 0.000% |

ETH and BNB have negative gross edge at every threshold. No cost reduction rescues
them — they lose money at zero fees.

BTC breaks even at 0.280% against a 0.300% taker cost: a miss of 0.02 percentage
points, the narrowest possible failure.

### Maker execution

Posting limit orders at 0.15% instead of crossing the spread at 0.30% does turn BTC
positive: −0.021% becomes **+0.130%** per trade. Criterion (c) passes.

Criterion (b) does not:

| Fold | Trades | Mean net (maker) |
|---|---|---|
| 1 | 160 | +0.455% |
| 2 | 31 | −1.888% |
| 3 | 168 | −0.755% |
| 4 | 27 | +3.730% |
| 5 | 43 | +1.568% |

The two largest folds — 328 of 429 trades — are roughly flat or negative. The
headline profit comes from **70 trades** across two folds, at per-trade returns
that are implausible as a repeatable effect. Folds 2 and 4 fire 31 and 27 times
respectively; a +3.730% mean over 27 trades is noise.

**Verdict: NO-GO on all four symbols.** The criteria were not revised.

---

## 7. Adverse selection

The most valuable result was incidental.

The original fill test — a limit buy at the close fills if the next bar's low reaches
it — proved degenerate on this data. Fill rate came out at 99.9%: only 16 misses in
14,422 bars. On continuous 24/7 crypto the next bar opens at the previous close
(6,603 of 14,422 bars exactly equal), so the condition holds almost by construction.
The test measured price risk when the real risk for an order resting at the touch is
queue position, which OHLC cannot observe.

Sweeping the limit price *below* the close made the effect visible:

| Offset | Filled | Fill rate | Mean filled | Mean missed | Gap | Maker total |
|---|---|---|---|---|---|---|
| 0.00% | 429 | 100.0% | +0.280% | — | — | +0.557 |
| 0.05% | 410 | 95.6% | +0.286% | +1.238% | +0.952pp | +0.555 |
| 0.10% | 385 | 89.7% | +0.230% | +1.592% | +1.362pp | +0.308 |
| 0.25% | 299 | 69.7% | +0.350% | +0.694% | +0.344pp | +0.598 |
| 0.50% | 208 | 48.5% | +0.064% | +0.955% | +0.891pp | **−0.179** |

Missed trades outperform filled ones by 0.9 to 1.4 percentage points, and the maker
route turns negative once fill rate drops below roughly half. ETH and BNB show the
same pattern at +1.0 to +1.6pp.

The mechanism is direct: **if price never returns to your bid, it moved up — and that
was the trade worth having.** A resting limit order is filled preferentially by the
adverse half of the return distribution. Chasing price improvement destroys edge
faster than it saves cost.

This is why "use limit orders to reduce fees" is poor advice for a directional
strategy, and the effect was measured in-sample rather than assumed.

---

## 8. What would be tried next

Not pursued here, and each carries a multiple-testing cost that would need
accounting for via a deflated Sharpe ratio:

**Shorter horizon.** Median holding was 22 of 30 bars, meaning most trades run near
full duration. A shorter horizon would produce more independent observations.

**Meta-labelling.** A secondary model predicting whether a primary signal will
succeed, rather than a single model doing both jobs.

**Better data.** Funding rates, open interest, and L2 order book imbalance were not
included. Microstructure features carry information that OHLCV cannot express.

**Regime conditioning.** Given that fold 2 is at or below chance, an explicit
regime filter — trading only in conditions resembling the folds where the model
works — is a natural extension, with the obvious risk of fitting to three regime
observations.

---

## 9. What was actually learned

**The cost wall dominates.** Raw signal is roughly 0.25% per trade; costs are 0.30%.
Every subsequent decision — horizon, barrier width, threshold — is downstream of that
single number. Measuring it first prevented weeks of work on an unreachable target.

**Validation design outweighs model choice.** No hyperparameter search would have
changed the conclusion. Purged CV, honest cost modelling, and the permutation check
determined what the results meant. The model was almost incidental.

**Aggregates conceal regimes.** Pooled AUC 0.554 looked like weak but real signal.
The per-fold breakdown showed a bull-market detector at or below chance in 2022.

**One symbol is one observation.** SOL looked exceptional in the baseline, in
training, and in the cost sweep. Three views of the same 2021 trend regime is not
three pieces of evidence.

**Pre-registration is the discipline that matters.** BTC missed by 0.02 percentage
points. Without criteria fixed in advance, that gap is trivially argued away — drop
the fold-consistency requirement, call the difference noise, and the project produces
a "profitable" strategy that would lose money live. The criteria were written first
precisely so they could not be renegotiated afterwards.

---

## 10. Conclusion

Standard technical and microstructure features, at 4-hour resolution with 5-day
holding periods, do not produce edge sufficient to overcome a 0.30% round-trip cost
on the four most liquid USDT pairs across 2020–2026.

BTCUSDT comes closest, breaking even at 0.280%, and passes under maker execution
only by concentrating its profit in 16% of trades across two of five folds.

The 12-month holdout remains unexamined. Nothing here earned the right to spend it.
