# Pricing the decision, not the prediction

**Razorpay Buildathon — Track 02, AI Risk Manager**

A fraud detector's threshold is a business decision, not a statistical one. This project
prices that decision in rupees and shows what the standard approach costs.

Two findings, both measured on a held-out set of 118,108 transactions:

1. **F1 cannot express merchant economics.** The threshold that maximises money moves
   from 0.70 to 0.84 as a merchant's cost of blocking a genuine customer rises. The
   F1-optimal threshold sits at 0.81 in every case — it has no input for cost, so it
   cannot move. For one merchant profile that gap is worth **₹23.8 lakh**.

2. **Binary allow/block leaves more on the table than the model does.** Adding a
   step-up challenge as a third action is worth **₹1.72 crore** under a realistic
   operational constraint, and lifts fraud caught from 56% to 69% *while blocking
   fewer genuine customers*.

The second finding survives 33 of 36 assumption combinations and never goes negative.

---

## The problem with F1

A model outputs a probability. Someone has to choose the cutoff. That choice trades two
errors against each other:

| | Model allows | Model blocks |
|---|---|---|
| **Genuine** | Fine | Customer rejected — lost margin, possibly lost customer |
| **Fraud** | Chargeback — lost goods and fees | Caught |

F1 weights those two errors equally. No merchant does. A blocked ₹200 order and a missed
₹50,000 fraud are not the same event, and a merchant selling ₹80,000 jewellery to repeat
customers has a completely different calculus from one selling ₹200 phone cases.

Optimising for F1 is therefore an assumption — that both failure modes cost the same —
made silently, by almost everyone, without noticing.

## What this does instead

Attach rupees to each cell, then sweep the threshold to find where the merchant keeps
the most money.

```
fraud caught       = + (amount + chargeback fee)
fraud missed       = − (amount + chargeback fee)
genuine blocked    = − (margin × amount + expected churn cost)
genuine allowed    =   0
```

Run for three merchant profiles:

| Profile | Margin | Churn cost | Money-optimal | F1-optimal |
|---|---|---|---|---|
| Conservative | 20% | ₹150 | **0.70** | 0.81 |
| Moderate | 30% | ₹400 | **0.80** | 0.81 |
| Aggressive | 40% | ₹900 | **0.84** | 0.81 |

The money-optimal threshold responds to the merchant. F1's does not.

Worth being precise about the moderate case: F1 lands at 0.81 against a money-optimal
0.80, a gap of only ₹2.2 lakh. F1 was *nearly right* there. That is the actual problem —
not that F1 is always wrong, but that it is sometimes accidentally right, and it gives
you no way to tell which situation you are in.

## The third action

Binary allow/block forces one threshold to answer two different questions: *is this risky
enough to interrupt?* and *is this risky enough to refuse?* Those are not the same
question.

Real payment systems have a cheap middle option — step-up authentication. A fraudster
usually cannot pass an OTP they do not control. A genuine customer usually can, with some
drop-off. That asymmetry is exploitable.

With two thresholds instead of one:

| | Two-action | Three-action | Change |
|---|---|---|---|
| Fraud stopped | 56.2% | 79.2% | **+23.0 pts** |
| Genuine customers blocked | 3,429 | 248 | **−93%** |
| Net saved | ₹1.87 Cr | ₹4.04 Cr | **+₹2.17 Cr** |

More fraud caught *and* fewer customers blocked. Not a trade-off — a strictly better
policy, because the binary version was answering two questions with one number.

There is a structural point here worth stating: every two-action policy is a point on the
diagonal `t_low = t_high` of the two-threshold space. The two-action policy space is a
one-dimensional line through a two-dimensional space, and the optimum does not lie on it.
See `artifacts/three_action.png`.

## The honest version of that number

The unconstrained optimum wants to challenge 24.5% of traffic. Real 3DS step-up rates sit
nearer 5–15%. Challenging a quarter of customers has costs this model does not capture —
checkout conversion, PSP relationships, merchant reputation.

Constrained to a realistic 10%:

| Challenge cap | Challenged | Fraud stopped | Gain |
|---|---|---|---|
| 5% | 4.8% | 58.8% | ₹1.13 Cr |
| **10%** | **9.8%** | **69.4%** | **₹1.72 Cr** |
| Unconstrained | 24.5% | 79.2% | ₹2.17 Cr |

**₹1.72 crore at an operationally plausible challenge rate** is the number to hold onto.

## Does it survive its own assumptions?

The three-action result rests on four numbers I chose rather than measured. So I swept
them. Gain in ₹ lakh, as the assumptions get worse in both directions:

| bypass ＼ abandon | 2% | 6% | 12% | 20% | 30% | 40% |
|---|---|---|---|---|---|---|
| **5%** | 315.3 | 260.0 | 211.0 | 162.7 | 123.3 | 90.8 |
| **15%** | 268.2 | 216.6 | 170.5 | 127.2 | 89.4 | 61.4 |
| **30%** | 201.5 | 156.8 | 114.8 | 80.9 | 49.3 | 25.7 |
| **45%** | 141.1 | 103.5 | 69.1 | 41.1 | 19.0 | 6.9 |
| **60%** | 89.5 | 58.8 | 33.6 | 14.7 | 3.5 | 0.0 |
| **75%** | 44.9 | 23.2 | 8.7 | 1.1 | 0.0 | 0.0 |

*(rows: how often a fraudster defeats the challenge; columns: how often a genuine
customer abandons at it)*

Positive in **33 of 36 combinations**, and never negative. The worst corner — a step-up
system where three quarters of fraudsters get through and two fifths of real customers
give up — degrades to ₹0, not to a loss. That is structural: the three-action policy space
contains the two-action space, so it can match it whenever the middle option is worthless.

Two things this sweep settled:

- **Cost per challenge barely matters.** ₹5 → ₹200, a fortyfold increase, moves the gain
  only from ₹214.9L to ₹167.4L. Chargeback losses dominate; challenge infrastructure is
  rounding error.
- **Bypass rate matters more than abandonment.** The fragile assumption is *can OTP
  actually stop a fraudster*, not *does it annoy customers*. That tells a merchant where
  to invest.

## Model

LightGBM, 434 features, IEEE-CIS Fraud Detection (590,540 transactions, 3.50% fraud).

**Split by time, not randomly.** Sorted on `TransactionDT`, trained on the first 80%,
evaluated on the last 20%. Fraud patterns drift; a random split leaks future information
backwards and inflates every number downstream. Train and test fraud rates differ (3.51%
vs 3.44%) — that drift is real, and it is the reason the split is time-ordered.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9112 |
| PR-AUC | 0.5178 |
| PR-AUC, random baseline | 0.0344 |
| Early stopping | iteration 107 |

PR-AUC is the honest one at a 3.4% base rate; ROC-AUC is inflated by the large pile of
easy true negatives. 15× better than chance on the part that is hard.

No feature engineering, no hyperparameter search, no ensembling. The model is deliberately
ordinary — the finding is about what you do with the scores, and a fancier model would
only move every number in this document up together.

## Limitations

- **Recall is 69% at the constrained optimum.** Roughly a third of fraud still gets
  through. Blocking harder costs more in lost customers than it saves.
- **The four step-up assumptions are chosen, not measured.** The sweep above is the
  defence, not a claim that the specific values are right. A real deployment would
  measure them per-merchant and re-solve.
- **IEEE-CIS is US e-commerce data.** Amounts converted at a flat ₹85/USD to make the
  economics legible for an Indian merchant. The fraud patterns are not Indian patterns
  and the conversion is a presentational device, not a claim about UPI or Indian cards.
- **Churn cost is the weakest input.** There is no clean published figure for what share
  of blocked customers never return. It is a judgement, and it is why the analysis runs
  three merchant profiles rather than one.
- **No segment-level thresholds.** One threshold pair for all traffic is itself an
  assumption — first-time buyers, repeat customers and high-value orders probably each
  want different treatment. Not tested.
- **The 90-day claim history and repeat-abuse patterns are not modelled.** Each
  transaction is scored independently.

## Running it

```bash
pip install -r requirements.txt

python 01_baseline.py        # train, time-split, save held-out predictions
python 02_cost_curve.py      # threshold sweep in rupees
python 03_plot.py            # readable version of the cost curve
python 04_three_action.py    # allow / challenge / block, joint sweep
python 05_sensitivity.py     # 36 assumption combinations + challenge-rate caps

streamlit run app.py         # interactive policy explorer
```

`01_baseline.py` needs `train_transaction.csv` and `train_identity.csv` from the
[IEEE-CIS competition](https://www.kaggle.com/c/ieee-fraud-detection/data) in `data/`.
Everything downstream reads `artifacts/holdout_predictions.csv`, which is committed — so
the analysis and the demo run without the raw data.

## Artifacts

| File | What it shows |
|---|---|
| `artifacts/cost_curve.png` | Money-optimal threshold moves with merchant economics; F1's does not |
| `artifacts/three_action.png` | Policy surface — every two-action policy lies on one line |
| `artifacts/sensitivity.png` | Gain across 36 assumption combinations, and the cost of a challenge-rate cap |
| `artifacts/*_summary.json` | Every number in this document, with the assumptions that produced it |

## Assumptions, in one place

| Parameter | Value | Basis |
|---|---|---|
| USD → INR | 85 | Presentational; IEEE-CIS amounts are USD |
| Chargeback fee | ₹1,500 | Razorpay's published range is ₹200–600 in direct dispute fees; their enterprise guidance cites ₹1,500–8,000 all-in |
| Gross margin | 20 / 30 / 40% | Typical Indian D2C range |
| Lost customer value | ₹150 / ₹400 / ₹900 | Judgement — swept, not asserted |
| Fraudster defeats challenge | 15% | SIM swap, phishing, social engineering |
| Genuine customer abandons | 6% | Step-up friction drop-off |
| Cost per challenge | ₹5 | 3DS/SMS infrastructure — shown not to matter |
| Abandoned-vs-blocked churn | 40% | Friction annoys less than rejection |

---

Built solo. Defence-only: nothing here detects, generates or assists fraud — it decides
what a merchant should do with a risk score it already has.
