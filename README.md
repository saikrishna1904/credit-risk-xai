# Explainable AI for Credit Risk Prediction

MSc Data Science dissertation project (York St John University).

I built a model to predict whether a LendingClub loan will be charged off, then used SHAP and LIME to explain its decisions and checked how its predictions differ across groups of borrowers. The aim was a model that is accurate and also easy for a credit team to question.

## Data

LendingClub accepted and rejected loans, 2007–2018Q4 ([Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)).

- 2.26M accepted loans. I kept the 1.35M with a final outcome (Fully Paid or Charged Off).
- Target: `default = 1` if Charged Off.
- I dropped columns with more than 80% missing values and anything only known after the loan was issued (payments, recoveries, last FICO, settlement flags), leaving 93 columns.

The raw files are too large for GitHub. See [`data/README.md`](data/README.md) to get them.

## Pipeline

| Notebook | What it does |
|---|---|
| [`01_data_preparation`](notebooks/01_data_preparation.ipynb) | Loads the data with Polars, filters outcomes, removes leakage and sparse columns |
| [`02_eda`](notebooks/02_eda.ipynb) | Default rates by grade, purpose, term, income and more |
| [`03_modelling`](notebooks/03_modelling.ipynb) | Preprocessing on a 200k stratified sample, 5 models compared, threshold tuning |
| [`04_explainability`](notebooks/04_explainability.ipynb) | SHAP summary and dependence plots, LIME for single applicants |
| [`05_evaluation_threshold`](notebooks/05_evaluation_threshold.ipynb) | Model comparison, calibration, scoring function with low/medium/high risk bands |
| [`06_fairness_accepted`](notebooks/06_fairness_accepted.ipynb) | Predicted default rates and disparate impact by purpose, home ownership, employment length and grade |
| [`07_fairness_rejected`](notebooks/07_fairness_rejected.ipynb) | Scores a 300k sample of rejected applications |
| [`08_monotonic_constraints`](notebooks/08_monotonic_constraints.ipynb) | Retrains XGBoost with monotonic constraints and checks that risk always moves in a defensible direction |
| [`09_out_of_time_drift`](notebooks/09_out_of_time_drift.ipynb) | Trains on 2007–2015, tests on 2016, 2017 and 2018 separately, and measures drift with PSI |
| [`10_counterfactuals`](notebooks/10_counterfactuals.ipynb) | Finds the smallest realistic changes that would get a declined applicant approved |
| [`11_reject_inference`](notebooks/11_reject_inference.ipynb) | Uses rejected applications to correct for the model only ever seeing approved loans |

Run them in order. Each one reads what the previous one saved.

## Results

Test set of 40,000 loans (20% default rate), after fixing target leakage (see below):

| Model | ROC AUC | PR AUC |
|---|---|---|
| Logistic Regression | 0.716 | 0.378 |
| Random Forest | 0.715 | 0.381 |
| **XGBoost** | **0.727** | **0.399** |
| HistGradientBoosting | 0.726 | 0.396 |
| Stacking Ensemble | 0.725 | 0.397 |

XGBoost came out best, and it is the model used for the SHAP and LIME explanations. A random guess would score 0.5 ROC AUC and 0.20 PR AUC (the default rate), so the model is picking up real signal, and these numbers are in the range usually reported for LendingClub when only application-time information is used.

### Fixing target leakage

My first version scored 0.91 ROC AUC, which looked too good. The cause was target encoding: I replaced high-cardinality columns such as `emp_title` with the average default rate for each category, but I computed those averages on the full dataset before the train/test split. Many job titles appear only once, so for those loans the encoded value was simply the loan's own outcome, and the test set leaked into the features.

In v2 the data is split first, training rows are encoded out-of-fold (a row never sees its own label), test rows use averages learned from training data only, and rare categories are smoothed towards the overall mean. Optuna now tunes hyperparameters on a validation set carved out of the training data, so the test set is only used once at the end. ROC AUC dropped from 0.91 to 0.73, which is the honest number.

### Fairness

Notebook 06 retrains the fixed pipeline (ROC AUC 0.727, matching notebook 03) and compares groups on the test set. Each group's average predicted default probability sits close to its actual default rate:

| Group | Avg predicted probability | Actual default rate |
|---|---|---|
| Renters | 0.228 | 0.230 |
| Own home | 0.204 | 0.204 |
| Mortgage | 0.176 | 0.174 |
| Employed 10+ years | 0.189 | 0.187 |
| Employed < 1 year | 0.197 | 0.199 |

So the model is well calibrated within these groups. Renters get higher scores because renters in the data default more often, not because the model is overshooting for them.

The disparate impact ratios at the default 0.5 threshold look severe (home ownership 0.32, employment length 0.64, purpose 0.17), but at that threshold the model flags only about 3% of loans, so the ratio compares very small rates and moves a lot with a handful of loans. Comparing average scores instead gives ratios of 0.77 for home ownership and 0.94 for employment length. Home ownership sits just under the 0.8 rule of thumb, so it is the attribute I would monitor if this model were used for real decisions.

### Monotonic constraints

A bank cannot defend a model where a higher interest rate or a lower income makes someone look *safer*. Gradient boosting can learn wiggles like that from noise, so in notebook 08 I retrained XGBoost with `monotone_constraints` on 22 features: higher interest rate, debt-to-income, utilisation and past delinquencies must never lower risk, and higher income, FICO score and share of accounts never delinquent must never raise it.

To test this, I took 2,000 test applicants, made one feature half a standard deviation worse, and counted how often predicted risk went down:

| | Unconstrained | Monotonic |
|---|---|---|
| ROC AUC | 0.727 | 0.726 |
| PR AUC | 0.401 | 0.395 |
| Applicants whose risk moved the wrong way (avg across features) | 11.8% | **0%** |
| ...for interest rate specifically | 64.8% | **0%** |
| ...for annual income | 14.7% | **0%** |

The worst case was interest rate. The most likely reason is that interest rate and sub-grade carry almost the same information, so the unconstrained model took the main effect from sub-grade and learned noisy, often backwards patterns for interest rate within a grade. The constraints removed every wrong-direction prediction for a ROC AUC cost of 0.002, so I would use the monotonic model in practice.

### Out-of-time testing and drift

A random split mixes loans from every year, so the model is tested on the same era it learned from. A lender uses a model on future applicants, so in notebook 09 I trained the monotonic model on loans issued 2007–2015 (300k-loan sample) and tested each later year separately:

| Evaluation | Loans | Default rate | ROC AUC | PR AUC |
|---|---|---|---|---|
| Random 80/20 split | 60,000 | 20.0% | 0.728 | 0.401 |
| Out-of-time 2016 | 65,424 | 23.1% | 0.717 | 0.427 |
| Out-of-time 2017 | 37,592 | 23.1% | 0.705 | 0.406 |
| Out-of-time 2018 | 12,650 | 16.3% | 0.698 | 0.289 |

ROC AUC falls a little each year the model gets further from its training data, from 0.728 on a random split to 0.698 two to three years later. The random split overstates how the model would perform in use.

I measured drift with the Population Stability Index (PSI) against the training period (below 0.10 stable, 0.10–0.25 worth watching, above 0.25 review the model):

| Variable | 2016 | 2017 | 2018 |
|---|---|---|---|
| Model score | 0.014 | 0.019 | 0.032 |
| Interest rate | 0.082 | 0.097 | **0.147** |
| Sub-grade (encoded) | 0.077 | 0.089 | **0.122** |
| Average current balance | 0.059 | 0.069 | 0.097 |
| Debt-to-income | 0.009 | 0.005 | 0.045 |

The overall score distribution stayed stable, but by 2018 interest rate and sub-grade had moved into the "watch" range as LendingClub's pricing changed. Score PSI alone would have missed this, which is why I track the key inputs as well as the output.

The default rates also need care. Only finished loans (Fully Paid or Charged Off) are in the data, and most loans issued in 2017–2018 had not reached the end of their term by the end of 2018. The finished ones are mostly early payoffs and early defaults, which is why 2018 shows a lower default rate than 2016–2017. In a real deployment I would measure performance on a fixed window, for example defaults within 12 months of issue, so every year is judged the same way.

### Counterfactual explanations: "what would get me approved?"

SHAP explains why a score is high, but a declined customer mostly wants to know what they could change. In notebook 10 I set an example policy of declining the riskiest 20% of applicants (cut-off: 30.7% predicted default risk). On the test set, approved loans actually defaulted 14.8% of the time and declined loans 41.0%, so the cut-off separates risk well.

For each declined applicant, the notebook searches for the smallest combination of three realistic actions that brings their risk under the cut-off, with each action capped at a 50% reduction:

- **Reduce other debt** (debt-to-income)
- **Pay down credit cards** (card balance and utilisation together)
- **Borrow less** (loan amount and monthly payment together)

Income, employment, credit history and the interest rate are never changed. Because the model is monotonic, reducing debt can never raise the predicted risk, so every suggestion points the right way.

Examples from the test set:

| Applicant | Risk before | Suggested change | Risk after |
|---|---|---|---|
| #30912 | 34.3% | Reduce debt-to-income by 20% | 30.4% (approved) |
| #24444 | 32.4% | Borrow 20% less | 30.1% (approved) |
| #27193 | 33.1% | Reduce debt-to-income by 10% and borrow 10% less | 29.6% (approved) |

Across 500 declined applicants:

| | |
|---|---|
| Could reach approval within the limits | 282 (56%) |
| ...with one change | 61% of those |
| ...with two or more | 39% of those |
| Solutions that borrow less (typical cut 40%) | 72% |
| Solutions that reduce other debt (typical cut 30%) | 58% |
| Solutions that pay down cards (typical cut 10%) | 16% |

The other 44% were too far above the cut-off for changes of this size, which is also useful to tell a customer honestly. These suggestions come from the model, not a guarantee of approval, and the actions are treated separately (for example, borrowing less would also slightly lower debt-to-income in reality, which the search does not model).

### Reject inference

Every model above learned only from loans LendingClub approved, so it has never seen how the people it turned away would have behaved. In notebook 11 I used a sample of the 27M rejected applications to test how much that matters. The rejected file only shares four columns with the accepted file, so this part uses a smaller model built on those (which is why its ROC AUC is 0.64 rather than 0.73):

| Feature (median) | Accepted | Rejected |
|---|---|---|
| Credit score | 690 | 637 |
| Debt-to-income | 17.6% | 19.9% |
| Employment length (years) | 6 | 0 |
| Loan amount | $12,000 | $10,000 |

I compared a baseline model trained on accepted loans only with **fuzzy augmentation**, where each rejected applicant is added to training twice (as a default and as repaid), weighted by their estimated default probability. Rejected applicants are usually riskier than an accepted-only model predicts, so I also tested inflating their default odds by 1.5x and 2x. Those factors are assumptions, so I report all of them rather than picking one:

| Model | Accepted test ROC AUC | Avg risk given to rejected applicants | Rejected applicants it would approve* |
|---|---|---|---|
| Baseline (accepted only) | 0.640 | 27.1% | 55.8% |
| Fuzzy augmentation (x1) | 0.640 | 27.2% | 55.6% |
| Fuzzy augmentation (x1.5) | 0.638 | 34.1% | 37.5% |
| Fuzzy augmentation (x2) | 0.636 | 39.6% | 20.7% |

*Using a cut-off that declines the riskiest 20% of accepted applicants.

What this shows:

- **The accepted-only model would approve over half of the applicants LendingClub rejected.** It treats them as only slightly riskier than accepted borrowers because it has never seen that part of the population.
- **Plain fuzzy augmentation changes almost nothing.** The rejects are labelled by the baseline model itself, so it mostly learns its own opinion back. This is a known weakness of the method.
- **Adding a realistic assumption about rejects changes the picture a lot** (55.8% approved down to 20.7%) while ranking quality on accepted loans barely moves (0.640 to 0.636).

Rejected applicants never received loans, so there is no outcome data to prove which version is right. In practice a lender would settle the inflation factor using a small "test-and-learn" sample of approved borderline applicants. This analysis shows how sensitive approval decisions are to that choice.

Plots are in [`outputs/`](outputs/) and summary tables in [`results/`](results/).

## Limitations

- Models were trained on a 200k sample rather than all 1.35M loans, to fit in memory.
- The rejected loans file only has 9 columns, so most model features are missing for those applicants. Notebook 07 shows the scoring works end to end, but its group results are not meaningful.
- The fairness checks use proxy attributes. LendingClub does not publish protected characteristics like age or gender.

## Run it yourself

```bash
git clone https://github.com/saikrishna1904/credit-risk-xai.git
cd credit-risk-xai
pip install -r requirements.txt
jupyter notebook
```

## Tools

Python, Polars, pandas, scikit-learn, XGBoost, Optuna, SHAP, LIME, matplotlib, seaborn
