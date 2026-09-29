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
