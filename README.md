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

Test set of 40,000 loans (20% default rate):

| Model | ROC AUC | PR AUC |
|---|---|---|
| Logistic Regression | 0.901 | 0.754 |
| Random Forest | 0.895 | 0.688 |
| **XGBoost** | **0.910** | **0.770** |
| HistGradientBoosting | 0.909 | 0.768 |
| Stacking Ensemble | 0.909 | 0.769 |

I chose XGBoost. At a tuned threshold of 0.475 it catches 50% of defaults with 82% precision and 87.8% overall accuracy.

**Fairness.** Using the 0.8 disparate impact rule of thumb, home ownership came out at 0.75 (renters flagged more often than mortgage holders) and employment length at 0.88. Loan grade is at 0.09, which is expected because grade is itself a risk measure, but it shows how much the model leans on LendingClub's own grading.

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
