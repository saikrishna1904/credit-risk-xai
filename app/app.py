"""Credit risk explainer: score an applicant, explain the decision, and show a path to approval."""
import itertools
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).parent
REPO_URL = "https://github.com/saikrishna1904/credit-risk-xai"

st.set_page_config(page_title="Credit Risk Explainer", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model(str(HERE / "model.json"))
    meta = json.loads((HERE / "model_meta.json").read_text())
    return booster, meta


booster, meta = load_model()
FEATURES = meta["features"]
CODES = meta["codes"]
D = meta["defaults"]
CUTOFF = meta["cutoff"]

LABELS = {
    "loan_amnt": "Loan amount", "int_rate": "Interest rate", "installment": "Monthly payment",
    "annual_inc": "Annual income", "dti": "Debt-to-income", "fico_range_low": "Credit score (FICO)",
    "revol_util": "Card utilisation", "revol_bal": "Card balance", "open_acc": "Open accounts",
    "total_acc": "Total accounts", "inq_last_6mths": "Credit enquiries (6 months)",
    "delinq_2yrs": "Late payments (2 years)", "pub_rec": "Public records", "mort_acc": "Mortgage accounts",
    "term_code": "Loan term", "grade_code": "LendingClub grade", "emp_length_code": "Employment length",
    "home_ownership_code": "Home ownership", "verification_status_code": "Income verification",
    "purpose_code": "Loan purpose",
}


def monthly_payment(amount, rate_pct, term):
    n = 60 if term.startswith("60") else 36
    r = rate_pct / 100 / 12
    return amount * r / (1 - (1 + r) ** -n) if r > 0 else amount / n


def encode(app):
    """Turn the form values into the model's feature row(s)."""
    rows = app if isinstance(app, list) else [app]
    out = []
    for a in rows:
        r = {}
        for c in meta["numeric"]:
            r[c] = float(a[c]) if a.get(c) is not None else np.nan
        for c in meta["categorical"]:
            r[c + "_code"] = float(CODES[c][a[c]]) if a.get(c) in CODES[c] else np.nan
        out.append(r)
    return pd.DataFrame(out)[FEATURES]


def predict(app):
    X = encode(app)
    return booster.predict(xgb.DMatrix(X, feature_names=FEATURES))


def contributions(app):
    X = encode(app)
    contrib = booster.predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[0]
    return pd.Series(contrib[:-1], index=FEATURES)  # last column is the bias term


def display_value(feature, app):
    base = feature.replace("_code", "")
    v = app.get(base)
    if feature in ("loan_amnt", "annual_inc", "revol_bal", "installment"):
        return f"${v:,.0f}"
    if feature in ("int_rate", "dti", "revol_util"):
        return f"{v:.1f}%"
    if isinstance(v, float) and v.is_integer():
        return f"{int(v)}"
    return str(v)


# ---------------------------------------------------------------- inputs
st.title("Credit Risk Explainer")
st.caption(
    "Enter an applicant's details to see their predicted default risk, what drove it, "
    "and what they could change to be approved. Trained on LendingClub loans 2007–2018. "
    "A portfolio demo, not a real lending decision."
)

def opts(c):
    return list(CODES[c])

def idx(c):
    o = opts(c)
    return o.index(D[c]) if D.get(c) in o else 0

with st.form("applicant"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Loan")
        loan_amnt = st.number_input("Loan amount ($)", 1000, 40000, int(round(D["loan_amnt"], -3)), step=500)
        term = st.selectbox("Term", opts("term"), index=idx("term"))
        purpose = st.selectbox("Purpose", opts("purpose"), index=idx("purpose"),
                               format_func=lambda s: s.replace("_", " ").capitalize())
        grade = st.selectbox("LendingClub grade", opts("grade"), index=idx("grade"))
        int_rate = st.slider("Interest rate (%)", 5.0, 31.0, float(round(D["int_rate"], 1)), 0.1)
    with c2:
        st.subheader("About the applicant")
        annual_inc = st.number_input("Annual income ($)", 5000, 500000, int(round(D["annual_inc"], -3)), step=1000)
        emp_length = st.selectbox("Employment length", opts("emp_length"), index=idx("emp_length"))
        home_ownership = st.selectbox("Home ownership", opts("home_ownership"), index=idx("home_ownership"))
        verification_status = st.selectbox("Income verification", opts("verification_status"),
                                           index=idx("verification_status"))
        mort_acc = st.number_input("Mortgage accounts", 0, 20, int(D["mort_acc"]))
    with c3:
        st.subheader("Credit profile")
        fico_range_low = st.slider("Credit score (FICO)", 600, 850, int(D["fico_range_low"]), 5)
        dti = st.slider("Debt-to-income (%)", 0.0, 50.0, float(round(D["dti"], 1)), 0.5)
        revol_util = st.slider("Card utilisation (%)", 0.0, 120.0, float(round(D["revol_util"])), 1.0)
        revol_bal = st.number_input("Card balance ($)", 0, 200000, int(round(D["revol_bal"], -2)), step=500)
        inq_last_6mths = st.number_input("Credit enquiries, last 6 months", 0, 10, int(D["inq_last_6mths"]))
        delinq_2yrs = st.number_input("Late payments, last 2 years", 0, 20, int(D["delinq_2yrs"]))
        pub_rec = st.number_input("Public records", 0, 10, int(D["pub_rec"]))
        open_acc = st.number_input("Open accounts", 0, 60, int(D["open_acc"]))
        total_acc = st.number_input("Total accounts", 1, 120, int(D["total_acc"]))
    submitted = st.form_submit_button("Assess applicant", type="primary", use_container_width=True)

app = dict(loan_amnt=float(loan_amnt), term=term, purpose=purpose, grade=grade, int_rate=float(int_rate),
           annual_inc=float(annual_inc), emp_length=emp_length, home_ownership=home_ownership,
           verification_status=verification_status, mort_acc=float(mort_acc),
           fico_range_low=float(fico_range_low), dti=float(dti), revol_util=float(revol_util),
           revol_bal=float(revol_bal), inq_last_6mths=float(inq_last_6mths), delinq_2yrs=float(delinq_2yrs),
           pub_rec=float(pub_rec), open_acc=float(open_acc), total_acc=float(total_acc))
app["installment"] = monthly_payment(app["loan_amnt"], app["int_rate"], app["term"])
app = {k: v for k, v in app.items() if k in meta["numeric"] or k in meta["categorical"]}

# ---------------------------------------------------------------- result
risk = float(predict(app)[0])
approved = risk < CUTOFF

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Predicted default risk", f"{risk:.1%}")
m2.metric("Approval cut-off", f"{CUTOFF:.1%}", help="Declines the riskiest 20% of applicants in the test data.")
m3.metric("Monthly payment", f"${monthly_payment(loan_amnt, int_rate, term):,.2f}")
if approved:
    st.success(f"**Approved.** Risk is {CUTOFF - risk:.1%} below the cut-off.")
else:
    st.error(f"**Declined.** Risk is {risk - CUTOFF:.1%} above the cut-off.")

left, right = st.columns([1.1, 1])

with left:
    st.subheader("What drove this score")
    st.caption("Each bar is how much that detail pushed the risk up (red) or down (green) "
               "compared with an average applicant. Computed with SHAP values from XGBoost.")
    contrib = contributions(app)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8]
    chart_df = pd.DataFrame({
        "factor": [f"{LABELS.get(f, f)}: {display_value(f, app)}" for f in top.index],
        "effect": top.values,
        "direction": ["Raises risk" if v > 0 else "Lowers risk" for v in top.values],
    })
    chart = (alt.Chart(chart_df)
             .mark_bar()
             .encode(x=alt.X("effect:Q", title="Effect on risk (log-odds)"),
                     y=alt.Y("factor:N", sort=None, title=None),
                     color=alt.Color("direction:N",
                                     scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                     range=["#d93025", "#188038"]),
                                     legend=alt.Legend(title=None, orient="bottom")),
                     tooltip=["factor", alt.Tooltip("effect:Q", format=".3f")])
             .properties(height=320))
    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Path to approval")
    ACTIONS = {
        "Reduce other debt": ["dti"],
        "Pay down credit cards": ["revol_bal", "revol_util"],
        "Borrow less": ["loan_amnt"],
    }
    ACTIONS = {k: [c for c in v if c in app] for k, v in ACTIONS.items()}
    ACTIONS = {k: v for k, v in ACTIONS.items() if v}
    names = list(ACTIONS)
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def changed(fracs):
        a = dict(app)
        for name, f in zip(names, fracs):
            for c in ACTIONS[name]:
                a[c] = app[c] * (1 - f)
        if "installment" in a:
            a["installment"] = monthly_payment(a["loan_amnt"], a["int_rate"], term)
        return a

    if approved:
        st.info("This applicant is already approved. Here is how much room they have:")
        st.write(f"Their risk could rise by **{CUTOFF - risk:.1%}** before they would be declined.")
    else:
        grid = list(itertools.product(steps, repeat=len(names)))
        variants = [changed(g) for g in grid]
        probs = predict(variants)
        ok = [i for i, p in enumerate(probs) if p < CUTOFF]
        if not ok:
            st.warning(f"Even with 50% reductions in all three areas the risk only falls to "
                       f"{probs.min():.1%}, still above the cut-off. The main drivers are things "
                       "that take longer to change, such as credit score or income.")
        else:
            best = min(ok, key=lambda i: (sum(grid[i]) + 0.01 * sum(f > 0 for f in grid[i]), probs[i]))
            g, new = grid[best], changed(grid[best])
            st.write("The smallest combination of realistic changes that gets this applicant approved:")
            rows = []
            fmt = {"dti": "{:.1f}%", "revol_util": "{:.0f}%", "revol_bal": "${:,.0f}", "loan_amnt": "${:,.0f}"}
            for name, f in zip(names, g):
                if f == 0:
                    continue
                for c in ACTIONS[name]:
                    rows.append({"Change": name, "Item": LABELS[c],
                                 "Now": fmt[c].format(app[c]), "Target": fmt[c].format(new[c]),
                                 "Reduction": f"{f:.0%}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.success(f"New predicted risk: **{probs[best]:.1%}** → approved")

        # one action at a time
        st.caption("Each change on its own:")
        single = []
        for k, name in enumerate(names):
            curve = []
            for s in steps:
                fr = [0.0] * len(names)
                fr[k] = s
                curve.append(float(predict(changed(fr))[0]))
            single += [{"action": name, "reduction": s * 100, "risk": r * 100} for s, r in zip(steps, curve)]
        line = (alt.Chart(pd.DataFrame(single)).mark_line(point=True)
                .encode(x=alt.X("reduction:Q", title="Reduction (%)"),
                        y=alt.Y("risk:Q", title="Predicted risk (%)", scale=alt.Scale(zero=False)),
                        color=alt.Color("action:N", legend=alt.Legend(title=None, orient="bottom"))))
        rule = alt.Chart(pd.DataFrame({"y": [CUTOFF * 100]})).mark_rule(strokeDash=[5, 4], color="#5f6368").encode(y="y:Q")
        st.altair_chart((line + rule).properties(height=260), use_container_width=True)

st.divider()
with st.expander("About this model"):
    st.markdown(f"""
- **Model:** XGBoost with monotonic constraints (a higher interest rate, DTI or utilisation can never lower risk;
  a higher income or credit score can never raise it).
- **Data:** LendingClub loans 2007–2018 with a final outcome; trained on {meta['train_rows']:,} loans.
- **Test performance:** ROC AUC {meta['test_roc_auc']:.3f}, PR AUC {meta['test_pr_auc']:.3f}
  (average default rate {meta['base_default_rate']:.1%}).
- **Cut-off:** declines the riskiest 20% of test applicants. This is an example policy, not LendingClub's.
- **Explanations:** SHAP values from XGBoost; the path to approval searches reductions of up to 50% in
  debt-to-income, card balances and loan amount.
- Full analysis, including leakage fixes, fairness, drift and reject inference: [GitHub repo]({REPO_URL})
""")
