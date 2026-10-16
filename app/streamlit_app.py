import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usage

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/itskumarmukesh/Churn_model"

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")

usage.is_owner()
if "visit_logged" not in st.session_state:
    st.session_state["visit_logged"] = True
    usage.log_event("visit")

# starting values for the one-click examples
PRESETS = {
    "Typical customer": {},
    "High-risk example: new fiber customer on a monthly contract": {
        "tenure": 3, "MonthlyCharges": 85.0, "Contract": "Month-to-month",
        "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes", "OnlineSecurity": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Partner": "No", "Dependents": "No",
    },
    "Low-risk example: long-time customer on a two-year DSL contract": {
        "tenure": 60, "MonthlyCharges": 55.0, "Contract": "Two year",
        "InternetService": "DSL", "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "No", "OnlineSecurity": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Partner": "Yes", "Dependents": "Yes",
    },
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "models" / "telco_model.joblib")
    meta = json.loads((ROOT / "models" / "telco_meta.json").read_text(encoding="utf-8"))
    return model, meta


def choose(meta, col, label, preset, prefix, drop=(), container=st):
    """A dropdown for one text column"""
    info = meta["categorical"][col]
    options = [o for o in info["options"] if o not in drop]
    start = preset.get(col, info["default"])
    if start not in options:
        start = options[0]
    return container.selectbox(label, options, index=options.index(start), key=prefix + col)


# tenure, monthly charges and total charges are strongly related, so their weights cannot be read one by one; we combine them into a single factor
GROUPED = {"tenure", "MonthlyCharges", "TotalCharges"}


def explain(model, meta, row):
    """How much each factor moves this customer's score versus an average customer"""
    pre, clf = model[:-1], model[-1]
    x = pre.transform(row)[0]
    means = np.array(meta["feature_means"])
    effects = clf.coef_[0] * (x - means)

    rows = []
    combined = 0.0
    by_column, chosen = {}, {}          # one factor per text column (sum of its one-hot columns)
    for raw, value, mean, effect in zip(pre.get_feature_names_out(), x, means, effects):
        name = raw.split("__", 1)[1] if "__" in raw else raw
        if name.startswith("missingindicator_"):
            name = name.replace("missingindicator_", "")
            if name in GROUPED:
                combined += effect
            else:
                rows.append((name + " is missing", effect))
        elif name in GROUPED:
            combined += effect
        elif raw.startswith("cat__"):
            col, _, category = name.partition("_")
            by_column[col] = by_column.get(col, 0.0) + effect
            if value > 0.5:
                chosen[col] = category
        else:
            info = meta["numeric"].get(name, {})
            if info.get("min") == 0 and info.get("max") == 1:
                rows.append((name + (" = yes" if value > mean else " = no"), effect))
            else:
                rows.append((name + (" is above average" if value > mean else " is below average"), effect))
    for col, effect in by_column.items():
        rows.append((col + " = " + chosen.get(col, "other"), effect))
    rows.append(("Tenure and billing amounts (combined)", combined))

    table = pd.DataFrame(rows, columns=["Factor", "Effect"])
    return table[table["Effect"].abs() > 0.02]


try:
    model, meta = load_artifacts()
except Exception as error:
    st.error("The model files could not be loaded. If you just changed library versions, "
             "re-run the export step and redeploy.")
    st.exception(error)
    st.stop()

base_rate = meta["baseline_churn_rate"]

#sidebar
st.sidebar.header("Business assumptions")
st.sidebar.caption("Illustrative numbers. Change them and watch the recommendation move.")
value_saved = st.sidebar.number_input("Value of a saved customer ($)", min_value=1.0,
                                      value=float(meta["business"]["value_saved"]), step=50.0)
cost_contact = st.sidebar.number_input("Cost of one retention offer ($)", min_value=1.0,
                                       value=float(meta["business"]["cost_contact"]), step=10.0)
success_rate = st.sidebar.slider("Share of contacted churners who stay", 0.05, 0.95,
                                 float(meta["business"]["success_rate"]), 0.05)
break_even = cost_contact / (success_rate * value_saved)
st.sidebar.write("Break-even churn probability: **" + f"{break_even:.0%}" + "**")
st.sidebar.caption("Contact a customer only if their churn probability is above this. "
                   "That rule is how the pipeline chose its threshold.")

#page
st.title("Customer Churn Intelligence")
st.caption("Live demo of an end-to-end churn model. Data: IBM Telco sample dataset (a fictional "
           "company). A demonstration, not a tool for real customer decisions.")

tab_score, tab_results, tab_usage, tab_about = st.tabs(["Score a customer", "Model results", "Usage", "About"])

with tab_score:
    preset_name = st.selectbox("Start from an example", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    if st.session_state.setdefault("last_preset", preset_name) != preset_name:
        usage.log_event("example_selected", {"example": preset_name})
        st.session_state["last_preset"] = preset_name
    prefix = str(list(PRESETS.keys()).index(preset_name)) + "_"

    left, right = st.columns([2, 1])
    values = {}
    with left:
        c1, c2, c3 = st.columns(3)

        c1.subheader("Customer")
        senior = c1.selectbox("Senior citizen", ["No", "Yes"],
                              index=1 if preset.get("SeniorCitizen", 0) == 1 else 0,
                              key=prefix + "senior")
        values["SeniorCitizen"] = 1 if senior == "Yes" else 0
        values["Partner"] = choose(meta, "Partner", "Has a partner", preset, prefix, container=c1)
        values["Dependents"] = choose(meta, "Dependents", "Has dependents", preset, prefix, container=c1)
        t_info = meta["numeric"]["tenure"]
        tenure_min = max(1, int(t_info["min"]))
        tenure_start = int(min(max(preset.get("tenure", t_info["default"]), tenure_min), t_info["max"]))
        tenure = c1.slider("Tenure (months)", tenure_min, int(t_info["max"]), tenure_start,
                           key=prefix + "tenure")
        values["tenure"] = float(tenure)

        c2.subheader("Services")
        values["PhoneService"] = choose(meta, "PhoneService", "Phone service", preset, prefix, container=c2)
        if values["PhoneService"] == "No":
            values["MultipleLines"] = "No phone service"
        else:
            values["MultipleLines"] = choose(meta, "MultipleLines", "Multiple lines", preset, prefix,
                                             drop=("No phone service",), container=c2)
        values["InternetService"] = choose(meta, "InternetService", "Internet service", preset, prefix,
                                           container=c2)
        addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
        for col in addons:
            if values["InternetService"] == "No":
                values[col] = "No internet service"
            else:
                values[col] = choose(meta, col, col, preset, prefix,
                                     drop=("No internet service",), container=c2)

        c3.subheader("Contract and billing")
        values["Contract"] = choose(meta, "Contract", "Contract", preset, prefix, container=c3)
        values["PaperlessBilling"] = choose(meta, "PaperlessBilling", "Paperless billing", preset, prefix,
                                            container=c3)
        values["PaymentMethod"] = choose(meta, "PaymentMethod", "Payment method", preset, prefix,
                                         container=c3)
        m_info = meta["numeric"]["MonthlyCharges"]
        m_start = float(min(max(preset.get("MonthlyCharges", m_info["default"]), m_info["min"]), m_info["max"]))
        monthly = c3.number_input("Monthly charges ($)", min_value=m_info["min"], max_value=m_info["max"],
                                  value=m_start, step=0.5, key=prefix + "monthly")
        values["MonthlyCharges"] = float(monthly)
        # total charges is roughly tenure x monthly charge
        values["TotalCharges"] = float(tenure * monthly)
        c3.caption("Total charges is estimated as tenure x monthly charges.")

    # "engaged" = the visitor changed something compared with what the page first showed
    signature = json.dumps(values, sort_keys=True, default=str)
    if "first_signature" not in st.session_state:
        st.session_state["first_signature"] = signature
    elif signature != st.session_state["first_signature"] and not st.session_state.get("engaged_logged"):
        st.session_state["engaged_logged"] = True
        usage.log_event("engaged")

    row = pd.DataFrame([values])[meta["feature_columns"]]
    prob = float(model.predict_proba(row)[0, 1])
    gain = prob * success_rate * value_saved - cost_contact

    with right:
        st.subheader("Result")
        st.metric("Churn probability", f"{prob:.1%}",
                  delta=f"{(prob - base_rate) * 100:+.1f} points vs the average customer",
                  delta_color="inverse")
        st.progress(min(max(prob, 0.0), 1.0))
        if gain > 0:
            st.success(f"Recommendation: contact this customer. Expected gain about ${gain:,.0f} per offer.")
        else:
            st.info(f"Recommendation: do not contact. Expected loss about ${-gain:,.0f} per offer.")
        if meta.get("threshold"):
            st.caption(f"With the default assumptions, the pipeline's data-driven threshold is "
                       f"{meta['threshold']}.")

    if meta["explainable"]:
        st.subheader("What moves this score?")
        st.caption("Effect on the log-odds of churn compared with an average customer. "
                   "Positive raises the risk, negative lowers it.")
        effects = explain(model, meta, row)
        up = effects[effects["Effect"] > 0].sort_values("Effect", ascending=False).head(5)
        down = effects[effects["Effect"] < 0].sort_values("Effect").head(5)
        e1, e2 = st.columns(2)
        e1.markdown("**Pushing the risk up**")
        e1.dataframe(up.round(2), hide_index=True)
        e2.markdown("**Pushing the risk down**")
        e2.dataframe(down.round(2), hide_index=True)

    st.divider()
    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("**Was this demo useful?**")
        rating = st.radio("Rating from 1 (not useful) to 5 (very useful)", [1, 2, 3, 4, 5],
                          index=None, horizontal=True)
        comment = st.text_area("Optional comment (please do not enter personal information)", max_chars=300)
        sent = st.form_submit_button("Send feedback")
    if sent:
        if rating is None:
            st.warning("Please pick a rating from 1 to 5 first.")
        else:
            usage.log_event("feedback", {"rating": rating, "comment": comment.strip()})
            st.success("Thank you for the feedback!")

with tab_results:
    st.subheader("How well does the model work? (held-out test set, evaluated once)")
    res_path = ROOT / "reports" / "results_telco.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        t = res["test"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test ROC-AUC", f"{t['roc_auc']:.3f}",
                  help=f"95% range {t['roc_auc_low']:.3f} to {t['roc_auc_high']:.3f}. 0.5 is a coin flip.")
        m2.metric("Riskiest 20% reaches", f"{t['top20_share_of_churners_pct']:.0f}% of churners",
                  help=f"Lift {t['top20_lift']:.2f}x versus contacting customers at random.")
        m3.metric("Profit per customer", f"${t['profit_model']:+.1f}",
                  help=f"Versus ${t['profit_everyone']:+.1f} when contacting everyone (illustrative assumptions).")
        m4.metric("Model chosen", res["primary_model"])
    figure = ROOT / "reports" / "figures" / "step7_test_curves.png"
    if figure.exists():
        st.image(str(figure), caption="ROC curve, precision-recall curve and calibration on the test set")
    comparison = ROOT / "reports" / "CROSS_DATASET_COMPARISON.md"
    if comparison.exists():
        st.subheader("The same pipeline on other datasets")
        st.markdown(comparison.read_text(encoding="utf-8"))

with tab_usage:
    st.subheader("Live usage of this demo")
    stats = usage.load_stats()
    if stats is None:
        st.info("Usage tracking is not set up in this deployment.")
    else:
        o = stats["overview"].iloc[0]
        if int(o["visits"]) == 0:
            st.info("No visits recorded yet.")
        else:
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Visits", int(o["visits"]), help="One visit = one browser session.")
            u2.metric("Engaged visits", int(o["engaged"]),
                      help="Visitors who changed something after the page first loaded.")
            u3.metric("Examples opened", int(o["example_clicks"]))
            fb = stats["feedback"].iloc[0]
            n_fb = int(fb["responses"])
            u4.metric("Feedback", f"{n_fb} response" + ("" if n_fb == 1 else "s"),
                      help="Average rating " + (f"{float(fb['avg_rating']):.1f} / 5" if int(fb["responses"]) else "n/a"))
            if len(stats["daily"]):
                st.markdown("**Last 30 days**")
                st.bar_chart(stats["daily"].set_index("day")[["visits", "engaged"]])
            if len(stats["examples"]):
                st.markdown("**Examples people opened**")
                st.dataframe(stats["examples"], hide_index=True)
            st.caption("Counting since " + str(o["since"])[:10] + ". A visit is one browser session "
                       "(reloading the page starts a new one), so this counts visits, not unique people. "
                       "The owner's own visits are excluded. Mostly friends and classmates so far.")

with tab_about:
    st.markdown(f"""
**What this is.** A churn model from a config-driven pipeline that ran unchanged on three
datasets. This demo serves the logistic regression model it chose for the IBM Telco sample data.

**What makes it trustworthy.** A locked test set opened once, preprocessing learned inside
cross-validation folds, thresholds chosen on out-of-fold predictions, honest probabilities, and
automatic data-quality gates that stop the run on leak-like patterns.

**Limits.** One snapshot of fictional sample data; the business numbers are illustrative; the model
shows who is likely to leave, not what would make them stay (that needs an A/B test).
Explanations are for a linear model and describe associations, not causes. The tenure slider starts at
1 month: the 11 customers with 0 months in the sample had not been billed yet, and none of them left,
so the model would wrongly read "no billing history" as "safe".

**Privacy.** The page counts anonymous visits and clicks to show real usage. It stores a random
session id, the time, which example was opened and optional feedback. It does not store IP addresses,
names or emails, and the customers shown here are made-up examples.

**Code, reports and tests:** [Churn_model on GitHub](https://github.com/itskumarmukesh/Churn_model)

""")