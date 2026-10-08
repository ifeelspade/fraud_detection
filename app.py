from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fraud Detection | End-to-End ML",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Fraud Detection — End-to-End Machine Learning")
st.caption(
    "Interactive presentation dashboard: Raw Data → Cleaning → "
    "Feature Engineering → Model Selection → Training → Evaluation → Fraud Detection"
    "Utkarsh Kulshrestha, Vinay Supekar, Siddhant Chiring, Shashwat Aayush"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Presentation Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "1. Executive Summary",
        "2. Raw Data",
        "3. Data Cleaning",
        "4. Feature Engineering",
        "5. Model Selection",
        "6. Train the Model",
        "7. Model Results",
        "8. How We Cure Bias",
        "9. Live Fraud Prediction",
        "10. Cleaned Data"
    ]
)

# ============================================================
# LOAD DATA
# ============================================================

FILE_PATH = "historical_data.csv"

@st.cache_data
def load_raw_data():
    return pd.read_csv(FILE_PATH)

# ============================================================
# CLEANING FUNCTION
# ============================================================

CHOSEN_COLUMNS = [
    "ClaimCause",
    "ClaimInvolvedCovers",
    "FirstPolicySubscriptionDate",
    "LossDate",
    "ConnectionBetweenParties",
    "FirstPartyVehicleType",
    "EasinessToStage",
    "ClaimAmount",
    "ClaimWihoutIdentifiedThirdParty",
    "LossHour",
    "FpVehicleAgeMonths",
    "NumberOfBodilyInjuries",
    "PolicyWasSubscribedOnInternet",
    "Fraud"
]


@st.cache_data
def clean_data(df_raw):

    df = df_raw[CHOSEN_COLUMNS].copy()

    # Duplicate check on full raw data
    duplicate_count = df_raw.duplicated().sum()
    df = df[~df_raw.duplicated().values].reset_index(drop=True)

    # Dates
    df["FirstPolicySubscriptionDate"] = pd.to_datetime(
        df["FirstPolicySubscriptionDate"], errors="coerce"
    )
    df["LossDate"] = pd.to_datetime(
        df["LossDate"], errors="coerce"
    )

    invalid_dates = (
        df["LossDate"] < df["FirstPolicySubscriptionDate"]
    ).sum()

    # Missing categorical values
    df["ConnectionBetweenParties"] = (
        df["ConnectionBetweenParties"].fillna("NoConnection")
    )

    df["ClaimCause"] = (
        df["ClaimCause"].fillna("Unknown")
    )

    df["ClaimInvolvedCovers"] = (
        df["ClaimInvolvedCovers"].fillna("NoCoverInfo")
    )

    vehicle_mode = df["FirstPartyVehicleType"].mode()[0]

    df["FirstPartyVehicleType"] = (
        df["FirstPartyVehicleType"].fillna(vehicle_mode)
    )

    # Category consistency
    df["FirstPartyVehicleType"] = (
        df["FirstPartyVehicleType"]
        .replace({"PrivateCar": "Car"})
    )

    cause_counts = df["ClaimCause"].value_counts()

    rare_causes = cause_counts[
        cause_counts < 50
    ].index

    df["ClaimCause"] = (
        df["ClaimCause"]
        .replace(rare_causes, "Other")
    )

    df["ConnectionBetweenParties"] = (
        df["ConnectionBetweenParties"].where(
            df["ConnectionBetweenParties"] == "NoConnection",
            "Connected"
        )
    )

    for col in [
        "ClaimCause",
        "ClaimInvolvedCovers",
        "ConnectionBetweenParties",
        "FirstPartyVehicleType"
    ]:
        df[col] = (
            df[col].astype(str).str.strip()
        )

    # Numeric corrections
    negative_age_count = (
        df["FpVehicleAgeMonths"] < 0
    ).sum()

    df.loc[
        df["FpVehicleAgeMonths"] < 0,
        "FpVehicleAgeMonths"
    ] = np.nan

    negative_amount_count = (
        df["ClaimAmount"] < 0
    ).sum()

    df.loc[
        df["ClaimAmount"] < 0,
        "ClaimAmount"
    ] = 0

    # Binary validation
    for col in [
        "PolicyWasSubscribedOnInternet",
        "ClaimWihoutIdentifiedThirdParty",
        "Fraud"
    ]:
        df[col] = df[col].astype(int)

    # LossHour = 0 treated as unknown
    zero_hour_count = (
        df["LossHour"] == 0
    ).sum()

    df["LossHour_Unknown"] = (
        df["LossHour"].isna()
        | (df["LossHour"] == 0)
    ).astype(int)

    df.loc[
        df["LossHour"] == 0,
        "LossHour"
    ] = np.nan

    return (
        df,
        {
            "duplicate_count": int(duplicate_count),
            "invalid_dates": int(invalid_dates),
            "negative_age_count": int(negative_age_count),
            "negative_amount_count": int(negative_amount_count),
            "zero_hour_count": int(zero_hour_count)
        }
    )


# ============================================================
# FEATURE ENGINEERING + MODEL
# ============================================================

NIGHT_HOURS = [22, 23, 1, 2, 3, 4, 5]

# Variables that can act as proxies for customer demographics
# (income level, age group, urban / digital profile).
# They are REMOVED from the final model (see page 8).
PROXY_FEATURES = [
    "FpVehicleAgeMonths",
    "PolicyWasSubscribedOnInternet"
]

NUM_COLS = [
    "PolicyTenureDays",
    "ClaimAmount_log",
    "NightLoss",
    "FpVehicleAgeMonths",
    "EasinessToStage",
    "NumberOfBodilyInjuries",
    "ClaimWihoutIdentifiedThirdParty",
    "PolicyWasSubscribedOnInternet"
]

CAT_COLS = [
    "ClaimCause",
    "ConnectionBetweenParties",
    "FirstPartyVehicleType"
]

# Customer segments used for the fairness audit
AUDIT_COLS = [
    "Channel",
    "Vehicle Age Band",
    "Claim Amount Band",
    "Vehicle Type",
    "Connection",
    "Loss Time"
]

MIN_GROUP = 30   # ignore very small segments in the fairness scorecard


def build_features(data, claim_cap):
    """
    ONE function used for both training and live prediction,
    so the simulator can never differ from the trained pipeline.
    """

    data = data.copy()

    data["PolicyTenureDays"] = (
        data["LossDate"]
        - data["FirstPolicySubscriptionDate"]
    ).dt.days

    # Bias cure: cap extreme claim amounts at the 99th percentile
    data["ClaimAmount_log"] = np.log1p(
        data["ClaimAmount"].clip(upper=claim_cap)
    )

    data["NightLoss"] = (
        data["LossHour"]
        .isin(NIGHT_HOURS)
        .astype(int)
    )

    cover_flags = (
        data["ClaimInvolvedCovers"]
        .str.get_dummies(sep=" ")
        .add_prefix("Cover_")
    )

    cat_flags = pd.get_dummies(
        data[CAT_COLS],
        dtype=int
    )

    return pd.concat(
        [
            data[NUM_COLS],
            cat_flags,
            cover_flags
        ],
        axis=1
    )


@st.cache_data
def prepare_features(df, claim_cap):

    X = build_features(df, claim_cap)
    y = df["Fraud"]

    return X, y


def make_segments(df):
    """Customer segments used ONLY for bias auditing / reweighing."""

    seg = pd.DataFrame(index=df.index)

    seg["Channel"] = np.where(
        df["PolicyWasSubscribedOnInternet"] == 1,
        "Online",
        "Offline"
    )

    age = df["FpVehicleAgeMonths"]

    seg["Vehicle Age Band"] = np.where(
        age.isna(),
        "Unknown",
        np.where(age <= age.median(), "Newer", "Older")
    )

    seg["Claim Amount Band"] = pd.qcut(
        df["ClaimAmount"],
        4,
        duplicates="drop"
    ).astype(str)

    seg["Vehicle Type"] = df["FirstPartyVehicleType"]
    seg["Connection"] = df["ConnectionBetweenParties"]

    seg["Loss Time"] = np.where(
        df["LossHour"].isna(),
        "Unknown",
        np.where(df["LossHour"].isin(NIGHT_HOURS), "Night", "Day")
    )

    return seg


def reweighing_weights(groups, y):
    """
    Reweighing (Kamiran & Calders): weight each training row so that
    the fraud label becomes statistically independent of the customer
    segment. Weights are clipped to avoid instability in tiny segments.
    """

    d = pd.DataFrame({
        "g": np.asarray(groups),
        "y": np.asarray(y)
    })

    n = len(d)

    p_g = d["g"].value_counts(normalize=True)
    p_y = d["y"].value_counts(normalize=True)
    p_gy = d.groupby(["g", "y"])["y"].transform("size") / n

    w = (
        d["g"].map(p_g).values
        * d["y"].map(p_y).values
        / p_gy.values
    )

    return np.clip(w, 0.25, 5.0)


@st.cache_resource
def train_model(X, y, drop_cols=(), groups=None):

    X_used = X.drop(columns=list(drop_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X_used,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    model = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="median")
        ),
        (
            "scale",
            StandardScaler()
        ),
        (
            "logreg",
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000
            )
        )
    ])

    grid = GridSearchCV(
        model,
        param_grid={
            "logreg__C": [
                0.01,
                0.03,
                0.1,
                0.3,
                1
            ]
        },
        cv=StratifiedKFold(
            5,
            shuffle=True,
            random_state=42
        ),
        scoring="average_precision"
    )

    fit_params = {}

    if groups is not None:
        fit_params["logreg__sample_weight"] = reweighing_weights(
            groups.loc[X_train.index],
            y_train
        )

    grid.fit(X_train, y_train, **fit_params)

    best_model = grid.best_estimator_

    scores = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        scores
    )

    flag_share = 0.05

    cutoff = np.quantile(
        scores,
        1 - flag_share
    )

    flagged = scores >= cutoff

    y_pred = flagged.astype(int)

    n_analysed = len(y_test)
    n_flagged = int(flagged.sum())

    true_fraud_flagged = int(
        (
            flagged
            &
            (y_test.values == 1)
        ).sum()
    )

    fraud_total = int(y_test.sum())

    detection_rate = (
        n_flagged / n_analysed
    )

    hit_rate = (
        true_fraud_flagged / n_flagged
        if n_flagged else 0
    )

    fraud_capture_rate = (
        true_fraud_flagged / fraud_total
        if fraud_total else 0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    coefficients = pd.Series(
        best_model
        .named_steps["logreg"]
        .coef_[0],
        index=X_used.columns
    )

    return {
        "model": best_model,
        "feature_cols": list(X_used.columns),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scores": scores,
        "flagged": flagged,
        "roc_auc": roc_auc,
        "cutoff": cutoff,
        "n_analysed": n_analysed,
        "n_flagged": n_flagged,
        "true_fraud_flagged": true_fraud_flagged,
        "fraud_total": fraud_total,
        "detection_rate": detection_rate,
        "hit_rate": hit_rate,
        "fraud_capture_rate": fraud_capture_rate,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "coefficients": coefficients,
        "best_c": grid.best_params_["logreg__C"]
    }


# ============================================================
# FAIRNESS AUDIT HELPERS
# ============================================================

def build_audit_frame(df, res):

    audit = make_segments(df).loc[res["X_test"].index].copy()
    audit["flagged"] = np.asarray(res["flagged"], dtype=bool)
    audit["Fraud"] = res["y_test"].values

    return audit


def audit_group(audit, col):

    overall_flag_rate = audit["flagged"].mean()

    rows = {}

    for name, d in audit.groupby(col):

        fraud = d["Fraud"] == 1

        rows[name] = {
            "Claims": len(d),
            "Actual Fraud Rate": d["Fraud"].mean(),
            "Flag Rate": d["flagged"].mean(),
            "Precision": (
                d.loc[d["flagged"], "Fraud"].mean()
                if d["flagged"].any() else np.nan
            ),
            "Recall": (
                d.loc[fraud, "flagged"].mean()
                if fraud.any() else np.nan
            ),
            "False Positive Rate": (
                d.loc[~fraud, "flagged"].mean()
                if (~fraud).any() else np.nan
            ),
            "Flag Rate vs Overall": (
                d["flagged"].mean() / overall_flag_rate
                if overall_flag_rate > 0 else np.nan
            )
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def fairness_summary(audit):

    rows = []

    for col in AUDIT_COLS:

        stats = audit_group(audit, col)
        stats = stats[stats["Claims"] >= MIN_GROUP]

        if len(stats) < 2:
            continue

        flag_rate = stats["Flag Rate"]

        ratio = (
            flag_rate.min() / flag_rate.max()
            if flag_rate.max() > 0 else np.nan
        )

        fpr_gap = (
            stats["False Positive Rate"].max()
            - stats["False Positive Rate"].min()
        ) * 100

        rows.append({
            "Segment View": col,
            "Flag-Rate Ratio (min/max)": ratio,
            "FPR Gap (pp)": fpr_gap
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Segment View",
            "Flag-Rate Ratio (min/max)",
            "FPR Gap (pp)"
        ]
    )


# ============================================================
# RUN PIPELINE
# ============================================================

try:
    df_raw = load_raw_data()
    df_clean, cleaning_info = clean_data(df_raw)

    # 99th percentile cap for claim amount (bias cure + stable simulation)
    CLAIM_CAP = float(df_clean["ClaimAmount"].quantile(0.99))

    X, y = prepare_features(df_clean, CLAIM_CAP)

    # Segments used for reweighing and fairness audit
    segments = make_segments(df_clean)
    fairness_groups = segments["Channel"] + " | " + segments["Vehicle Age Band"]

    # BEFORE: baseline model (all features, no reweighing)
    results_base = train_model(X, y, (), None)

    # AFTER: final bias-mitigated model (proxies removed + reweighing)
    results = train_model(X, y, tuple(PROXY_FEATURES), fairness_groups)

except FileNotFoundError:
    st.error(
        "historical_data.xlsx was not found. "
        "Place the Excel file in the same folder as app.py."
    )
    st.stop()


# ============================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ============================================================

if page == "1. Executive Summary":

    st.header("1️⃣ Executive Summary")

    st.markdown("""
    ### Project Objective

    Build a machine-learning based fraud detection system that can:

    - identify potentially fraudulent claims
    - reduce manual investigation effort
    - handle severe class imbalance
    - measure and reduce bias (proxy removal, reweighing, fairness audit)
    - provide an interpretable risk score
    """)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Raw Claims",
        f"{len(df_raw):,}"
    )

    c2.metric(
        "Fraud Cases",
        f"{df_clean['Fraud'].sum():,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{df_clean['Fraud'].mean():.2%}"
    )

    c4.metric(
        "ROC-AUC",
        f"{results['roc_auc']:.3f}"
    )

    st.divider()

    st.subheader("End-to-End Pipeline")

    st.markdown("""
    **Raw Excel Data**
    ↓  
    **Select relevant variables**
    ↓  
    **Clean missing/inconsistent/invalid values**
    ↓  
    **Feature engineering**
    ↓  
    **Train/Test Split with stratification**
    ↓  
    **Bias mitigation: proxy removal + reweighing + claim cap**
    ↓  
    **Median Imputation + StandardScaler**
    ↓  
    **L1 Logistic Regression**
    ↓  
    **5-Fold Cross Validation + GridSearchCV**
    ↓  
    **Risk scoring**
    ↓  
    **Flag top 5% highest-risk claims**
    ↓  
    **Fairness audit across customer segments**
    """)

    st.info(
        "Presentation point: We are not simply predicting every claim as fraud. "
        "Because fraud is rare, the model produces risk scores and investigators "
        "can review the highest-risk claims first."
    )


# ============================================================
# PAGE 2 — RAW DATA
# ============================================================

elif page == "2. Raw Data":

    st.header("2️⃣ Raw Data — Before Cleaning")

    st.write(
        "This is the original dataset before applying the cleaning rules."
    )

    st.write(
        f"Shape: **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**"
    )

    st.dataframe(
        df_raw.head(100),
        use_container_width=True
    )

    st.subheader("Original Missing Values")

    missing = (
        df_raw.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# ============================================================
# PAGE 3 — CLEANING
# ============================================================

elif page == "3. Data Cleaning":

    st.header("3️⃣ Data Cleaning")

    st.subheader("Why did we clean the data?")

    st.write(
        "The objective was to make the dataset consistent and usable "
        "without blindly deleting observations."
    )

    cleaning_steps = pd.DataFrame({
        "Problem": [
            "Unnecessary variables",
            "Missing ConnectionBetweenParties",
            "Missing ClaimCause",
            "Missing ClaimInvolvedCovers",
            "Missing Vehicle Type",
            "Inconsistent vehicle categories",
            "Rare claim causes",
            "Rare connection categories",
            "Negative vehicle age",
            "Negative claim amount",
            "LossHour = 0",
            "Date validation",
            "Binary variable validation"
        ],
        "Treatment": [
            "Keep only selected variables",
            "Replace with NoConnection",
            "Replace with Unknown",
            "Replace with NoCoverInfo",
            "Use mode",
            "PrivateCar → Car",
            "Categories with <50 rows → Other",
            "Group as Connected",
            "Convert to missing",
            "Replace with 0",
            "Treat as unknown + create flag",
            "Check loss date >= policy start",
            "Validate values are 0/1"
        ]
    })

    st.dataframe(
        cleaning_steps,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Actual Cleaning Statistics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Duplicates",
        cleaning_info["duplicate_count"]
    )

    c2.metric(
        "Negative Ages",
        cleaning_info["negative_age_count"]
    )

    c3.metric(
        "Negative Amounts",
        cleaning_info["negative_amount_count"]
    )

    c4.metric(
        "LossHour = 0",
        cleaning_info["zero_hour_count"]
    )

    c5.metric(
        "Invalid Dates",
        cleaning_info["invalid_dates"]
    )

    st.subheader("Missing Values After Cleaning")

    remaining_missing = (
        df_clean.isna()
        .sum()
        .sort_values(ascending=False)
    )

    st.dataframe(
        remaining_missing[
            remaining_missing > 0
        ].rename("Missing Values"),
        use_container_width=True
    )

    st.info(
        "Important: LossHour and FpVehicleAgeMonths are not forcibly filled here. "
        "Their median imputation happens later inside the ML pipeline using the "
        "training data only. This prevents data leakage."
    )


# ============================================================
# PAGE 4 — FEATURE ENGINEERING
# ============================================================

elif page == "4. Feature Engineering":

    st.header("4️⃣ Feature Engineering")

    st.markdown("""
    The cleaned variables are transformed into numerical features that
    the machine-learning algorithm can understand.
    """)

    feature_table = pd.DataFrame({
        "Technique": [
            "Policy Tenure",
            "Log Transformation",
            "Night Loss",
            "One-Hot Encoding",
            "Multi-label Cover Encoding",
            "Target Variable"
        ],
        "What We Did": [
            "LossDate − FirstPolicySubscriptionDate",
            "log1p(ClaimAmount)",
            "Flag claims occurring between 10 PM and 5 AM",
            "Convert categorical variables into 0/1 columns",
            "Create one binary feature for each involved cover",
            "Fraud = 1, Non-Fraud = 0"
        ],
        "Reason": [
            "Capture how long the policy existed before the loss",
            "Reduce the effect of large claim amounts",
            "Capture possible time-related patterns",
            "Machine learning requires numerical inputs",
            "Represent multiple covers mathematically",
            "Define the prediction objective"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Selected Variables",
        len(CHOSEN_COLUMNS)
    )

    c2.metric(
        "Final Model Features",
        len(results["feature_cols"])
    )

    c3.metric(
        "Fraud Target",
        "Binary 0 / 1"
    )

    st.subheader("Example Engineered Data")

    engineered_preview = X.copy()

    st.dataframe(
        engineered_preview.head(20),
        use_container_width=True
    )


# ============================================================
# PAGE 5 — MODEL SELECTION
# ============================================================

elif page == "5. Model Selection":

    st.header("5️⃣ Why Logistic Regression?")

    st.markdown("""
    ### Selected model: Logistic Regression with L1 Regularization

    We selected Logistic Regression because the objective is not only
    prediction, but also **interpretability**.

    Each feature receives a coefficient that helps explain whether that
    feature pushes the prediction toward or away from fraud.
    """)

    model_reason = pd.DataFrame({
        "Technique": [
            "Logistic Regression",
            "L1 Regularization",
            "class_weight='balanced'",
            "StandardScaler",
            "Median Imputation",
            "GridSearchCV",
            "Stratified 5-Fold CV"
        ],
        "Why Used": [
            "Interpretable binary classification",
            "Can shrink weak coefficients to zero",
            "Handles rare fraud class better",
            "Puts numerical features on comparable scale",
            "Handles missing numeric values",
            "Finds the best regularization strength",
            "Maintains fraud/non-fraud ratio during validation"
        ]
    })

    st.dataframe(
        model_reason,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Hyperparameter Selection")

    st.write(
        "We tested C values: **0.01, 0.03, 0.1, 0.3 and 1** "
        "using 5-fold stratified cross-validation."
    )

    st.success(
        f"Best C selected by GridSearchCV: {results['best_c']}"
    )

    st.subheader("Why not rely only on Accuracy?")

    st.warning(
        f"Only {df_clean['Fraud'].mean():.2%} of the complete dataset is fraud. "
        "A model predicting almost everything as non-fraud could still appear "
        "accurate. Therefore we also use ROC-AUC, fraud capture rate and hit rate."
    )


# ============================================================
# PAGE 6 — TRAIN MODEL
# ============================================================

elif page == "6. Train the Model":

    st.header("6️⃣ Model Training")

    st.markdown("""
    ### What happens when we run the model?

    **Step 1 — Train/Test Split**

    70% training and 30% testing, with `stratify=y`.

    **Step 2 — Imputation**

    Missing numerical values are replaced with the training-set median.

    **Step 3 — Scaling**

    Numerical features are standardized using StandardScaler.

    **Step 4 — Logistic Regression**

    L1-regularized Logistic Regression is trained.

    **Step 5 — Hyperparameter Tuning**

    GridSearchCV evaluates multiple C values using 5-fold stratified CV.

    **Step 6 — Prediction**

    The trained model produces a fraud probability for each test claim.

    **Step 7 — Risk Threshold**

    The top 5% highest-risk claims are flagged for investigation.
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Training Rows",
        f"{len(results['X_train']):,}"
    )

    c2.metric(
        "Testing Rows",
        f"{len(results['X_test']):,}"
    )

    c3.metric(
        "Best C",
        results["best_c"]
    )

    st.subheader("Fraud Distribution")

    distribution = pd.DataFrame({
        "Dataset": [
            "Training",
            "Testing"
        ],
        "Fraud": [
            results["y_train"].sum(),
            results["y_test"].sum()
        ],
        "Non-Fraud": [
            (results["y_train"] == 0).sum(),
            (results["y_test"] == 0).sum()
        ]
    })

    st.dataframe(
        distribution,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 7 — RESULTS
# ============================================================

elif page == "7. Model Results":

    st.header("7️⃣ Model Results")

    st.caption("Final model = bias-mitigated version (proxies removed, reweighed, claim amount capped).")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "ROC-AUC",
        f"{results['roc_auc']:.3f}"
    )

    c2.metric(
        "Accuracy",
        f"{results['accuracy']:.2%}"
    )

    c3.metric(
        "Claims Flagged",
        f"{results['n_flagged']:,}"
    )

    c4.metric(
        "True Frauds Flagged",
        f"{results['true_fraud_flagged']:,}"
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Detection / Flag Rate",
        f"{results['detection_rate']:.2%}"
    )

    c2.metric(
        "Hit Rate",
        f"{results['hit_rate']:.2%}"
    )

    c3.metric(
        "Fraud Capture Rate",
        f"{results['fraud_capture_rate']:.2%}"
    )

    st.subheader("Confusion Matrix")

    cm = results["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Non-Fraud", "Actual Fraud"],
        columns=["Predicted Non-Fraud", "Predicted Fraud"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.subheader("What does this mean?")

    st.write(
        f"""
        We evaluated {results['n_analysed']:,} unseen test claims.

        Instead of investigating all claims, we flagged the highest-risk
        **5% ({results['n_flagged']:,} claims)**.

        Within those flagged claims, **{results['true_fraud_flagged']:,} were actual
        fraud cases**, meaning the model captured
        **{results['fraud_capture_rate']:.0%} of the fraud cases in the test set**.
        """
    )

    st.subheader("Most Important Model Coefficients")

    coefficients = (
        results["coefficients"]
        [results["coefficients"] != 0]
        .sort_values(key=abs, ascending=False)
        .head(15)
    )

    coef_df = pd.DataFrame({
        "Feature": coefficients.index,
        "Coefficient": coefficients.values,
        "Direction": np.where(
            coefficients.values > 0,
            "Pushes toward fraud",
            "Pushes away from fraud"
        )
    })

    st.dataframe(
        coef_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        coef_df.sort_values("Coefficient"),
        x="Coefficient",
        y="Feature",
        color="Direction",
        orientation="h",
        title="Top Non-Zero Logistic Regression Coefficients"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 8 — HOW WE CURE BIAS
# ============================================================

elif page == "8. How We Cure Bias":

    st.header("8️⃣ How We Cure Bias")

    st.markdown(
        """
        A fraud model can be **accurate and still unfair**. For an insurer, an unfair
        model means honest customers from certain segments are investigated more
        often — leading to churn, complaints and regulatory risk.

        Our dataset has **no direct protected attributes** (no gender, age, religion
        or caste), but some variables can act as **proxies** for them. We therefore
        (1) removed risk at the source, (2) corrected the training process, and
        (3) **measured** the result with a fairness audit.
        """
    )

    st.subheader("Where bias can enter")

    st.markdown(
        """
        - **Proxy bias** — vehicle age can stand in for income, online policy for age / urban profile, night loss for shift workers.
        - **Label bias** — `Fraud = 1` comes from past investigations, so the model can copy past investigators' habits.
        - **Sampling bias** — some segments have very few fraud cases, so their scores are unreliable.
        - **Extrapolation** — extreme claim amounts push scores to arbitrary extremes.
        """
    )

    st.subheader("Cures applied in this project")

    cure_table = pd.DataFrame({
        "#": [1, 2, 3, 4, 5, 6, 7],
        "Bias Risk": [
            "Direct use of protected attributes",
            "Proxy variables (income / age / digital profile)",
            "Fraud rate differs by segment in the training data",
            "Extreme claim amounts get arbitrary scores",
            "Automatic decisions wrongly accuse honest customers",
            "Unfairness stays invisible if nobody measures it",
            "The model is wrong for an individual (e.g. a high-premium, loyal customer treated as a suspect)"
        ],
        "Cure Applied": [
            "Only 13 claim-related columns are selected; gender, age, religion, caste etc. are never used",
            "Vehicle Age and Online-Policy flag are REMOVED from the model features",
            "Reweighing (Kamiran & Calders): training rows are weighted so fraud is independent of Channel × Vehicle-Age segment",
            "Claim amount is capped at the 99th percentile in training AND in the live simulator",
            "Score is used only to prioritise the top-5% queue for human investigators; percentile is shown, not a verdict",
            "Fairness audit across 6 segment views, Before vs After (below)",
            "Case-by-case human-in-the-loop review: tiered routing by risk AND customer value, review committee for sensitive cases, AI never denies alone (Step 4)"
        ]
    })

    st.dataframe(
        cure_table,
        use_container_width=True,
        hide_index=True
    )

    audit_base = build_audit_frame(df_clean, results_base)
    audit_final = build_audit_frame(df_clean, results)

    # --------------------------------------------------------
    st.subheader("Step 1 — What did fairness cost us in accuracy?")

    def _pp(a, b):
        return f"{(a - b) * 100:+.2f} pp"

    perf = pd.DataFrame({
        "Metric": [
            "ROC-AUC",
            "Hit Rate (precision in top 5%)",
            "Fraud Capture Rate (recall in top 5%)"
        ],
        "Before (baseline)": [
            f"{results_base['roc_auc']:.3f}",
            f"{results_base['hit_rate']:.2%}",
            f"{results_base['fraud_capture_rate']:.2%}"
        ],
        "After (final model)": [
            f"{results['roc_auc']:.3f}",
            f"{results['hit_rate']:.2%}",
            f"{results['fraud_capture_rate']:.2%}"
        ],
        "Change": [
            f"{results['roc_auc'] - results_base['roc_auc']:+.3f}",
            _pp(results["hit_rate"], results_base["hit_rate"]),
            _pp(results["fraud_capture_rate"], results_base["fraud_capture_rate"])
        ]
    })

    st.dataframe(
        perf,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Fairness usually costs a little accuracy. This table shows exactly how much."
    )

    # --------------------------------------------------------
    st.subheader("Step 2 — Fairness scorecard (Before vs After)")

    sum_base = fairness_summary(audit_base)
    sum_final = fairness_summary(audit_final)

    scorecard = sum_base.merge(
        sum_final,
        on="Segment View",
        suffixes=(" — Before", " — After")
    )

    if scorecard.empty:
        st.info("Not enough claims per segment to build the scorecard.")
    else:
        improved = int(
            (
                scorecard["Flag-Rate Ratio (min/max) — After"]
                > scorecard["Flag-Rate Ratio (min/max) — Before"]
            ).sum()
        )

        gap_improved = int(
            (
                scorecard["FPR Gap (pp) — After"]
                < scorecard["FPR Gap (pp) — Before"]
            ).sum()
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Flag-rate parity improved in",
            f"{improved} of {len(scorecard)} segment views"
        )

        c2.metric(
            "False-positive gap reduced in",
            f"{gap_improved} of {len(scorecard)} segment views"
        )

        st.dataframe(
            scorecard.round(3),
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        """
        **How to read it**

        - **Flag-Rate Ratio (min/max)**: 1.0 means every segment is flagged at the same rate.
          Below about 0.8 deserves a review (four-fifths rule).
        - **FPR Gap (pp)**: difference in the share of *honest* customers wrongly flagged
          between the best and worst treated segment. Lower is fairer.
        - If a segment truly has more fraud, a higher flag rate can be justified —
          but a higher **false-positive rate** is not.
        """
    )

    # --------------------------------------------------------
    st.subheader("Step 3 — Segment-level audit")

    audit_col = st.selectbox(
        "Choose a segment view",
        AUDIT_COLS
    )

    fmt = {
        "Claims": "{:,.0f}",
        "Actual Fraud Rate": "{:.2%}",
        "Flag Rate": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "False Positive Rate": "{:.2%}",
        "Flag Rate vs Overall": "{:.2f}"
    }

    tab_after, tab_before = st.tabs(
        ["After (final model)", "Before (baseline)"]
    )

    with tab_after:
        st.dataframe(
            audit_group(audit_final, audit_col).style.format(fmt, na_rep="–"),
            use_container_width=True
        )

    with tab_before:
        st.dataframe(
            audit_group(audit_base, audit_col).style.format(fmt, na_rep="–"),
            use_container_width=True
        )

    st.caption(
        f"Segments with fewer than {MIN_GROUP} claims are shown here "
        "but ignored in the scorecard because their rates are too noisy."
    )

    # --------------------------------------------------------
    st.subheader("Step 4 — Managerial perspective: AI recommends, humans decide")

    st.markdown(
        """
        Fairness is not only a statistics problem — it is a **business-judgement problem**.
        Two mistakes are costly, and they pull in opposite directions:

        - **Paying a fraudulent claim** → direct financial loss.
        - **Wrongly suspecting an honest, high-value customer** → delay, embarrassment,
          complaint, churn and loss of future premium.

        One threshold cannot balance both. So the model **prioritises**, and a
        **case-by-case human review** takes the decision.
        """
    )

    # ---- empirical fraud rate by risk tier (unseen test claims) ----
    tier_scores = pd.Series(
        results["scores"],
        index=results["y_test"].index
    )

    tier_pct = tier_scores.rank(pct=True)

    tier = pd.Series(
        np.where(
            tier_scores >= results["cutoff"],
            "1. High (top-5% queue)",
            np.where(
                tier_pct >= 0.80,
                "2. Elevated (80th-95th pct)",
                "3. Lower (below 80th pct)"
            )
        ),
        index=tier_scores.index
    )

    tier_tbl = (
        pd.DataFrame({
            "Tier": tier,
            "Fraud": results["y_test"]
        })
        .groupby("Tier")
        .agg(
            Claims=("Fraud", "size"),
            Frauds=("Fraud", "sum"),
            FraudRate=("Fraud", "mean")
        )
    )

    tier_tbl["Share of all frauds"] = (
        tier_tbl["Frauds"] / tier_tbl["Frauds"].sum()
    )

    tier_rate = tier_tbl["FraudRate"].to_dict()

    st.markdown("#### 4a. What does each risk tier really contain?")

    st.dataframe(
        tier_tbl.rename(columns={"FraudRate": "Actual Fraud Rate"}).style.format({
            "Claims": "{:,.0f}",
            "Frauds": "{:,.0f}",
            "Actual Fraud Rate": "{:.2%}",
            "Share of all frauds": "{:.1%}"
        }),
        use_container_width=True
    )

    honest_per_100 = (1 - results["hit_rate"]) * 100

    k1, k2 = st.columns(2)

    k1.metric(
        "Honest customers per 100 claims in the top-5% queue",
        f"{honest_per_100:.0f}"
    )

    k2.metric(
        "Frauds that sit OUTSIDE the top-5% queue",
        f"{(1 - results['fraud_capture_rate']):.0%}"
    )

    st.caption(
        "Read this as a manager: most flagged claims can still be honest, and some fraud "
        "is always missed. Hence: never deny on the score alone, and keep a small random "
        "audit on the lower tier."
    )

    # ---- decision matrix ----
    st.markdown("#### 4b. Decision matrix: risk tier × customer value")

    matrix = pd.DataFrame({
        "Risk tier": [
            "Lower",
            "Elevated",
            "High (top-5% queue)"
        ],
        "Standard customer": [
            "Straight-through payment; 2–5% random quality audit",
            "Adjuster checks documents (normal turnaround)",
            "Investigator review; pay, hold or decline only on documented evidence"
        ],
        "High-value / long-tenure customer": [
            "Straight-through payment; 2–5% random quality audit",
            "Senior adjuster, fast-track; relationship manager informed",
            "Review committee (claims, underwriting, relationship manager, compliance) — no decision on the score alone"
        ]
    })

    st.dataframe(
        matrix,
        use_container_width=True,
        hide_index=True
    )

    # ---- workflow ----
    st.markdown("#### 4c. Human-in-the-loop workflow")

    workflow = pd.DataFrame({
        "Stage": [
            "1. Automated scoring",
            "2. Triage",
            "3. Evidence review",
            "4. Review committee",
            "5. Customer communication",
            "6. Feedback loop"
        ],
        "Who": [
            "Model",
            "Claims system",
            "Adjuster / investigator",
            "Claims head, underwriting, relationship manager, compliance",
            "Claims team",
            "Analytics team"
        ],
        "What happens": [
            "Risk score, percentile and top drivers are produced for every claim",
            "Claim is routed by risk tier and customer value (matrix above)",
            "Documents, photos and third-party facts are checked; findings are written down",
            "High-risk + high-value or disputed cases are decided case by case, with written reasons",
            "Any hold or decline is explained neutrally, with an appeal route",
            "Closed-case outcomes and overrides are logged and used to retrain and re-audit the model"
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        **Governance principles**

        - The AI score is **never the sole ground** for denying a claim.
        - Reviewers see the **top drivers** and may **override with a written reason**.
        - Customers get a neutral explanation and a **right to appeal**.
        - Track **override, complaint and churn rates by segment** — this catches bias the model audit cannot see.
        - Confirmed outcomes flow back into training, which reduces label bias over time.
        """
    )

    # ---- simulator ----
    st.markdown("#### 4d. Case review simulator (decision support for the committee)")

    st.caption(
        "Fraud likelihood comes from the tier table above (actual test data). "
        "Premium, churn, cost and detection inputs are manager assumptions — "
        "replace the illustrative defaults with real figures."
    )

    med_claim = float(round(df_clean["ClaimAmount"].median(), -2))

    s1, s2, s3 = st.columns(3)

    with s1:
        case_tier = st.selectbox(
            "Model risk tier",
            list(tier_rate.keys())
        )

        case_claim = st.number_input(
            "Claim amount",
            min_value=0.0,
            value=max(med_claim, 1000.0),
            step=1000.0,
            key="case_claim"
        )

    with s2:
        case_premium = st.number_input(
            "Annual premium paid by the customer",
            min_value=0.0,
            value=max(round(med_claim * 0.2, -2), 500.0),
            step=500.0,
            key="case_premium"
        )

        case_years = st.number_input(
            "Expected remaining relationship (years)",
            min_value=0.0,
            value=5.0,
            step=1.0,
            key="case_years"
        )

    with s3:
        case_churn = st.slider(
            "Chance an honest customer leaves after an investigation (%)",
            0, 100, 30,
            key="case_churn"
        )

        case_cost = st.number_input(
            "Cost of one investigation",
            min_value=0.0,
            value=max(round(med_claim * 0.05, -2), 100.0),
            step=100.0,
            key="case_cost"
        )

    case_detect = st.slider(
        "Chance an investigation catches real fraud (%)",
        0, 100, 80,
        key="case_detect"
    )

    p_fraud = float(tier_rate[case_tier])

    loss_avoided = p_fraud * case_claim * case_detect / 100
    relationship_cost = (
        (1 - p_fraud) * (case_churn / 100) * case_premium * case_years
    )
    net_value = loss_avoided - case_cost - relationship_cost
    lifetime_premium = case_premium * case_years

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("Fraud likelihood (tier)", f"{p_fraud:.2%}")
    r2.metric("Fraud loss avoided", f"{loss_avoided:,.0f}")
    r3.metric(
        "Investigation + relationship cost",
        f"{case_cost + relationship_cost:,.0f}"
    )
    r4.metric("Net value of investigating", f"{net_value:,.0f}")

    if net_value > 0 and lifetime_premium >= case_claim:
        st.warning(
            "⚖️ **Send to review committee.** Investigation pays off, but this customer's "
            "lifetime premium is at least as large as the claim. Handle it as a relationship "
            "case: senior reviewer, fast turnaround, neutral communication, written decision."
        )
    elif net_value > 0:
        st.error(
            "🔍 **Route to investigator.** Expected fraud loss avoided is higher than the "
            "cost of investigating and the risk to the relationship."
        )
    else:
        st.success(
            "✅ **Light-touch check, then pay.** Investigating costs more than it is expected "
            "to save. Pay with a post-payment sample audit unless the documents show a red flag."
        )

    st.caption(
        "Decision support only: the committee has the final say. Rule used: net value > 0 → "
        "investigate; and if lifetime premium ≥ claim amount → committee."
    )

    # ---- KPIs ----
    st.markdown("#### 4e. Balanced scorecard for management")

    kpis = pd.DataFrame({
        "KPI": [
            "Fraud capture rate (top 5%)",
            "Honest customers per 100 reviewed claims",
            "Override rate by segment",
            "Complaints and churn after review",
            "Review turnaround time",
            "Flag-rate ratio and FPR gap"
        ],
        "Protects against": [
            "Fraud loss",
            "Customer friction",
            "Hidden bias (model and reviewers)",
            "Loss of valuable customers",
            "Delay for honest customers",
            "Unequal treatment of segments"
        ],
        "Source": [
            "Model results (page 7)",
            "Model results (this page)",
            "Committee decision log",
            "CRM / claims operations",
            "Claims operations",
            "Fairness audit (Steps 2–3)"
        ]
    })

    st.dataframe(
        kpis,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    st.subheader("Limitations")

    st.warning(
        "1) The Fraud label comes from past investigations, so historical bias can remain.  \n"
        "2) We audit proxy segments, not gender/age/religion directly — the dataset does not contain them.  \n"
        "3) Fairness and accuracy trade off; the business must decide the acceptable balance.  \n"
        "4) The audit must be repeated whenever the model is retrained.  \n"
        "5) Human reviewers have biases too — their decisions must be logged and audited as well."
    )


# ============================================================
# PAGE 9 — LIVE FRAUD PREDICTION
# ============================================================

elif page == "9. Live Fraud Prediction":

    st.header("9️⃣ Live Fraud Prediction")

    st.write(
        "Enter claim information below. The final (bias-mitigated) model calculates "
        "a risk score and shows where the claim ranks among all unseen test claims."
    )

    st.caption(
        "Vehicle Age and Online Policy are shown but disabled — they were removed "
        "from the model as bias proxies (see page 8)."
    )

    st.divider()

    today = date.today()

    default_amount = float(round(df_clean["ClaimAmount"].median(), -2))
    default_age = float(round(df_clean["FpVehicleAgeMonths"].median()))

    easiness_options = sorted(
        df_clean["EasinessToStage"].dropna().unique().tolist()
    )

    cover_options = [
        c.replace("Cover_", "")
        for c in results["feature_cols"]
        if c.startswith("Cover_") and c != "Cover_NoCoverInfo"
    ]

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=default_amount,
            step=100.0
        )

        vehicle_age = st.number_input(
            "Vehicle Age (months)",
            min_value=0.0,
            value=default_age,
            step=1.0,
            disabled=True,
            help="Not used by the model — removed as a bias proxy."
        )

        bodily_injuries = st.number_input(
            "Number of Bodily Injuries",
            min_value=0,
            value=0,
            step=1
        )

    with col2:

        easiness = st.selectbox(
            "Easiness To Stage",
            easiness_options
        )

        loss_hour = st.slider(
            "Loss Hour",
            min_value=1,
            max_value=23,
            value=12
        )

        internet_policy = st.selectbox(
            "Policy Subscribed Online?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No",
            disabled=True,
            help="Not used by the model — removed as a bias proxy."
        )

    with col3:

        third_party = st.selectbox(
            "Claim Without Identified Third Party?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        connection = st.selectbox(
            "Connection Between Parties",
            ["NoConnection", "Connected"]
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            sorted(
                df_clean["FirstPartyVehicleType"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    claim_cause = st.selectbox(
        "Claim Cause",
        sorted(
            df_clean["ClaimCause"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    covers = st.multiselect(
        "Claim Involved Covers",
        cover_options,
        default=[
            c for c in ["MaterialDamages"]
            if c in cover_options
        ]
    )

    policy_start = st.date_input(
        "First Policy Subscription Date",
        value=today - timedelta(days=730),
        min_value=date(2000, 1, 1),
        max_value=today
    )

    loss_date = st.date_input(
        "Loss Date",
        value=today,
        min_value=date(2000, 1, 1),
        max_value=today
    )

    # ---------------- input checks ----------------
    dates_ok = loss_date >= policy_start
    tenure_days = (loss_date - policy_start).days

    if not dates_ok:
        st.error("Loss Date cannot be earlier than the First Policy Subscription Date.")
    elif tenure_days < 30:
        st.info(
            f"Policy tenure is only {tenure_days} day(s). "
            "A claim this soon after inception is unusual."
        )

    if claim_amount > CLAIM_CAP:
        st.warning(
            f"Claim amount is above the 99th percentile of historical claims "
            f"({CLAIM_CAP:,.0f}). The model caps it at this value, so the score "
            "cannot go beyond what it learned from real data."
        )

    if st.button(
        "🔎 Predict Fraud Risk",
        type="primary",
        disabled=not dates_ok
    ):

        live = pd.DataFrame({
            "ClaimCause": [claim_cause],
            "ClaimInvolvedCovers": [
                " ".join(covers)
                if covers else "NoCoverInfo"
            ],
            "FirstPolicySubscriptionDate": [
                pd.to_datetime(policy_start)
            ],
            "LossDate": [
                pd.to_datetime(loss_date)
            ],
            "ConnectionBetweenParties": [connection],
            "FirstPartyVehicleType": [vehicle_type],
            "EasinessToStage": [easiness],
            "ClaimAmount": [claim_amount],
            "ClaimWihoutIdentifiedThirdParty": [third_party],
            "LossHour": [loss_hour],
            "FpVehicleAgeMonths": [vehicle_age],
            "NumberOfBodilyInjuries": [bodily_injuries],
            "PolicyWasSubscribedOnInternet": [
                internet_policy
            ]
        })

        # Same feature function as training (no train/serve mismatch)
        live_X = build_features(live, CLAIM_CAP).reindex(
            columns=results["feature_cols"],
            fill_value=0
        )

        model = results["model"]

        score = float(model.predict_proba(live_X)[0, 1])

        percentile = float((results["scores"] < score).mean())

        st.divider()

        st.subheader("Live Model Output")

        if score >= results["cutoff"]:
            st.error(
                "🚨 HIGH RISK — inside the top-5% investigation queue"
            )
        elif percentile >= 0.80:
            st.warning(
                "⚠️ ELEVATED RISK — above 80% of test claims, "
                "but below the top-5% queue threshold"
            )
        else:
            st.success(
                "✅ LOWER RISK — below 80% of test claims"
            )

        m1, m2, m3 = st.columns(3)

        m1.metric("Risk Score", f"{score:.2%}")
        m2.metric("Percentile vs test claims", f"{percentile:.1%}")
        m3.metric("Top-5% queue threshold", f"{results['cutoff']:.2%}")

        st.progress(min(max(score, 0.0), 1.0))

        st.caption(
            "The score is a ranking signal, not a calibrated probability of fraud "
            "(class balancing and reweighing inflate it). Use the percentile and "
            "the queue threshold to interpret it."
        )

        # ---- why this score? ----
        scaled = model[:-1].transform(live_X)

        contrib = pd.Series(
            scaled[0] * model.named_steps["logreg"].coef_[0],
            index=results["feature_cols"]
        )

        contrib = contrib[contrib != 0]

        top = contrib.loc[
            contrib.abs().sort_values(ascending=False).index
        ].head(6)

        st.subheader("Top drivers of this score")

        st.dataframe(
            pd.DataFrame({
                "Feature": top.index,
                "Contribution (log-odds)": top.values.round(3),
                "Direction": np.where(
                    top.values > 0,
                    "Pushes toward fraud",
                    "Pushes away from fraud"
                )
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Important: This output is a risk score, not proof that a claim is fraudulent. "
            "The purpose is to prioritize claims for human investigation."
        )

    with st.expander("Simulator sanity check (real unseen test claims)"):

        check = pd.Series(
            results["scores"],
            index=results["y_test"].index
        )

        is_fraud = results["y_test"] == 1

        check_df = pd.DataFrame({
            "Group": ["Actual fraud", "Actual non-fraud"],
            "Claims": [int(is_fraud.sum()), int((~is_fraud).sum())],
            "Average Risk Score": [
                f"{check[is_fraud].mean():.2%}",
                f"{check[~is_fraud].mean():.2%}"
            ],
            "Share inside top-5% queue": [
                f"{(check[is_fraud] >= results['cutoff']).mean():.2%}",
                f"{(check[~is_fraud] >= results['cutoff']).mean():.2%}"
            ]
        })

        st.dataframe(
            check_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Actual fraud claims should score clearly higher than non-fraud claims. "
            "If not, the model is weak, not the simulator."
        )


# ============================================================
# PAGE 9 — CLEANED DATA
# ============================================================

elif page == "10. Cleaned Data":

    st.header("🔟 Final Cleaned Dataset")

    st.write(
        "This is the dataset after the documented cleaning process."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{len(df_clean):,}"
    )

    c2.metric(
        "Columns",
        df_clean.shape[1]
    )

    c3.metric(
        "Fraud Cases",
        int(df_clean["Fraud"].sum())
    )

    st.dataframe(
        df_clean,
        use_container_width=True
    )

    csv = df_clean.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption(
    "Fraud Detection Project\n"
    "Logistic Regression + L1 Regularization\n"
    "5-Fold Cross Validation\n"
    "Top 5% Risk-Based Investigation\n"
    "Bias Mitigation: Proxy Removal + Reweighing + Fairness Audit"
)
