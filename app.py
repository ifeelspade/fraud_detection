
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
        "8. Live Fraud Prediction",
        "9. Cleaned Data"
    ]
)

# ============================================================
# LOAD DATA
# ============================================================

FILE_PATH = "C:\\Users\\Acer\\Desktop\\AIML\\historical_data.csv"

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

@st.cache_data
def prepare_features(df):

    data = df.copy()

    data["PolicyTenureDays"] = (
        data["LossDate"]
        - data["FirstPolicySubscriptionDate"]
    ).dt.days

    data["ClaimAmount_log"] = np.log1p(
        data["ClaimAmount"]
    )

    data["NightLoss"] = (
        data["LossHour"]
        .isin([22, 23, 1, 2, 3, 4, 5])
        .astype(int)
    )

    cover_flags = (
        data["ClaimInvolvedCovers"]
        .str.get_dummies(sep=" ")
        .add_prefix("Cover_")
    )

    cat_cols = [
        "ClaimCause",
        "ConnectionBetweenParties",
        "FirstPartyVehicleType"
    ]

    cat_flags = pd.get_dummies(
        data[cat_cols],
        dtype=int
    )

    num_cols = [
        "PolicyTenureDays",
        "ClaimAmount_log",
        "NightLoss",
        "FpVehicleAgeMonths",
        "EasinessToStage",
        "NumberOfBodilyInjuries",
        "ClaimWihoutIdentifiedThirdParty",
        "PolicyWasSubscribedOnInternet"
    ]

    X = pd.concat(
        [
            data[num_cols],
            cat_flags,
            cover_flags
        ],
        axis=1
    )

    y = data["Fraud"]

    return X, y


@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
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

    grid.fit(X_train, y_train)

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
        index=X.columns
    )

    return {
        "model": best_model,
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
# RUN PIPELINE
# ============================================================

try:
    df_raw = load_raw_data()
    df_clean, cleaning_info = clean_data(df_raw)
    X, y = prepare_features(df_clean)
    results = train_model(X, y)

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
    - limit unnecessary bias from unrelated variables
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
    **Median Imputation + StandardScaler**
    ↓  
    **L1 Logistic Regression**
    ↓  
    **5-Fold Cross Validation + GridSearchCV**
    ↓  
    **Risk scoring**
    ↓  
    **Flag top 5% highest-risk claims**
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
        X.shape[1]
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
# PAGE 8 — LIVE FRAUD PREDICTION
# ============================================================

elif page == "8. Live Fraud Prediction":

    st.header("8️⃣ Live Fraud Prediction")

    st.write(
        "Enter claim information below. The trained model will calculate "
        "a fraud probability and classify the claim as high or low risk."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        claim_amount = st.number_input(
            "Claim Amount",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )

        vehicle_age = st.number_input(
            "Vehicle Age (months)",
            min_value=0.0,
            value=100.0,
            step=1.0
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
            [0.25, 0.50]
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
                "Yes" if x == 1 else "No"
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
        [
            "MaterialDamages",
            "ActLiability",
            "ReplacementVehicle",
            "MedicalCare",
            "Fire",
            "Theft"
        ],
        default=["MaterialDamages"]
    )

    policy_start = st.date_input(
        "First Policy Subscription Date"
    )

    loss_date = st.date_input(
        "Loss Date"
    )

    if st.button(
        "🔎 Predict Fraud Risk",
        type="primary"
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
            ],
            "LossHour_Unknown": [0]
        })

        live["PolicyTenureDays"] = (
            live["LossDate"]
            - live["FirstPolicySubscriptionDate"]
        ).dt.days

        live["ClaimAmount_log"] = np.log1p(
            live["ClaimAmount"]
        )

        live["NightLoss"] = (
            live["LossHour"]
            .isin([22, 23, 1, 2, 3, 4, 5])
            .astype(int)
        )

        cover_flags_live = (
            live["ClaimInvolvedCovers"]
            .str.get_dummies(sep=" ")
            .add_prefix("Cover_")
        )

        cat_flags_live = pd.get_dummies(
            live[
                [
                    "ClaimCause",
                    "ConnectionBetweenParties",
                    "FirstPartyVehicleType"
                ]
            ],
            dtype=int
        )

        num_cols_live = [
            "PolicyTenureDays",
            "ClaimAmount_log",
            "NightLoss",
            "FpVehicleAgeMonths",
            "EasinessToStage",
            "NumberOfBodilyInjuries",
            "ClaimWihoutIdentifiedThirdParty",
            "PolicyWasSubscribedOnInternet"
        ]

        live_X = pd.concat(
            [
                live[num_cols_live],
                cat_flags_live,
                cover_flags_live
            ],
            axis=1
        )

        # Align exactly with training columns
        live_X = live_X.reindex(
            columns=X.columns,
            fill_value=0
        )

        probability = (
            results["model"]
            .predict_proba(live_X)[0, 1]
        )

        st.divider()

        st.subheader("Live Model Output")

        if probability >= results["cutoff"]:
            st.error(
                f"🚨 HIGH RISK — Fraud Probability: {probability:.2%}"
            )
        else:
            st.success(
                f"✅ LOWER RISK — Fraud Probability: {probability:.2%}"
            )

        st.progress(
            min(float(probability), 1.0)
        )

        st.write(
            f"Model risk threshold used for the top-5% investigation queue: "
            f"**{results['cutoff']:.4f}**"
        )

        st.info(
            "Important: This output is a risk score, not proof that a claim is fraudulent. "
            "The purpose is to prioritize claims for human investigation."
        )


# ============================================================
# PAGE 9 — CLEANED DATA
# ============================================================

elif page == "9. Cleaned Data":

    st.header("9️⃣ Final Cleaned Dataset")

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
    "Top 5% Risk-Based Investigation"
)
