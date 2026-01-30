import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# Custom class for Hybrid Calibration - MOVED TO TOP
class HybridCalibrator:
    def __init__(self, calibrators, weights):
        self.calibrators = calibrators
        self.weights = weights

    def predict_proba(self, X_proba_uncalibrated):
        p_cal_platt_input = X_proba_uncalibrated.reshape(-1, 1)

        calibrated_probas = []
        calibrated_probas.append(self.weights[0] * self.calibrators[0].predict_proba(p_cal_platt_input)[:, 1])
        calibrated_probas.append(self.weights[1] * self.calibrators[1].predict(X_proba_uncalibrated))
        calibrated_probas.append(self.weights[2] * self.calibrators[2].predict(X_proba_uncalibrated))

        return sum(calibrated_probas)

# Custom function for ECE
def expected_calibration_error(y_true, y_prob, n_bins):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(
                np.mean(y_prob[mask]) - np.mean(y_true[mask])
            )
    return ece

# Function to check calibration by group
def calibration_group_check(y_true, y_prob, bins):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    results = []

    for low, high in bins:
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            continue

        avg_prob = y_prob[mask].mean()
        actual_rate = y_true[mask].mean()
        count = mask.sum()

        results.append({
            "Prob_Range": f"{low:.2f} - {high:.2f}",
            "Jumlah_Data": count,
            "Rata_Probabilitas": round(avg_prob, 3),
            "Kejadian_Aktual": round(actual_rate, 3),
            "Selisih": round(abs(avg_prob - actual_rate), 3)
        })

    return pd.DataFrame(results)

# Set Streamlit page configuration as the first command
st.set_page_config(layout="wide")

# Load models and test data
@st.cache_resource
def load_resources():
    preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
    xgb_model = joblib.load("xgboost_baseline.pkl")
    hybrid_calibrator = joblib.load("hybrid_calibrator.pkl")
    X_test_loaded = joblib.load("X_test.pkl")
    y_test_loaded = joblib.load("y_test.pkl")
    return preprocessing_pipeline, xgb_model, hybrid_calibrator, X_test_loaded, y_test_loaded

preprocessing_pipeline, xgb_model, hybrid_calibrator, X_test, y_test = load_resources()

st.title("Model Calibration Evaluation for Bank Marketing Prediction")

st.markdown("""
This application evaluates the performance and calibration of a baseline XGBoost model
and a hybrid calibrated XGBoost model on the Bank Marketing dataset.
""")

# # --- Predictions ---
# st.header("1. Model Predictions")

# Baseline predictions
y_proba_baseline = xgb_model.predict_proba(X_test)[:, 1]
y_pred_baseline = (y_proba_baseline >= 0.5).astype(int)

# Hybrid predictions
# First get uncalibrated probabilities for the hybrid calibrator
p_test_uncal = xgb_model.predict_proba(X_test)[:, 1]
y_proba_hybrid_weighted = hybrid_calibrator.predict_proba(p_test_uncal)
y_pred_hybrid = (y_proba_hybrid_weighted >= 0.5).astype(int)


# st.subheader("Classification Threshold Optimization for Hybrid Model")
# st.markdown("We'll check recall and precision at different thresholds for the hybrid model to find a suitable balance.")

# thresholds = np.arange(0.1, 0.6, 0.05)
# results = []

# for t in thresholds:
#     y_pred_t = (y_proba_hybrid_weighted >= t).astype(int)
#     rec = recall_score(y_test, y_pred_t, zero_division=0)
#     pre = precision_score(y_test, y_pred_t, zero_division=0)
#     results.append({
#         "Threshold": t,
#         "Recall": rec,
#         "Precision": pre
#     })

# results_df = pd.DataFrame(results)
# st.dataframe(results_df.round(3), use_container_width=True)
# st.markdown("Based on the evaluation in the notebook, a threshold of 0.5 was chosen for the hybrid model.")

# --- Evaluation Metrics ---
# st.header("Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Metrics")
    metrics_comparison = pd.DataFrame({
        "Model": ["Baseline XGBoost", "Hybrid Weighted"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_baseline),
            accuracy_score(y_test, y_pred_hybrid)
        ],
        "Precision": [
            precision_score(y_test, y_pred_baseline, zero_division=0),
            precision_score(y_test, y_pred_hybrid, zero_division=0)
        ],
        "Recall": [
            recall_score(y_test, y_pred_baseline, zero_division=0),
            recall_score(y_test, y_pred_hybrid, zero_division=0)
        ],
        "F1-score": [
            f1_score(y_test, y_pred_baseline, zero_division=0),
            f1_score(y_test, y_pred_hybrid, zero_division=0)
        ]
    }).round(3)
    st.dataframe(metrics_comparison, use_container_width=True)

    st.subheader("Confusion Matrices")
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(12, 6))

    # Baseline Confusion Matrix
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=["No", "Yes"])
    disp_baseline.plot(cmap="Reds", ax=axes_cm[0])
    axes_cm[0].set_title("Confusion Matrix - Baseline XGBoost")

    # Hybrid Confusion Matrix
    cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
    disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid, display_labels=["No", "Yes"])
    disp_hybrid.plot(cmap="Reds", ax=axes_cm[1])
    axes_cm[1].set_title("Confusion Matrix - Hybrid Weighted")

    plt.tight_layout()
    st.pyplot(fig_cm)


with col2:
    st.subheader("Probability Metrics")
    calibration_comparison = pd.DataFrame({
        "Model": [
            "Baseline XGBoost",
            "Hybrid Weighted"
        ],
        "Brier Score": [
            brier_score_loss(y_test, y_proba_baseline),
            brier_score_loss(y_test, y_proba_hybrid_weighted)
        ],
        "ECE": [
            expected_calibration_error(y_test.values, y_proba_baseline, n_bins=10),
            expected_calibration_error(y_test.values, y_proba_hybrid_weighted, n_bins=10)
        ]
    }).round(4)
    st.dataframe(calibration_comparison, use_container_width=True)

    st.subheader("Reliability Diagrams")
    fig_rel, axes_rel = plt.subplots(1, 2, figsize=(12, 6))

    # Baseline Reliability Diagram
    prob_true_baseline, prob_pred_baseline = calibration_curve(
        y_test, y_proba_baseline, n_bins=10, strategy="uniform"
    )
    axes_rel[0].plot(prob_pred_baseline, prob_true_baseline, marker="o", label="Model")
    axes_rel[0].plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    axes_rel[0].set_title("Reliability Diagram - Baseline XGBoost")
    axes_rel[0].set_xlabel("Mean predicted probability")
    axes_rel[0].set_ylabel("Fraction of positives")
    axes_rel[0].set_xlim(0, 1)
    axes_rel[0].set_ylim(0, 1)
    axes_rel[0].legend()

    # Hybrid Reliability Diagram
    prob_true_hybrid, prob_pred_hybrid = calibration_curve(
        y_test, y_proba_hybrid_weighted, n_bins=10, strategy="uniform"
    )
    axes_rel[1].plot(prob_pred_hybrid, prob_true_hybrid, marker="o", label="Model")
    axes_rel[1].plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    axes_rel[1].set_title("Reliability Diagram - Hybrid Weighted Calibration")
    axes_rel[1].set_xlabel("Mean predicted probability")
    axes_rel[1].set_ylabel("Fraction of positives")
    axes_rel[1].set_xlim(0, 1)
    axes_rel[1].set_ylim(0, 1)
    axes_rel[1].legend()

    plt.tight_layout()
    st.pyplot(fig_rel)

st.header("Probability per Bin Analysis")
st.markdown("This table shows the average predicted probability and actual event rate within specific probability ranges, highlighting calibration.")

bins = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00)
]

st.subheader("Baseline Model")
baseline_group_check_df = calibration_group_check(y_test, y_proba_baseline, bins)
st.dataframe(baseline_group_check_df, use_container_width=True)

st.subheader("Hybrid Calibrated Model")
hybrid_group_check_df = calibration_group_check(y_test, y_proba_hybrid_weighted, bins)
st.dataframe(hybrid_group_check_df, use_container_width=True)

st.markdown("""
The `HybridCalibrator` class in `hybrid_calibrator.pkl` combines Platt Scaling, Isotonic Regression,
and Beta Calibration with data-driven weights.
""")