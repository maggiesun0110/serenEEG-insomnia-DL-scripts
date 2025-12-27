import os
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =====================
# Paths
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DL_INS = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_PATH = os.path.join(
    DL_INS,
    "dl_ins_results",
    "combined_subject_features.npz"
)

RF_PATH = os.path.join(SCRIPT_DIR, "results", "rf_cnn_embeddings.joblib")
XGB_PATH = os.path.join(SCRIPT_DIR, "results", "xgb_cnn_embeddings.joblib")

OUT_DIR = os.path.join(SCRIPT_DIR, "results", "shap")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
data = np.load(DATA_PATH)
X = data["X"]        # (subjects, 128)
y = data["y"]

# =====================
# Load models
# =====================
rf = joblib.load(RF_PATH)
xgb_model = joblib.load(XGB_PATH)

# =====================
# SHAP for Random Forest
# =====================
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X)

# Handle binary-class output safely
if isinstance(shap_values_rf, list):
    shap_rf_insomnia = shap_values_rf[1]   # class 1 = insomnia
else:
    shap_rf_insomnia = shap_values_rf      # already correct shape

shap.summary_plot(
    shap_rf_insomnia,
    X,
    show=False
)
plt.savefig(os.path.join(OUT_DIR, "rf_shap_summary.png"), dpi=300)
plt.close()

# =====================
# SHAP for XGBoost
# =====================
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X)

shap.summary_plot(
    shap_values_xgb,
    X,
    show=False
)
plt.savefig(os.path.join(OUT_DIR, "xgb_shap_summary.png"), dpi=300)
plt.close()