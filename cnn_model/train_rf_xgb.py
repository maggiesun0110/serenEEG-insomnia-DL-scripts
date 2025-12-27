import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

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

OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
data = np.load(DATA_PATH)
X = data["X"]        # (subjects, 128)
y = data["y"]        # (subjects,)
subjects = data["subject_ids"]  # subject IDs

print("Loaded features:", X.shape)
print("Labels distribution:", np.bincount(y.astype(int)))

# =====================
# LOSO-style CV
# =====================
gkf = GroupKFold(n_splits=len(np.unique(subjects)))

rf_preds, xgb_preds, y_true = [], [], []

for train_idx, test_idx in gkf.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds.append(rf.predict(X_test)[0])

    # ---- XGBoost ----
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)
    xgb_preds.append(xgb_clf.predict(X_test)[0])

    y_true.append(y_test[0])

y_true = np.array(y_true)
rf_preds = np.array(rf_preds)
xgb_preds = np.array(xgb_preds)

# =====================
# Metrics
# =====================
def report(name, preds):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, preds))
    print("Precision:", precision_score(y_true, preds))
    print("Recall:", recall_score(y_true, preds))
    print("F1:", f1_score(y_true, preds))
    print("Confusion matrix:\n", confusion_matrix(y_true, preds))

report("Random Forest", rf_preds)
report("XGBoost", xgb_preds)

# =====================
# Train final models
# =====================
rf_final = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
rf_final.fit(X, y)

xgb_final = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb_final.fit(X, y)

joblib.dump(rf_final, os.path.join(OUT_DIR, "rf_cnn_embeddings.joblib"))
joblib.dump(xgb_final, os.path.join(OUT_DIR, "xgb_cnn_embeddings.joblib"))

print("\nSaved final RF + XGB models.")