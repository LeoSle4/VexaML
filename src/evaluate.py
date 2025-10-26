import json, yaml, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

CFG = yaml.safe_load(Path("configs/config.yml").read_text())
SEED = CFG["ml"]["seed"]
TEST_SIZE = CFG["ml"]["test_size"]
DATA_PATH = Path(CFG["data"]["processed_path"])
MODEL_PATH = Path("models/model.pkl")
THRESH_PATH = Path("models/threshold.json")


def bootstrap_ci(metric_fn, y_true, y_pred, B=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df = pd.read_parquet(DATA_PATH)
    y = df["loss_event"].astype(int)
    X = df.drop(columns=["loss_event", "event_id"], errors="ignore")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    pipe = joblib.load(MODEL_PATH)
    th = json.loads(Path(THRESH_PATH).read_text())["threshold"]

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= th).astype(int)

    rec = recall_score(y_te, pred)
    pre = precision_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred)
    acc = accuracy_score(y_te, pred)
    try:
        auc = roc_auc_score(y_te, proba)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_te, pred).tolist()

    rec_ci = bootstrap_ci(recall_score, y_te.values, pred)
    pre_ci = bootstrap_ci(
        lambda a, b: precision_score(a, b, zero_division=0), y_te.values, pred
    )
    f1_ci = bootstrap_ci(f1_score, y_te.values, pred)
    acc_ci = bootstrap_ci(accuracy_score, y_te.values, pred)

    report = {
        "threshold": th,
        "metrics": {
            "recall": {"value": float(rec), "ci95": rec_ci},
            "precision": {"value": float(pre), "ci95": pre_ci},
            "f1": {"value": float(f1), "ci95": f1_ci},
            "accuracy": {"value": float(acc), "ci95": acc_ci},
            "roc_auc": float(auc),
        },
        "confusion_matrix": cm,
        "n_test": int(len(y_te)),
        "prevalence": float(y_te.mean()),
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics_test.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
