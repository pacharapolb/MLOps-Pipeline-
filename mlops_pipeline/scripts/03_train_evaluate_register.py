import sys, os, tempfile
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EXPERIMENT = "BC Wisconsin - Training"
REGISTERED_NAME = "bc-classifier-prod"
ACC_THRESHOLD = 0.95  # เกณฑ์สำหรับลงทะเบียน

def download_all_artifacts(run_id: str) -> Path:
    # ดาวน์โหลด artifacts จาก run preprocess
    try:
        from mlflow import artifacts
        local_dir = artifacts.download_artifacts(run_id=run_id, artifact_path="")
        return Path(local_dir)
    except Exception:
        from mlflow.tracking import MlflowClient
        tmpdir = Path(tempfile.mkdtemp())
        MlflowClient().download_artifacts(run_id, "", str(tmpdir))
        return tmpdir

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocess_run_id> [C_value]")
        sys.exit(1)

    preprocess_run_id = sys.argv[1]
    C_val = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)

    art_dir = download_all_artifacts(preprocess_run_id)
    train_path = art_dir / "train.csv"
    test_path = art_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Pipeline อีกชั้น (กันพลาดถ้าใครข้าม preprocess)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # ข้อมูลถูกสเกลแล้ว แต่กันพลาด
        ("clf", LogisticRegression(max_iter=200, C=C_val, solver="lbfgs"))
    ])

    with mlflow.start_run(run_name="03_train_evaluate") as run:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        prec = precision_score(y_test, y_pred, average="binary")
        rec = recall_score(y_test, y_pred, average="binary")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C_val)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        signature = infer_signature(X_test, y_pred)
        input_example = X_test.head(2)

        # ถ้า accuracy ผ่านเกณฑ์ -> ลงทะเบียนเข้า Model Registry
        registered = None
        if acc >= ACC_THRESHOLD:
            registered = REGISTERED_NAME

        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered,  # ถ้า None จะไม่ register
        )

        print(f"[OK] Metrics: acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f}")
        print(f"[OK] Run ID (train): {run.info.run_id}")
        if registered:
            print(f"[OK] Registered to Model Registry as '{REGISTERED_NAME}' (new version created).")
        else:
            print("[WARN] Accuracy < threshold; model not registered. You can still serve from runs:/<RUN_ID>/model")

if __name__ == "__main__":
    main()