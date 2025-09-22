from pathlib import Path
import os, json
import pandas as pd
import mlflow
from sklearn.datasets import load_breast_cancer

EXPERIMENT = "BC Wisconsin - Data Validation"

def main():
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    artifacts.mkdir(exist_ok=True)

    # โหลด dataset จาก sklearn
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame  # มีคอลัมน์ 'target' อยู่แล้ว (0=malignant, 1=benign)

    raw_path = artifacts / "raw_bc.csv"
    df.to_csv(raw_path, index=False)

    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "target_counts": {int(k): int(v) for k, v in df["target"].value_counts().to_dict().items()},
    }
    report_path = artifacts / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="01_data_validation"):
        mlflow.log_param("dataset", "breast_cancer_wisconsin")
        mlflow.log_metric("n_rows", report["n_rows"])
        mlflow.log_metric("n_cols", report["n_cols"])
        mlflow.log_metric("missing_total", report["missing_total"])
        for k, v in report["target_counts"].items():
            mlflow.log_metric(f"target_count_{k}", v)
        mlflow.log_artifact(str(raw_path))
        mlflow.log_artifact(str(report_path))

    print(f"[OK] Saved raw -> {raw_path}")
    print(f"[OK] Saved report -> {report_path}")

if __name__ == "__main__":
    main()
