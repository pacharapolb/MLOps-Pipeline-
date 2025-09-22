from pathlib import Path
import os
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXPERIMENT = "BC Wisconsin - Data Preprocessing"

def main():
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    artifacts.mkdir(exist_ok=True)

    raw_path = artifacts / "raw_bc.csv"
    df = pd.read_csv(raw_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_s, columns=X.columns)
    train_df["target"] = y_train.to_numpy()
    test_df = pd.DataFrame(X_test_s, columns=X.columns)
    test_df["target"] = y_test.to_numpy()

    train_path = artifacts / "train.csv"
    test_path = artifacts / "test.csv"
    scaler_path = artifacts / "scaler.joblib"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="02_data_preprocessing") as run:
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("train_rows", train_df.shape[0])
        mlflow.log_metric("test_rows", test_df.shape[0])
        mlflow.log_artifact(str(train_path))
        mlflow.log_artifact(str(test_path))
        mlflow.log_artifact(str(scaler_path))

        print(f"[OK] Run ID: {run.info.run_id}")  # ใช้ต่อในขั้นถัดไป

if __name__ == "__main__":
    main()