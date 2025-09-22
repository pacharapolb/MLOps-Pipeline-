# scripts/04_load_and_predict.py
from pathlib import Path
import os
import pandas as pd
import mlflow
import mlflow.sklearn

MODEL_NAME = "bc-classifier-prod"
STAGE = "Staging"  # เปลี่ยนใน UI ก่อน

def main():
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    raw_path = artifacts / "raw_bc.csv"

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))

    # พยายามโหลดจาก Model Registry ก่อน
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        source = "registry"
    except Exception:
        # fallback: โหลดจากไฟล์โมเดลล่าสุดใน artifacts (กรณีไม่ได้ register)
        candidates = sorted((artifacts / "model").glob("**/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if (artifacts / "model").exists():
            model = mlflow.sklearn.load_model(str(artifacts / "model"))
            source = "artifacts/model"
        else:
            raise

    df = pd.read_csv(raw_path).drop(columns=["target"]).head(5)
    preds = model.predict(df)

    print(f"[INFO] Loaded model from: {source}")
    print("[OK] Sample predictions (first 5 rows):", preds.tolist())

    # log ตัวอย่าง input/output เป็น artifacts ของ run นี้
    mlflow.set_experiment("BC Wisconsin - Inference")
    with mlflow.start_run(run_name="04_inference"):
        sample_in = artifacts / "inference_input_sample.csv"
        sample_out = artifacts / "inference_output_sample.csv"
        df.to_csv(sample_in, index=False)
        pd.DataFrame({"prediction": preds}).to_csv(sample_out, index=False)
        mlflow.log_artifact(str(sample_in))
        mlflow.log_artifact(str(sample_out))

if __name__ == "__main__":
    main()