# mlops_pipeline/scripts/04_transition_model.py
import sys, os
import mlflow
from mlflow.tracking import MlflowClient

def transition_model_alias(model_name: str, target_stage: str):
    """
    เปลี่ยน Stage ของ model version ล่าสุดใน Model Registry
    เช่น Staging / Production / Archived
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"[WARN] No versions found for model '{model_name}'")
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage=target_stage,
        archive_existing_versions=False
    )
    print(f"[OK] Transitioned {model_name} v{latest.version} -> {target_stage}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/04_transition_model.py <model_name> <target_stage>")
        sys.exit(1)
    transition_model_alias(sys.argv[1], sys.argv[2])
