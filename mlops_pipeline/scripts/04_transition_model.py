# mlops_pipeline/scripts/04_transition_model.py
import sys, os
import mlflow
from mlflow.tracking import MlflowClient

def transition_model_alias(model_name: str, target_alias: str):
    """
    กำหนด alias ให้ model version ล่าสุด เช่น "staging" หรือ "prod"
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"[WARN] No versions found for model '{model_name}'")
        return

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=latest.version,
    )
    print(f"[OK] Set alias '{target_alias}' for {model_name} v{latest.version}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/04_transition_model.py <model_name> <target_alias>")
        sys.exit(1)
    transition_model_alias(sys.argv[1], sys.argv[2])
