import mlflow
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MLFLOW_TRACKING_URI = str(ROOT / "mlruns")
EXPERIMENT_NAME = "urbaneye"


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)