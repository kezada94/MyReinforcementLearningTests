import mlflow
import typer

import sys
sys.path.append('.')

from constants import MLFLOW_DASHBOARD_URI, MLFLOW_EXPERIMENT_NAME

def start_pipeline(run_name):
    mlflow.set_tracking_uri(MLFLOW_DASHBOARD_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        print(mlflow.active_run().info.run_id)
        mlflow.log_artifact("dvc.yaml")


if __name__ == "__main__":
    typer.run(start_pipeline)
