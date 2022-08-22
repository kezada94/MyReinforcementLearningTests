from functools import wraps
from textwrap import wrap

import mlflow

from constants import MLFLOW_DASHBOARD_URI, MLFLOW_EXPERIMENT_NAME


def mlflow_run(wrapped_function):
    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        mlflow.set_tracking_uri(MLFLOW_DASHBOARD_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        import os
        print("ENVIRON", os.environ['MLFLOW_RUN_ID'])
        with mlflow.start_run(run_id=os.environ['MLFLOW_RUN_ID']):
            with mlflow.start_run(run_name=wrapped_function.__name__, nested=True):
                return wrapped_function(*args, **kwargs)
    return wrapper