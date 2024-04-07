from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.lifecycle_stage import LifecycleStage


class MlflowHandler:
    """Class for handling Mlflow metrics logging."""

    def __init__(
        self,
        experiment_name: str,
        run_name: None | str = None,
        tags: dict = {},
        mlflow_url: str = "http://127.0.0.1:5000/",
    ) -> None:
        """Metrics Logger handler.

        Args:
            experiment_name: A name of the experiment.
            run_name: A name of run.
            tags: Tags of software version.
            mlflow_url: URL address of Mlflow server.
        """
        self.mlflow_url: str = mlflow_url
        self.experiment_name: str = experiment_name
        self.run_name: str | None = run_name
        self.tags: dict = tags

        self.client: MlflowClient = MlflowClient(self.mlflow_url)
        self.experiment_id: int = self._get_or_create_experiment_id(
            self.experiment_name
        )
        self.run: Run | None = None

    @property
    def run_id(self) -> int:
        """Get run id."""
        return self.run.info.run_id

    def _create_run(self) -> Run:
        """Create Mlflow run."""
        return self.client.create_run(
            self.experiment_id, tags=self.tags, run_name=self.run_name
        )

    def run_experiment(self) -> None:
        """Start the run in Mlflow."""
        if self.run is None:
            self.run = self._create_run()

        mlflow.set_tracking_uri(self.mlflow_url)
        mlflow.set_experiment(experiment_id=self.experiment_id)
        if self.run.info.lifecycle_stage == LifecycleStage.DELETED:
            self.client.restore_run(run_id=self.run_id)

        mlflow.start_run(run_id=self.run_id)

    def finish_experiment(self):
        """Terminate run."""
        self.client.set_terminated(self.run_id, status="FINISHED")

    def log_param(self, param: str, value: Any) -> None:
        """Logs param to Mlflow."""
        try:
            self.client.log_param(self.run_id, param, value)
        except mlflow.exceptions.MlflowException:
            pass

    def log_artifact(self, local_path: str, artifact_path: Any | None = None) -> None:
        """Logs artifact to Mlflow."""
        try:
            self.client.log_artifact(self.run_id, local_path, artifact_path)
        except mlflow.exceptions.MlflowException:
            pass

    def log_metric(self, metric: str, value: float, step: int) -> None:
        """Log metric to Mlflow."""
        try:
            self.client.log_metric(self.run_id, metric, value, step=step)
        except mlflow.exceptions.MlflowException:
            pass

    def _get_or_create_experiment_id(self, experiment_name: str) -> str:
        """If experiment exists get experiment id else create experiment.

        Args:
            experiment_name: A name of the experiment.

        Returns:
            If of the experiment.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage != "deleted":
            return experiment.experiment_id
        return self.client.create_experiment(experiment_name)
