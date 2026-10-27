import lightning as L
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


class MLFlowSystemMonitorCallback(L.Callback):
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not isinstance(trainer.logger, MLFlowLogger):
            raise MisconfigurationException(
                "MLFlowSystemMonitorCallback requires MLFlowLogger"
            )

        self.system_monitor = SystemMetricsMonitor(
            run_id=trainer.logger.run_id,
            tracking_uri=trainer.logger.experiment.tracking_uri,
        )
        self.system_monitor.start()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.system_monitor.finish()
