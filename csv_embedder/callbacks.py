from lightning.pytorch.callbacks import Callback, TQDMProgressBar
import lightning.pytorch as pl
import pdb

from lightning.pytorch import loggers
from lightning.pytorch.utilities import rank_zero_only

class TBLogger(loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)
    
class PretrainLoaderCallback(Callback):
    def __init__(self, pretrained_path):
        super().__init__()
        self.pretrained_path = pretrained_path

    def on_train_start(self, trainer, pl_module):
        pl_module.load_weights(self.pretrained_path)
        print("Loaded pretraining weights from {}".format(self.pretrained_path))

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_training_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)
        pdb.set_trace()
