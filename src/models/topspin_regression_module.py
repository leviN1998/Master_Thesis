from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import ConfusionMatrix
import src.utils.regression_utils as regression_utils


class RegressionLitModule(LightningModule):

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """Initialize a `TopspinLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = regression_utils.RotationLoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mean_deg = MeanMetric()
        self.val_mean_deg = MeanMetric()
        self.test_mean_deg = MeanMetric()

        self.train_acc5 = MeanMetric()
        self.val_acc5 = MeanMetric()
        self.test_acc5 = MeanMetric()

        self.val_deg_best = MinMetric()
    

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the model `self.net`.
        """
        return self.net(x, lengths)


    def on_train_start(self):
        self.val_loss.reset()
        self.val_mean_deg.reset()
        self.val_deg_best.reset()


    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of voxel-grids, sequence-lengths and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, lengths, y = batch
        logits = self.forward(x, lengths)
        loss, R_hat = self.criterion(logits, y)

        return loss, R_hat, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of voxel-grids, sequence-lengths and target labels.
        :param batch_idx: The index of the current batch.

        :return: A tensor of losses between model predictions and targets.
        """
        loss, R_hat, targets = self.model_step(batch)
        theta = regression_utils.angle_error_deg(R_hat, targets)

        # update and log metrics
        self.train_loss.update(loss.detach())
        self.train_mean_deg.update(theta.mean().detach())
        self.train_acc5.update(regression_utils.acc_at_threshold_deg(theta, thr=5.0).detach())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mean_deg", self.train_mean_deg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc5", self.train_acc5, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of voxel-grids, sequence-lengths and target labels.
        :param batch_idx: The index of the current batch.
        """
        loss, R_hat, targets = self.model_step(batch)
        theta = regression_utils.angle_error_deg(R_hat, targets)

        # update and log metrics
        self.val_loss.update(loss.detach())
        self.val_mean_deg.update(theta.mean().detach())
        self.val_acc5.update(regression_utils.acc_at_threshold_deg(theta, thr=5.0).detach())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mean_deg", self.val_mean_deg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc5", self.val_acc5, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        deg = self.val_mean_deg.compute()  # get current val deg
        self.val_deg_best(deg)  # update best so far val deg
        self.log("val/deg_best", self.val_deg_best.compute(), sync_dist=True, prog_bar=True)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, R_hat, targets = self.model_step(batch)
        theta = regression_utils.angle_error_deg(R_hat, targets)

        # update and log metrics
        self.test_loss.update(loss.detach())
        self.test_mean_deg.update(theta.mean().detach())
        self.test_acc5.update(regression_utils.acc_at_threshold_deg(theta, thr=5.0).detach())


        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mean_deg", self.test_mean_deg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc5", self.test_acc5, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}