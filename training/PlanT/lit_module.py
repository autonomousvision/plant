import logging

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy

from training.PlanT.model import HFLM

logger = logging.getLogger(__name__)


class LitHFLM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.last_epoch = 0
        self.cfg_train = self.cfg.model.training
        self.model = HFLM(self.cfg.model.network, self.cfg)

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_forecast = nn.CrossEntropyLoss(ignore_index=-999)
        
        # Metrics
        if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
            self.metrics_forecasting_acc = nn.ModuleList(
                [Accuracy() for i in range(self.model.num_attributes)]
            )
            

    def forward(self, x, y=None, target_point=None, light_hazard=None):
        return self.model(x, y, target_point, light_hazard)


    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.cfg.model.training)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[self.cfg.lrDecay_epoch, self.cfg.lrDecay_epoch + 10],
            gamma=0.1,
        )
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y, wp, tp, light = batch

        # training with only waypoint loss
        if self.cfg.model.pre_training.get("pretraining", "none") == "none":
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "train/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

        elif self.cfg.model.pre_training.get("pretraining", "none") == "forecast":

            if self.cfg.model.pre_training.get("multitask", False) == True:
                # multitask training
                logits, targets, pred_wp, _ = self(x, y, tp, light)
                loss_wp = F.l1_loss(pred_wp, wp)
                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_forecast = torch.mean(torch.stack(losses_forecast))

                loss_all = (
                    1                                                           * loss_wp
                    + self.cfg.model.pre_training.get("forecastLoss_weight", 0) * loss_forecast
                )
                self.log(
                    "train/loss_forecast",
                    loss_forecast,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )
                self.log(
                    "train/loss_wp",
                    loss_wp,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

            else:
                # 2 stage training (pre-training only on forecasting - no waypoint loss)
                logits, targets = self(x, y)

                losses_forecast = [
                    torch.mean(self.criterion_forecast(logits[i], targets[i].squeeze()))
                    for i in range(len(logits))
                ]
                loss_all = torch.mean(torch.stack(losses_forecast))

            self.log(
                "train/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            for i, name in enumerate(
                ["x", "y", "yaw", "speed", "extent_x", "extent_y"]
            ):
                if i > self.model.num_attributes:
                    break
                self.log(
                    f"train/loss_{name}",
                    losses_forecast[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

                mask = targets[i].squeeze() != -999
                self.metrics_forecasting_acc[i](
                    logits[i][mask], targets[i][mask].squeeze()
                )
                self.log(
                    f"train/acc_{name}",
                    self.metrics_forecasting_acc[i],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.cfg.gpus > 1,
                    batch_size=self.cfg.model.training.batch_size,
                )

        return loss_all


    def validation_step(self, batch, batch_idx):

        if self.cfg.model.pre_training.get("pretraining", "none") == "forecast":
            # TODO: add proper validation set for multitask
            pass

        else:
            x, y, wp, tp, light = batch

            self.y = y
            logits, pred_wp, _ = self(x, y, tp, light)

            loss_pred = F.l1_loss(pred_wp, wp)
            loss_all = loss_pred

            self.log(
                "val/loss_all",
                loss_all,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )
            self.log(
                "val/loss_pred",
                loss_pred,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.cfg.gpus > 1,
                batch_size=self.cfg.model.training.batch_size,
            )

            self.last_epoch = self.current_epoch


    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg_train.grad_norm_clip
        )
