"""Base model with common methods."""

import numpy as np
import torch
import torch.nn as nn
from utils.genLD import genLD
from model.resnet50 import resnet50
from pytorch_lightning import LightningModule
import torchmetrics
from utils.utils import load_obj


class BaseModel(LightningModule):
    """
    Describe model's forward pass, mterics, losses, optimizers etc.

    forward - forward pass of the model
    configure_optimizers - optimizer setting
    generate_ld - generates label distribution
    """

    def __init__(self, config):
        """Init of the model, CNN, loss and metrics objects."""
        super().__init__()
        self.config = config
        # Model
        self.cnn = resnet50(num_acne_cls=self.config.train_val_params.num_acne_cls)
        # Loss functions
        self.kl_loss = nn.KLDivLoss()
        # Metrics
        self.accuracy = torchmetrics.Accuracy()
        self.prec = torchmetrics.Precision(average="macro", num_classes=4)
        self.specificity = torchmetrics.Specificity(average="macro", num_classes=4)
        self.sensitivity = torchmetrics.Recall(average="macro", num_classes=4)
        self.mcc_cls = torchmetrics.MatthewsCorrCoef(num_classes=4)
        self.you_index = self.sensitivity + self.specificity - 1
        self.mae = torchmetrics.MeanAbsoluteError()
        # squared=False indicates, that we are using RMSE metric
        self.mse = torchmetrics.MeanSquaredError(squared=False)

    def forward(self, x):
        """Calculate outputs of the Neural Network."""
        return self.cnn(x)

    def training_epoch_end(self, training_outs):
        """Process data on the end of training epoch.

        The metrics are calculated at the end of the epoch i.e. we
        collect the outputs after each batch and calculate metrics.
        """
        preds_cls = torch.cat([outs["preds_cls"] for outs in training_outs])
        preds_cnt = torch.cat([outs["preds_cnt"] for outs in training_outs])
        b_y = torch.cat([outs["b_y"] for outs in training_outs])
        b_l = torch.cat([outs["b_l"] for outs in training_outs])

        # metrics to log
        metrics = {
            "train_accuracy": self.accuracy(preds_cls, b_y),
            "train_sensitivity": self.sensitivity(preds_cls, b_y),
            "train_prec": self.prec(preds_cls, b_y),
            "train_specificity": self.specificity(preds_cls, b_y),
            "train_you_index": self.you_index(preds_cls, b_y),
            "train_mae": self.mae(preds_cnt, b_l),
            "train_mse": self.mse(preds_cnt, b_l),
            "train_mcc_class": self.mcc_cls(preds_cls, b_y),
        }

        self.log("Perfomance", metrics)

    def validation_epoch_end(self, val_outs):
        """Calculate metrics using data collected on validation step."""
        preds_cls = torch.cat([outs["preds_cls"] for outs in val_outs])
        preds_cnt = torch.cat([outs["preds_cnt"] for outs in val_outs])
        b_y = torch.cat([outs["b_y"] for outs in val_outs])
        b_l = torch.cat([outs["b_l"] for outs in val_outs])

        metrics = {
            "val_accuracy": self.accuracy(preds_cls, b_y),
            "val_sensitivity": self.sensitivity(preds_cls, b_y),
            "val_prec": self.prec(preds_cls, b_y),
            "val_specificity": self.specificity(preds_cls, b_y),
            "val_you_index": self.you_index(preds_cls, b_y),
            "val_mae": self.mae(preds_cnt, b_l),
            "val_mse": self.mse(preds_cnt, b_l),
            "val_mcc_class": self.mcc_cls(preds_cls, b_y),
        }

        self.log("Perfomance", metrics)

    def configure_optimizers(self):
        """Define optimizer and scheduler."""
        # Load optimizer object via custom function (see utils.py)
        optimizer = load_obj(self.config.optimizers.optim_name)(self.cnn.parameters(), **self.config.optimizers.params)
        # Use scheduler to change lr during training
        scheduler = load_obj(self.config.scheduler.scheduler_name)(optimizer, **self.config.scheduler.params)

        return [optimizer], [scheduler]

    def generate_ld(self, b_l):
        """Generate label distribution."""
        # Label distribution
        ld = genLD(b_l.numpy(), self.config.train_val_params.sigma, "klloss", 65)
        ld_13 = np.vstack(
            (
                np.sum(ld[:, :5], 1),
                np.sum(ld[:, 5:10], 1),
                np.sum(ld[:, 10:15], 1),
                np.sum(ld[:, 15:20], 1),
                np.sum(ld[:, 20:25], 1),
                np.sum(ld[:, 25:30], 1),
                np.sum(ld[:, 30:35], 1),
                np.sum(ld[:, 35:40], 1),
                np.sum(ld[:, 40:45], 1),
                np.sum(ld[:, 45:50], 1),
                np.sum(ld[:, 50:55], 1),
                np.sum(ld[:, 55:60], 1),
                np.sum(ld[:, 60:], 1),
            )
        ).transpose()
        ld_4 = np.vstack(
            (np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))
        ).transpose()

        ld = torch.from_numpy(ld).float()
        ld_13 = torch.from_numpy(ld_13).float()
        ld_4 = torch.from_numpy(ld_4).float()

        if torch.cuda.is_available():
            ld = ld.cuda()
            ld_13 = ld_13.cuda()
            ld_4 = ld_4.cuda()

        return ld, ld_13, ld_4
