"""Model with description of the train/val steps."""

import torch
from utils.smooth_ldl import SmoothLDL
from model.base_model import BaseModel


class AcneModel(BaseModel):
    """
    Describe model, train/val steps.

    training_step - a step during training stage
    validation_step - a step during validation stage
    """

    def __init__(self, config):
        """Init of the model."""
        super().__init__(config=config)
        # Label Distribution smoothing object
        self.label_smooth = SmoothLDL(
            eps_min=self.config.train_val_params.eps_min,
            eps_max=1.0,
            sigma=self.config.train_val_params.sigma,
            eps_type="piecewise",
        )

    def training_step(self, batch, batch_idx):
        """Run training loop."""
        b_x, b_y, b_l = batch
        # ld - label distribution
        ld, ld_13, ld_4 = self.generate_ld(b_l.cpu().detach() - 1)
        # cls - class, cou - counting task, cou2cls - class converted from the
        # counting result (see paper)
        cls, cou, cou2cls = self.cnn(b_x)  # nn output

        # smooth labels
        b_l_smooth = self.label_smooth.smooth_labels(b_l.cpu().detach() - 1)

        loss_cls = self.kl_loss(torch.log(cls), ld_13) * 13.0
        loss_cou = self.kl_loss(torch.log(cou), b_l_smooth) * 65.0
        loss_cls_cou = self.kl_loss(torch.log(cou2cls), ld_4) * 4.0

        # Total loss
        loss = (loss_cls + loss_cls_cou) * 0.5 * self.config.train_val_params.lam + loss_cou * (
            1.0 - self.config.train_val_params.lam
        )

        # Convert predictions back to Hayashi scale
        prob_cls_4 = torch.stack(
            (
                torch.sum(cls[:, :1], 1),
                torch.sum(cls[:, 1:4], 1),
                torch.sum(cls[:, 4:10], 1),
                torch.sum(cls[:, 10:], 1),
            ),
            1,
        )

        preds_cls = torch.argmax(0.5 * (prob_cls_4 + cou2cls), dim=1)
        preds_cnt = torch.argmax(cou, dim=1) + torch.tensor(1)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train_loss_cou", loss_cou, on_step=False, on_epoch=True)
        self.log("train_loss_cou2cls", loss_cls_cou, on_step=False, on_epoch=True)

        return {"loss": loss, "preds_cls": preds_cls, "preds_cnt": preds_cnt, "b_y": b_y, "b_l": b_l}

    def validation_step(self, batch, batch_idx):
        """Run validation loop."""
        b_x, b_y, b_l = batch
        # ld - label distribution
        ld, ld_13, ld_4 = self.generate_ld(b_l.cpu().detach() - 1)
        # cls - class, cou - counting task, cou2cls - class converted from the
        # counting result (see paper)
        cls, cou, cou2cls = self.cnn(b_x)  # nn output

        # smooth labels
        b_l_smooth = self.label_smooth.smooth_labels(b_l.cpu().detach() - 1)

        loss_cls = self.kl_loss(torch.log(cls), ld_13) * 13.0
        loss_cou = self.kl_loss(torch.log(cou), b_l_smooth) * 65.0
        loss_cls_cou = self.kl_loss(torch.log(cou2cls), ld_4) * 4.0

        # Total loss
        loss = (loss_cls + loss_cls_cou) * 0.5 * self.config.train_val_params.lam + loss_cou * (
            1.0 - self.config.train_val_params.lam
        )

        # Convert predictions back to Hayashi scale
        prob_cls_4 = torch.stack(
            (
                torch.sum(cls[:, :1], 1),
                torch.sum(cls[:, 1:4], 1),
                torch.sum(cls[:, 4:10], 1),
                torch.sum(cls[:, 10:], 1),
            ),
            1,
        )

        preds_cls = torch.argmax(0.5 * (prob_cls_4 + cou2cls), dim=1)
        preds_cnt = torch.argmax(cou, dim=1) + torch.tensor(1)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("val_loss_cou", loss_cou, on_step=False, on_epoch=True)
        self.log("val_loss_cou2cls", loss_cls_cou, on_step=False, on_epoch=True)

        return {"loss": loss, "preds_cls": preds_cls, "preds_cnt": preds_cnt, "b_y": b_y, "b_l": b_l}
