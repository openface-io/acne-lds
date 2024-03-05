"""Train script that runs training process."""

from dataset.acne_data_module import AcneDataModule
from pytorch_lightning import Trainer
import wandb
from pytorch_lightning.loggers import WandbLogger
from utils.utils import seed_everything
import importlib
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(cfg_train: DictConfig):
    """Define main function."""
    # fix random seed
    seed_everything(seed=42)

    # Create loaders for the model
    acne_module = AcneDataModule(
        cfg_train.path.train_file,
        cfg_train.path.val_file,
        cfg_train.path.data_path,
        cfg_train.train_val_params.batch_size,
        cfg_train.train_val_params.batch_size_val,
    )

    train_loader, val_loader = acne_module.create_loaders()
    # Import model
    model_module = importlib.import_module("model." + cfg_train.train_val_params.model_type)
    main_model = model_module.AcneModel(cfg_train)

    # Init logger
    if cfg_train.trainer.logger == "wandb":
        wandb.login()
        wandb.init(project="acne-ldl")
        # Wandb logger for visializing of different metrics, losses etc.
        wandb_logger = WandbLogger(project="acne-ldl")
        logger = wandb_logger
    else:
        logger = None

    trainer = Trainer(
        max_epochs=cfg_train.trainer.max_epochs,
        logger=logger,
        devices=cfg_train.trainer.devices,
        accelerator=cfg_train.trainer.accelerator,
    )

    trainer.fit(main_model, train_loader, val_loader)

    # Finish the run
    wandb.finish()


if __name__ == "__main__":
    main()
