"""Script that runs prediction process on ACNE04 images."""

import hydra
from omegaconf import DictConfig
from transforms.acne_transforms import AcneTransformsTorch
from torch.utils.data import DataLoader
from dataset.acne_dataset import AcneDataset
import torch
from model.resnet50 import resnet50
import pandas as pd


@hydra.main(config_path="configs/predict", config_name="default")
def main(config: DictConfig):
    """Define main function."""
    # Create model
    num_acne_cls = 13 if config.model_type == "model_ld_smoothing" else 4
    model = resnet50(num_acne_cls=num_acne_cls)
    # load checkpoint
    checkpoint = torch.load(config.path_checkpoint, map_location=torch.device(config.device))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create dataset and dataloader
    dset_test = AcneDataset(
        config.path_images, config.path_images_metadata, transform=AcneTransformsTorch(train=False)
    )
    test_loader = DataLoader(dset_test, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # make a prediction
    cls_test = torch.tensor([], dtype=torch.int32)
    cnt_test = torch.tensor([], dtype=torch.int32)
    model.eval()
    for step, (b_x, b_y, b_l) in enumerate(test_loader):
        print(f"Step {step}")
        cls, cou, cou2cls = model(b_x)
        # Convert predictions back to Hayashi scale if needed
        if config.model_type == "model_ld_smoothing":
            cls = torch.stack(
                (
                    torch.sum(cls[:, :1], 1),
                    torch.sum(cls[:, 1:4], 1),
                    torch.sum(cls[:, 4:10], 1),
                    torch.sum(cls[:, 10:], 1),
                ),
                1,
            )
        preds_cls = torch.argmax(0.5 * (cls + cou2cls), dim=1)
        preds_cnt = torch.argmax(cou, dim=1) + torch.tensor(1)
        # accumulate predictions
        cls_test = torch.cat((cls_test, preds_cls))
        cnt_test = torch.cat((cnt_test, preds_cnt))

    # save predictions to .csv file
    if config.save_preds:
        df = pd.DataFrame(data={"severity_class": cls_test, "num_acne": cnt_test})
        df.to_csv("predictions.csv")


if __name__ == "__main__":
    main()
