# Improving acne image grading with Label Distribution Smoothing

This is a PyTorch implementation of our method that improves acne severity grading from facial images by extending the [previously existing approach](https://github.com/xpwu95/LDL) based on label distribution learning.
We made two improvements: (1) generated more informative label distributions for lesion counting by incorporating information about the grading scale, while (2) simultaneously improving the performance of direct image grading by converting the severity grades into simpler class definitions.

More generally, this approach can be viewed as a combination of _Label Distribution Learning_ and _Label Smoothing_ for count-based classification, where we smooth each hard count label with the Gaussian label distribution based on its proximity to the class border.

If you find our work useful, please [cite our paper](#citation).

## Quick links
- Link to the paper [Improving Acne Image Grading with Label Distribution Smoothing](https://arxiv.org/abs/2403.00268)
- ACNE04 dataset used in the paper [repository](https://github.com/xpwu95/LDL)
- Pre-trained weights both for our model (`lds-weights`) and for the [LDL baseline model](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf) (`ldl-weights`) are available in [Google Drive](https://drive.google.com/drive/folders/1yCQfosewm5MctzbrCdbVNFiM9NFo80UL?usp=sharing)

## Dependencies
- Python 3.8+
- Pytorch 1.10.1
- Pytorch Lightning 1.5.10

Full list of dependencies can be found in `requirements.txt`

## Training
- Download the ACNE04 dataset and unpack by running `tar -xvf Classification.tar` and `tar -xvf Detection.tar`.
- To change the data path, modify the config file `configs/path/path_data.yaml`

- To train our LDS model:

  ```
  python train.py
  ```
- To train the [baseline LDL model](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf):
  ```
  python train.py train_val_params.model_type=model_ldl train_val_params.num_acne_cls=4 train_val_params.lam=0.6
  ```

## Inference
- To make a prediction on any part of ACNE04 data (either train or validation), run
  ```
  python predict.py path_checkpoint=CHECK_PATH.pth path_images=IMG_FOLDER path_images_metadata=IMG_META.txt
  ```
  This script outputs `.csv` file with predicted severity level and number of acne for every image.

  More settings can be changed in `configs/predict/default.yaml`
- For prediction on single image the following example can be useful:
  ```python
  from predict_on_img import ModelInit
  from PIL import Image

  model = ModelInit(path_checkpoint=CHECKPOINT_PATH)
  img = Image.open(PATH_TO_IMAGE)
  predictions = model.predict_on_img(img)
  ```

## Citation
```
@inproceedings{prokhorov2024improving,
  title={Improving Acne Image Grading with Label Distribution Smoothing},
  author={Prokhorov, Kirill and Kalinin, Alexandr A},
  booktitle={2024 IEEE 21th International Symposium on Biomedical Imaging (ISBI)},
  year={2024},
  organization={IEEE}
}
```
