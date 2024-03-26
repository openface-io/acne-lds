# Improving acne image grading with Label Distribution Smoothing
- Link to paper [Improving Acne Image Grading with Label Distribution Smoothing](https://arxiv.org/abs/2403.00268)
- For validation of our approach ACNE04 dataset was used. One can find a link to the data in the following [repository](https://github.com/xpwu95/LDL).
- In [Google Drive folder](https://drive.google.com/drive/folders/1yCQfosewm5MctzbrCdbVNFiM9NFo80UL?usp=sharing) one can find weights for our model (`lds-weights` folder) and for LDL model from [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf) (`ldl-weights` folder)

# About/Overview
Proposed method tries to improve Acne Imgage Grading by generating more informative label distributions for lesion counting by incorporate information about grading, while simultaneously improving the performance of direct grading by converting the severity scale into simpler class definitions.

This approach can be viewed as a combination of _Label Distribution Learning (LDL)_ and _label smoothing_, where we smooth each hard label in counting task with the Gaussian label distribution based on its proximity to the class border. For the classification
branch, we reduce the complexity of the task by converting Hayashi-defined grade ranges into evenly-sized classes.

# Dependencies
- Python 3.8
- Pytorch 1.10.1
- Pytorch Lightning 1.5.10

Full list of dependencies can be found in `requirements.txt`

# Training
- To extract files from downloaded archives, one can move to their directory and run ```tar -xvf Classification.tar``` and ```tar -xvf Detection.tar``` sequentially.
- To change the data path one need to modify the config file `configs/path/path_data.yaml`

- To start training one can simply run

  ```
  python train.py
  ```
- To run Label Distribution Learning method described in [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf)], one should use
  ```
  python train.py train_val_params.model_type=model_ldl train_val_params.num_acne_cls=4 train_val_params.lam=0.6
  ```

# Inference
- To make a prediction on any part of ACNE04 data (either train or validation parts), one can run
  ```
  python predict.py path_checkpoint=CHECK_PATH.pth path_images=IMG_FOLDER path_images_metadata=IMG_META.txt
  ```
  Script creates ```.csv``` file with predicted severity level and number of acne for every image.

  One can find more settings in ```configs/predict/default.yaml```
- For prediction on single image the following example can be useful:
  ```python 
  from predict_on_raw_img import ModelInit
  from PIL import Image
  
  model = ModelInit(path_checkpoint=CHECKPOINT_PATH)
  img = Image.open(PATH_TO_IMAGE)
  predictions = model.predict_on_raw_img(img)
  ```

# Citation
```
@article{prokhorov2024improving,
  title={Improving Acne Image Grading with Label Distribution Smoothing},
  author={Prokhorov, Kirill and Kalinin, Alexandr A},
  journal={arXiv preprint arXiv:2403.00268},
  year={2024}
}
```
