# Improving acne image grading with Label Distribution Smoothing
For validation of our approach ACNE04 dataset was used [link](https://github.com/xpwu95/LDL).

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



# Citation
