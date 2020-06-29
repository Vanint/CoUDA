# CoUDA 
We provide the original implementation for “Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis (IEEE TIP 2020)”.

## Dependencies
Python 2.7\
Tensorflow-gpu 1.12.0\
Opencv-python\
Numpy 1.15.4

## Usage
Please use this code after downloading the dataset and the model.

### Dataset and Model
Dataset: The attached dataset is Office-31 corrupted by the label noise rate 0.1. \
Please download the dataset [[here](https://drive.google.com/file/d/1SBrPKQqpZfe1c2J9NDV3E9smLxKhgbNY/view?usp=sharing)] and put the dataset file (domain_adaptation_images) in the main directoty.

Model: Plase download the trained model [[here](https://drive.google.com/file/d/1BkaWHP2C0GQtz-Zu46kZIAe0Fav3VORO/view?usp=sharing)], and put the the model file (dual_log_office) in the main directory.

### Training
Take the model adapted from webcam to amazon as an example. There are two steps:
1. Set the environment file: vi src_office/conf/local_nn_dual.yml
2. CUDA_VISIBLE_DEVICES=0 python src_office/train_dual.py >> ./dual_log_office/ours/webcam_2_amazon.txt src_office/conf/local_mn_dual.yml

### Test
Take the model adapted from dslr to webcam as an example. There are two steps:
1. Set the environment file: vi src_office/conf/predict_dual.yml
2. CUDA_VISIBLE_DEVICES=0 python src_office/predict_dual.py >> ./dual_log_office/ours/dslr_2_webcam/test.txt src_office/conf/predict_dual.yml

## Citation:
If you use this code and dataset, please cite:
```
@article{zhang2020collaborative,
  title={Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis},
  author={Zhang, Yifan and Wei, Ying and Wu, Qingyao and Zhao, Peilin and Niu, Shuaicheng and Huang, Junzhou and Tan, Mingkui},
  journal={IEEE Transactions on Image Processing},
  year={2020}
}  
```
