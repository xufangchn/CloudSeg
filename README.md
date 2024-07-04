# CloudSeg: A Multi-Modal Learning Framework for Robust Land Cover Mapping under Cloudy Conditions

This repository contains the codes for the paper "CloudSeg: A Multi-Modal Learning Framework for Robust Land Cover Mapping under Cloudy Conditions" 


If you use the codes for your research, please cite us accordingly:

```
@article{xu2024cloudseg,
  title={CloudSeg: A multi-modal learning framework for robust land cover mapping under cloudy conditions},
  author={Xu, Fang and Shi, Yilei and Yang, Wen and Xia, Gui-Song and Zhu, Xiao Xiang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={214},
  pages={21--32},
  year={2024}
}
```

## Prerequisites & Installation

This code has been tested with CUDA 11.7 and Python 3.8.

```
conda create -n CloudSeg python=3.8
conda activate CloudSeg
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install rasterio tqdm timm six scikit-learn
pip install pretrainedmodels efficientnet_pytorch
```

## Get Started
You can download the pretrained model ([TeacherNet.pth](https://www.tobeupdated) & [StudentNet.pth](https://drive.google.com/file/d/1aOGPRHEEtDXI6i-v9VQ-1Jo_Shhkmgme/view?usp=sharing)) and put it in './checkpoints'.

Use the following command to test the network:
```
cd ./StudentNet
python test_SS.py
```
Use the following command to train the network:
```
'''
1. Train the Teacher network 
'''
cd ./TeacherNet
python train_SS.py

'''
2. Train the Student network 
'''
cd ./StudentNet
python train_SS.py
```

## Datasets
Our experiments are conducted on two benchmark datasets: [M3M-CR](https://github.com/zhu-xlab/M3R-CR) and [WHU-OPT-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset). The M3M-CR dataset features cloud-covered optical images derived from real remote sensing scenarios, while the WHU-OPT-SAR dataset does not include cloud-covered images corresponding to its cloud-free counterparts. We perform artificial cloud layer synthesis on the available cloud-free images within the WHU-OPT-SAR dataset to simulate the effect of cloud cover.

## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to xufang@whu.edu.cn
