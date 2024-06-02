# CloudSeg: A Multi-Modal Learning Framework for Robust Land Cover Mapping under Cloudy Conditions

This repository contains the codes for the paper "CloudSeg: A Multi-Modal Learning Framework for Robust Land Cover Mapping under Cloudy Conditions" 


If you use the codes for your research, please cite us accordingly:

```
@article{xu2024cloudseg,
  title={CloudSeg: A Multi-Modal Learning Framework for Robust Land Cover Mapping under Cloudy Conditions},
  author={Xu, Fang and Shi, Yilei and Yang, Wen and Xia, Gui-Song and Zhu, Xiao Xiang},
}
```

## Prerequisites & Installation

This code has been tested with CUDA 11.8 and Python 3.8.

```
conda create -n CloudSeg python=3.8
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install rasterio timm noise
pip install tqdm easydict thop easydict paramiko
pip install opencv-python matplotlib
pip install tensorboardX torchstat
pip install rtree kornia visdom einops fiona pyproj shapely scikit-learn
pip install efficientnet-pytorch
```

## Get Started
You can download the pretrained model (coming soon) and put it in './checkpoints'.

Use the following command to test the neural network:
```
python test_SS.py
```

## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to xufang@whu.edu.cn
