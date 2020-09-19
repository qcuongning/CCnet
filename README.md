

## Instructions for Code (2019/08 version):
### Requirements

To install PyTorch==0.4.0 or 0.4.1, please refer to https://github.com/pytorch/pytorch#installation.   
4 x 12G GPUs (_e.g._ TITAN XP)  
Python 3.6   
gcc (GCC) 4.8.5  
CUDA 8.0  

### Compiling

Some parts of **InPlace-ABN** and **Criss-Cross Attention** have native CUDA implementations, which must be compiled with the following commands:
```bash
cd libs
sh build.sh
python build.py

cd ../cc_attention
sh build.sh
python build.py
``` 
The `build.sh` script assumes that the `nvcc` compiler is available in the current system search path.
The CUDA kernels are compiled for `sm_50`, `sm_52` and `sm_61` by default.
To change this (_e.g._ if you are using a Kepler GPU), please edit the `CUDA_GENCODE` variable in `build.sh`.

### Dataset and pretrained model

Plesae download cityscapes dataset and unzip the dataset into `YOUR_CS_PATH`.

Please download MIT imagenet pretrained [resnet101-imagenet.pth](http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth), and put it into `dataset` folder.

### Training and Evaluation
Training script.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --recurrence 2
``` 

【**Recommend**】You can also open the OHEM flag to reduce the performance gap between val and test set.
```bash
python train.py --data-dir ${YOUR_CS_PATH} --random-mirror --random-scale --restore-from ./dataset/resnet101-imagenet.pth --gpu 0,1,2,3 --learning-rate 1e-2 --input-size 769,769 --weight-decay 1e-4 --batch-size 8 --num-steps 60000 --recurrence 2 --ohem 1 --ohem-thres 0.7 --ohem-keep 100000
``` 

Evaluation script.
```bash
python evaluate.py --data-dir ${YOUR_CS_PATH} --restore-from snapshots/CS_scenes_60000.pth --gpu 0 --recurrence 2
``` 

All in one.
```bash
./run_local.sh YOUR_CS_PATH
``` 

### Models
We run CCNet with *R=1,2* three times on cityscape dataset separately and report the results in the following table.
Please note there exist some problems about the validation/testing set accuracy gap (1~2%). You need to run multiple times
to achieve a small gap or turn on OHEM flag. Turning on OHEM flag also can improve the performance on the val set. In general，
I recommend you use OHEM in training step.

We train all the models on fine training set and use the single scale for testing.
The trained model with **R=2 79.74**  can also achieve about **79.01** mIOU on **cityscape test set** with single scale testing (for saving time, we use the whole image as input).

| **R** | **mIOU on cityscape val set (single scale)**           | **Link** |
|:-------:|:---------------------:|:---------:|
| 1 | 77.31 & **77.91** & 76.89 | [77.91](https://drive.google.com/open?id=13j06I4e50T41j_2HQl4sksrLZihax94L) |
| 2 | **79.74** & 79.22 & 78.40 | [79.74](https://drive.google.com/open?id=1IxXm8qxKmfDPVRtT8uuDNEvSQsNVTfLC) |
| 2+OHEM | 78.67 & **80.00** & 79.83  | [80.00](https://drive.google.com/open?id=1eiX1Xf1o16DvQc3lkFRi4-Dk7IBVspUQ) |

## Acknowledgment
We thank NSFC, ARC DECRA DE190101315, ARC DP200100938, HUST-Horizon Computer Vision ResearchCenter, and IBM-ILLINOIS Center for Cognitive ComputingSystems Research (C3SR).

## Thanks to the Third Party Libs
Self-attention related methods:   
[Object Context Network](https://github.com/PkuRainBow/OCNet)    
[Dual Attention Network](https://github.com/junfu1115/DANet)   
Semantic segmentation toolboxs:   
[pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox)   
[semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)   
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
