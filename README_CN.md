# GLSAN
GLSAN 是一个用于无人机视觉微小目标检测的网络。这个版本源于 dengsutao/glsan，并修正了一些错误。
## Installation
我们的源代码主要是基于 [Detectron2](https://github.com/facebookresearch/detectron2)，请查看 [Detectron2.installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)。
## Get Started
关于初始化 [Detectron2](https://github.com/facebookresearch/detectron2)，请参考 [Detectron2.Getting_started](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)。
### dataset transformation
如果准备训练 VisDrone 和 UAVDT 数据集，你需要首先把它们转换为 COCO 格式。
我们提供了 './tools/txt2xml_\*.py' 和 './tools/xml2json_\*.py' 以生成 COCO 格式的 json 文件。
### dataset augmentation
对应论文中的网络是使用增强数据集训练的。
我们提供了 './tools/crop_dataset.py' 和 './tools/sr_dataset.py' 来对原始数据集进行 SARSA（裁剪） 和 LSRN（超分辨率）操作。其中'./tools/sr_dataset.py'修改了原作者代码中的诸多错误，包括索引计数异常，错误的自身调用，图片索引错误，cfg文件多出一个非法字段等。
### pretrained models
我们网络的预训练模型可以在以下网站下载 [Detectron2.model_zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)。
你也可以通过点击 [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) 或者 [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl) 来直接下载到你 'home' 目录下的 '.torch/iopath_cache/detectron2/ImageNetPretrained/MSRA/' 。
Or they will be downloaded automatically when training.
### training
We provide "train_net.py" for network training.
To train a model with "train_net.py", first setup the corresponding datasets following [Detectron2.datasets](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
you need to put the transformed or augmented datasets into './datasets' directory.
The settings of VisDrone and UAVDT can be found in './glsan/data/datasets'.

To train with 8 GPUs, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --num-gpus 8
```


To train with 1 GPU, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2
```


### evaluation
However, please note that end-to-end cropping and super-resolution operations are only supported for the inference process, so please run crop_dataset.py and sr_dataset.py first for training.  
To evaluate a model's performance, there are threee modes corresponding to three different
cropping strategies: NoCrop, UniformlyCrop, SelfAdaptiveCrop.
You can run following codes to switch the cropping strategy:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP UniformlyCrop
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP SelfAdaptiveCrop
```
To add super-resolution operation to the network, run:
```python
python train_net.py --config-file ./configs/faster_rcnn_res50_visdrone.yaml --eval-only --num-gpus 8 GLSAN.CROP SelfAdaptiveCrop GLSAN.SR True
```

To acquire more parameters of our method, see './glsan/config/defaults.py' and './glsan/modeling/meta_arch/glsan.py'
