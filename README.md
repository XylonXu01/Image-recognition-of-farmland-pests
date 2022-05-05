# Image-recognition-of-farmland-pests
## 该项目使用目标检测方法实现农田害虫识别
## 该项目主要参考的源码

* <https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection>

## 环境配置：
* Python3.8
* Pytorch1.10.0(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── YOLOV5:YOLOv5训练测试代码
  ├── csv2xml:将数据集csv格式的标签转为PASCAL VOC格式的xml标签
  └── Faster R-CNN: Faster R-CNN训练代码
```
 
 
## 数据集
* 本例程使用的是PASCAL VOC2012数据集格式
* 项目数据集地址：链接: https://pan.baidu.com/s/16Koi2sTA3ZbaIjkjde0eag 提取码: q4cj 
* 使用ResNet50+FPN以及迁移学习在VOC2012数据集上得到的权重: 链接:<https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w> 提取码:dsz8

## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
* 可以更改backbone
* 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`(例如我只要使用设备中的第1块和第4块GPU设备)
* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`(VOC_root)设置为自己存放`VOCdevkit`文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 在使用预测脚本时，要将`train_weights`设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改`--num-classes`、`--data-path`和`--weights-path`即可，其他代码尽量不要改动



## Faster RCNN框架图
![Faster R-CNN](fasterRCNN.png) 
