# Substation-Defect-Detection-based-on-Self-supervised-Learning
![image](https://github.com/L-iffer/Substation-Defect-Detection-based-on-Self-supervised-Learning/blob/master/images/SimSiamPro.png)
自监督预训练
------
仅支持多卡（多进程）分布式训练，不支持单卡（单进程）多线程训练。在COCO数据集上进行自监督预训练，运行脚本如下：

```
python main.py \  
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  [your COCO-folder with train folders]
```
存放自监督预训练数据集的目录结构如下：

```
datasets/
  COCO2017/
    train/
      n0/
        0001.jpg
        0002.jpg
        ...
      n1/
        0003.jpg
        0004.jpg
        ...
      ...
```
下游目标检测
------
首先，需要安装[detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)。

__预测：__

```
cd detection \
  python demo.py \
  --config-file configs/voc.yaml \
  --input images/input1.jpg \
  --output output/
  --opts MODEL.WEIGHTS models/voc_inference.pth
```
在预测的脚本文件中：  
* --config-file：在缺陷图像上预测使用configs/defect.yaml，默认在Pascal VOC图像上进行预测。
* --input：待预测的图像。
* --output：保存预测结果。
* MODEL.WEIGHTS：指定模型权重文件，在缺陷图像上预测使用models/defect_inference.pth，默认在Pascal VOC图像上进行预测。
* 使用cpu进行预测，在 --opts 后面添加 MODEL.DEVICE cpu，默认使用cuda加速。
