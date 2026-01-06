# 基于SAM2微调的显著性目标检测

## 选题理由

个人研究方向为显著性目标检测，SAM2作为一个可以分割任意图像的视觉大模型在该领域可以展现出显著的效果，因此尝试使用微调来加深对个人研究方向的认识以及强化代码能力。

## 数据集

显著性目标检测数据集:

DUTS、ECSSD、HKU-IS、PACAL-S

```
dataset
├─ DUTS-TR
| |─ gt
│ └─ im
├─ DUTS-TE
│ |─ gt
│ └─ im
├─ ECSSD
| |─ gt
│ └─ im
├─ HKU-IS
│ |─ gt
│ └─ im
├─ PASCAL-S
│ |─ gt
│ └─ im
└─
```

## 模型架构

<div align=center>  <img src=".\result\model.png" width=100%></div>

采用多类数据集DUTS、ECSSD、HKU-IS、PACAL-S

当前采用适配器的方式来微调image_encoder，mask_decoder以及prompt_ecoder全量微调

采用CBAM模块精细化特征图

数据预处理采用随机旋转、随机翻转、缩放

损失函数采用BCE、DICE、SSIM

测试集评估采用Fmax、WeightedF、Emean、Emax、MAE

## 文件描述
/dataset 数据集

/checkpoints 预训练模型

/checkpoints_finetuned 微调训练模型

/oss_json 阿里云对象存储oss文件相关文件内容

/sam2 模型

requirement.txt: 本次实验所使用的依赖

dataset_download.py: 数据集、模型下载

adapter.py: 微调image_encoder

dataloader.py: 数据集处理

evaluator.py: 测试评估器

metric.py: 评估函数

finetuning.py: 训练

predict_finetuning.py: 预测

option.py: 参数配置

loss.py: 损失函数

app.py: UI展示

## 未来工作

代码优化（使用类、函数去使得代码更加可观，添加适量的注释）

UI界面更新


