
## 代码说明
### 目录结构
```
project
│   README.md   
|
└───code
│   │
│   └───dataset
│   |
│   └───losses
│   |
│   └───metrics
│   |
│   └───networks
│   |
│   └───pretrained
│   |
│   └───runs
│   |
│   └───utils
│   |
│   └───train.py
│   |
│   └───test.py
│   
└───Data
│   │
│   └───imgs
│   |
│   └───masks
│
└───Document
│   │
│   └───面部分割说明.pdf

  
```
### 说明

- code 目录
  
  训练相关代码
    - datasets 数据集读取以及处理
    - losses 损失函数定义（包括DiceLoss、FocalLoss）
    - metrics 评价指标（F1、acc、mIoU）
    - networks 网络模型（Unet、UnetVGG、Unet++）
    - utils 工具函数目录
    - train.py 训练
    - test.py 测试
  
- data 目录
	
	存放数据集的位置
	 - imgs 原图
	 - masks 标注
  
- model
	训练好的模型参数
	- Unet++.pth
	- Unet.pth

- document
	项目文件
  
