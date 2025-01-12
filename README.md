# 工作报告

## 仓库结构

```
.
├── 工作报告.pptx
├── code_adv
│   ├── adv_rec_aliyun.py   //视觉模型接口
│   ├── detect.png
│   ├── detect_2d
│   ├── images              //传入图片存放路径
│   └── requirements.txt    //视觉模型依赖库
├── yolo_adv
│   ├── __pycache__
│   ├── best.pt             //最佳权重文件
│   ├── ceshi.py
│   ├── classify
│   ├── data                //所需检测的数据存放在该路径下的images中
│   ├── detect.py           //执行检测文件
│   ├── export.py
│   ├── images              //训练图像数据集
│   ├── labels              //训练标签数据集
│   ├── models
│   ├── mydata              //原训练数据存放文件夹
│   ├── pyproject.toml
│   ├── requirements.txt    //预测模型依赖库
│   ├── runs                //运行结果保存
│   ├── segment 
│   ├── train.py            //训练文件
│   ├── utils
│   ├── val.py
│   └── yolov5s.pt          //默认权重文件
└── README.md               //工作总体介绍 
```
#### *具体文件含义可参照以上结构说明

## 原工作环境
在Anaconda上构建了虚拟环境进行本地调试，后在kaggle上进行正式训练。

## 运行说明
对于code_adv与yolo_adv，在运行前都需要安装相应的依赖：
```
pip install -r requirements.txt
```
### yolo检测：
#### 文件夹：yolo_adv

文件已经训练完成，为了验证训练，只在训练文件夹中放了少量的数据，执行train.py即可；

对于检测识别，将需要处理的图片放置在data/images中即可，运行detect.py文件，可在runs/detect/exp*中得到结果。


### 大模型检测：
#### 文件夹：code_adv
这里用的是阿里云的大模型，文件连接需要key，这里已经进行了填充。运行adv_rec_aliyun.py以传入输入并进行识别（传入图片在images中），等待返回结果。

#### *更详细的展示在ppt中可见


