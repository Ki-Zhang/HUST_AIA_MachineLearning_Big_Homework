# HUST-AIA模式识别大课设
## Introduction
本仓库由华中科技大学人工智能与自动化学院5名大三本科生所作，用于保存人工智能与自动化学院于2023年下半学期开设的模式识别课设所写代码。本仓库所写代码均经过实践测试，可以将其下载下来配置好后即可运行。
本仓库主要围绕选题跨域小样本目标识别进行，小组成员均贡献出不同方向的代码，所以读者可以将代码下载下来后分别进行验证。如果本代码对你有帮助，麻烦给我们star一下以示支持！

![haoye](haoye.jpg)

## Data Preparation
本代码主要是在跨域小样本数据集Office31和Office-home上进行编写。
Office31数据集的包括三个域：非别是亚马逊商城(在线电商)图片，单反相机拍摄图片，网络摄像头拍摄图片，包括31个类。[[官网下载]](https://faculty.cc.gatech.edu/~judy/domainadapt/)
Office-home是域适应的一个基准数据集，包含4个不同域，每个域由65个类别组成，分别是艺术、剪贴画、产品和真实世界。它包含15500个图像，每个类平均约70个图像，一个类最多99个图像。[[官网下载]](https://www.hemanthdv.org/officeHomeDataset.html)
为了方便读者下载，我们提供[[百度网盘]](https://pan.baidu.com/s/1CkmSknfceYJQ5ls_G84pvg?pwd=PRML)的数据集下载方式

下载完成后，需要将数据集整理到data_set文件夹下，按照如下方式整理：
```
data_set
├── office-31
│   ├── amazon
|   |       └── images
|   |           ├── back_pack
|   |           |       └── frame_0001.jpg
|   |           |       └── frame_xxx.jpg
|   |           ├── xxx
|   |           |
|   |           └── trash_can
|   |                   └── frame_0001.jpg
|   |                   └── frame_xxx.jpg
|   ├── dslr
|   └── amazon
└── OfficeHome
    ├── Art
    |   ├── Alarm_Clock
    |   |       ├── 00001.jpg
    |   |       └── xxxxx.jpg
    |   ├── xxx
    |   └── Webcam
    ├── Clipart
    ├── Product
    ├── RealWorld
    ├── ImageInfo.csv
    └── Imagelist.txt
```
## Model
本仓库共有4种模型，接下来对该5种模型进行简单介绍，详细介绍请移步每个模型的文件夹。
### MAML
MAML(Model-Agnostic Meta-Learning)是一种元学习的模型，原论文可以共通过[[论文地址]](https://arxiv.org/pdf/1703.03400.pdf)进行获取。
其主要解决的是小样本以及模型手链速度太慢两个问题，由于其考虑两个数据集的分布问题，因而其可以实现在两个任务中达到全局最优，其核心算法如下所示：
![key_algorithm](maml/maml_algorithm.png)
### PMF
### Teacher-Student
### RSDA















