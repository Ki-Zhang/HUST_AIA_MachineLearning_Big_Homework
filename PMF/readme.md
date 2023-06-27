# README.md
PMF代码实现
### FSEmbNet
特征提取网络，将数据映射为128-d特征向量。

backbone：ResNet-50，将fc层输出维度修改为128，冻结fc层以外的所有权重。在源域使用孪生网络对网络进行训练

### FSPredictor
在FSEmbNet基础上增加分类头，使用softmax分类器。

FSPredictor使用目标域的Support Set进行微调

### my_dataset
自行实现的Dataset和Dataloader，用于转载Office-31数据集

### test
测试网络性能

### config
配置网络参数

### benchmark
a-d、a-w、d-a、d-w、w-a、w-d共计6个跨域任务的性能总测试
