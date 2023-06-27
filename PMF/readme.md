# README.md
PMF代码实现
### FSEmbNet
特征提取网络，将数据映射为128-d特征向量。

backbone：ResNet-50，将fc层输出维度修改为128，冻结fc层以外的所有权重。使用孪生网络对网络进行训练
