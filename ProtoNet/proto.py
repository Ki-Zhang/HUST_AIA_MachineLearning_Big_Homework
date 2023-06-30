import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import ConvEncoder


def cosine_similarity(x, y):
    """
    余弦相似度矩阵
    Args:
        x: (N, D)
        y: (M, D)

    Returns:
        dist: (N, M) 距离矩阵
    """
    cos = nn.CosineSimilarity(dim=0)
    cos_sim = []
    for xi in x:
        cos_sim_i = []
        for yj in y:
            cos_sim_i.append(cos(xi, yj))
        cos_sim_i = torch.stack(cos_sim_i)
        cos_sim.append(cos_sim_i)
    cos_sim = torch.stack(cos_sim)
    return cos_sim  # (N, M)


def euclidean_dist_similarity(x, y):
    """
    欧式距离相似度矩阵
    Args:
        x: (N, D)
        y: (M, D)

    Returns:
        dist: (N, M) 距离矩阵
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return -torch.pow(x - y, 2).sum(2)  # N*M


def get_acc(supp_x, query_x, lpf):
    """
    计算准确率
    Args:
        supp_x: (K, N, ch, w, h), 支撑集
        query_x: (K, Q, ch, w, h), 查询集
        lpf: protonet计算出来的查询集log p(y=k|X), (K, Q, K)

    Returns:
        acc: 准确率
    """
    k = supp_x.shape[0]  # 类别数
    q = query_x.shape[1]  # 每个类别的样本数
    # 生成ground truth
    query_y = torch.arange(0, k, requires_grad=False)\
                   .view(k, 1)\
                   .expand(k, q)\
                   .reshape(k * q, ).cpu()
    pred = torch.argmax(lpf, dim=-1).cpu().numpy()  # 预测类别
    query_y = query_y.cpu().numpy()
    pred = np.array(pred).reshape(-1)
    query_y = np.array(query_y).reshape(-1)
    acc = np.mean((pred == query_y))  # 计算准确率
    return acc


def meta_loss(supp_x, query_x, log_p_y):
    """
    计算元学习损失
    Args:
        supp_x: (K, N, ch, w, h), 支撑集
        query_x: (K, Q, ch, w, h), 查询集
        log_p_y: protonet计算出来的查询集log p(y=k|X), (K, Q, K)

    Returns:
        loss: 元学习损失
    """
    kq, k = log_p_y.shape
    q = kq // k
    log_p_y = log_p_y.reshape(k, q, k)
    loss = torch.einsum('k q k->', -log_p_y) / (k * q)
    return loss


def finetune_loss(supp_x, query_x, log_p_y):
    """
    计算微调损失,复现P>M>F论文中的微调损失，但是本项目中域偏移没有过大，所以不需要微调
    Args:
        supp_x: (K, N, ch, w, h), 支撑集
        query_x: (K, Q, ch, w, h), 查询集
        log_p_y: (K, Q, K), protonet计算出来的查询集log p(y=k|X)

    Returns:
        loss: 微调损失
    """
    k = supp_x.shape[0]
    q = query_x.shape[1]
    query_y = torch.arange(0, k, requires_grad=False)\
                   .view(k, 1)\
                   .expand(k, q)\
                   .reshape(k * q, ).to(supp_x.device)
    output = log_p_y
    loss = F.cross_entropy(output, query_y)  # 计算交叉熵损失
    return loss


class ProtoNet(nn.Module):
    """
    原型网络
    """
    def __init__(self, backbone):
        super(ProtoNet, self).__init__()
        self.encoder = backbone  # 待训练的embedding

    def forward(self, supp_x, query_x):
        """
        计算log p(y=k|X)，前向计算
        Args:
            supp_x: (N, K, ch, w, h), 支撑集
            query_x: (N, Q, ch, w, h), 查询集

        Returns:
            log_p_y: (N, Q, K), log p(y=k|X)
        """
        supp_shape = supp_x.shape  # 支撑集形状

        # batch化方便向量化计算
        supp_x = supp_x.reshape(-1, *supp_shape[-3:])
        query_shape = query_x.shape
        query_x = query_x.reshape(-1, *query_shape[-3:])

        # 计算原型
        x_proto = self.encoder(supp_x)  # (n* k, embed_dim)
        x_proto = x_proto.reshape(*supp_shape[:-3], -1)  # (n, k, embed_dim)
        x_proto = x_proto.mean(1)  # (n, embed_dim)

        # 变回原来的形状
        x_q = self.encoder(query_x)  # (n* q, embed_dim)
        x_q = x_q.reshape(*query_shape[:-3], -1)  # (n, q, embed_dim)
        x_q = x_q.view(-1, x_q.shape[-1])  # (n*q, embed_dim)

        # 计算相似度
        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)
        log_p_y = F.log_softmax(sim_result, dim=1)
        return log_p_y  # (n*q, n)

    def get_backbone(self):
        return self.encoder

    @staticmethod
    def similarity(a, b, sim_type='cosine'):
        """
        计算相似度
        Args:
            a: (N, D)特征向量
            b: (M, D)特征向量
            sim_type:
                {'euclidean': 欧式距离相似度矩阵, 'cosine': 余弦相似度矩阵}
        Returns:
            sim: (N, M)相似度矩阵
        """
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高
