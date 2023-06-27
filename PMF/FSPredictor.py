import torch
from torch import nn
from torch.utils.data import DataLoader
import my_dataset as ds
from FSEmbNet import FSCNN
import os
import numpy as np
from tqdm import trange
import tqdm
from torchvision.transforms import Resize
import toml
from box import Box


class FSPredictor(nn.Module):
    def __init__(self, support_set, saved=None, cfg=None) -> None:
        """
        小样本分类器
        Args:
            support_set: ds.FSData, 支撑集
            saved: 是否加载预训练嵌入层网络
            cfg: Box, 配置文件
        """
        super().__init__()
        self.EmbNet = FSCNN()
        if saved is not None:
            state_dict = torch.load(cfg.checkpoint)
            self.EmbNet.load_state_dict(state_dict)

        mat = []
        support_loader = DataLoader(support_set,
                                    batch_size=cfg.predict.batch_size,
                                    shuffle=False)
        for batch, (feature, label) in enumerate(support_loader):
            class_embs = self.EmbNet(feature)
            this_emb = torch.mean(class_embs, dim=0)
            mat.append(this_emb.detach().numpy())
        mat = np.array(mat, dtype="float32")   # n个类别的平均特征[n, d]
        mat = torch.tensor(mat)  # [n, d]
        self.mat = nn.parameter.Parameter(mat, requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.zeros(mat.shape[0]),
                                           requires_grad=True)

    def forward(self, x):
        """
        前向传播
        Args:
            x: torch.Tensor[b, c, w, h], 输入图像

        Returns:
            torch.Tensor, 预测结果
        """
        x = self.EmbNet(x)
        normed_mat = self.mat / torch.norm(self.mat, dim=1, keepdim=True)
        x = torch.matmul(x, normed_mat.T)
        x += self.bias
        return x


def fine_tune_FSPredictor(epochs,
                          learning_rate,
                          support_set,
                          use_entropy_regularization=True,
                          cfg=None
                          ):
    """
    微调FSPredictor
    Args:
        epochs: int, 训练轮数
        learning_rate: float, 学习率
        support_set: [n, c, w, h], 支撑集
        use_entropy_regularization: bool, 是否使用熵正则化
        cfg: Box, 配置文件
    Returns:
        None
    """
    model = FSPredictor(support_set=support_set, saved=True, cfg=cfg).to(cfg.device)  # 分类器
    support_loader = DataLoader(support_set,
                                batch_size=len(support_set),
                                shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    loss_fun = nn.CrossEntropyLoss()  # 损失函数
    model.train()  # 设置为训练模式
    for i in range(epochs):
        # 将整个support set一次性训练
        for batch, (feature, label) in enumerate(support_loader):
            feature = feature.to(cfg.device)  # [n, c, w, h]输入图像
            label = label.to(cfg.device)  # [n]标签
            pred = model(feature)  # [n, d]预测结果
            loss = loss_fun(pred, label)  # 计算损失
            if use_entropy_regularization:
                logit = nn.functional.softmax(pred, dim=1)  # [n, d]预测结果
                reg = nn.functional.binary_cross_entropy(logit, logit)  # 计算熵
                loss += reg  # 加上熵正则化
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            tqdm.tqdm.write(f"epoch: {i + 1}/{epochs}, loss: {loss.item():.4f}")  # 打印损失
    print("fine tune over!")
    torch.save(model.state_dict(), cfg.predict_check)  # 保存模型


def test_model(support_set, test_set,
               cfg:Box = None
               ):
    """
    测试模型
    Args:
        support_set: 支持及
        test_set: 测试集
        cfg: Box 配置文件

    Returns:

    """
    model = FSPredictor(support_set=support_set, cfg=cfg)  # 分类器
    state_dict = torch.load(cfg.predict_check)  # 加载微调后的模型
    model.load_state_dict(state_dict)  # 加载模型参数
    model = model.to(DEVICE)  # 模型放到DEVICE上
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)  # 测试集加载器
    total_num = len(test_set)  # 测试集样本总数
    correct = 0  # 正确样本数
    model.eval()  # 设置为测试模式
    with torch.no_grad():
        for batch, (feature, label) in enumerate(test_loader):
            feature, label = feature.to(DEVICE), label.to(DEVICE)
            pred = model(feature)
            if pred.argmax(1) == label:
                print(f"test feature: {(batch + 1)}/{total_num}, CORRECT")
                correct += 1
            else:
                print(f"test feature: {(batch + 1)}/{total_num}, WRONG")
    print("test over!")
    print(
        f"total correct num: {correct}/{total_num}, accuracy: {(correct / total_num * 100.0):>0.1f}%"
    )
    return correct / total_num


if __name__ == "__main__":
    cfg = Box(toml.load('./config.toml'))
    cfg.checkpoint = os.path.join(cfg.checkpoint, "test1.pth")
    cfg.predict_check = os.path.join(cfg.predict_check, "test_1p.pth")
    DEVICE = torch.device(cfg.device)
    root = cfg.dataset.office31.root
    post = cfg.dataset.office31.post

    source_folder = os.path.join(root, 'amazon', post)
    target_folder1 = os.path.join(root, 'dslr', post)
    target_folder2 = os.path.join(root, 'webcam', post)

    support_dict, query_dict = ds.split_support_query(target_folder2)
    support_set = ds.target_support_dataset(support_dict, trans=Resize((224, 224)))
    test_set = ds.target_query_dataset(query_dict, trans=Resize((224, 224)))
    fine_tune_FSPredictor(epochs=cfg.predict.epoch,
                          learning_rate=cfg.predict.lr,
                          support_set=support_set,
                          use_entropy_regularization=cfg.predict.regulation,
                          cfg=cfg)
    test_model(support_set, test_set, cfg=cfg)
