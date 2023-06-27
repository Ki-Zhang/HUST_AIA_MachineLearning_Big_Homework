import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange
import toml
from box import Box
import my_dataset as ds
from torchvision.models import resnet50, ResNet50_Weights
import timm
import os


class FSCNN(nn.Module):
    """
    嵌入网络
    """
    def __init__(self, embedding_dim=128) -> None:
        super().__init__()

        weights = ResNet50_Weights
        self.backbone = resnet50(weights=weights)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features=in_features,
                                     out_features=embedding_dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x: torch.Tensor[b, c, w, h], 输入图像
        Returns:
            torch.Tensor, 嵌入特征
        """
        x = self.backbone(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x


def train_FSCNN(epochs, learning_rate, train_loader, embedding_dim=128, cfg=None):
    """
    训练FSCNN模型
    Args:
        epochs: int, 训练轮数
        learning_rate: float, 学习率
        train_loader: DataLoader, 训练集
        embedding_dim: int, 嵌入维度
        cfg: Box, 配置文件

    Returns:
        None
    """
    DEVICE = cfg.device
    model = FSCNN(embedding_dim=embedding_dim).to(DEVICE)  # 嵌入网络
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    loss_fun = nn.MSELoss()  # 损失函数
    model.train()  # 训练模式
    for i in range(epochs):
        for batch, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(DEVICE)  # 锚点样本
            positive = positive.to(DEVICE)  # 正类样本
            negative = negative.to(DEVICE)  # 负类样本
            anchor_embedding = model(anchor)  # 锚点特征
            positive_embedding = model(positive)  # 正类特征
            negative_embedding = model(negative)  # 负类特征
            pos_sim = nn.functional.cosine_similarity(anchor_embedding,
                                                      positive_embedding)  # 锚点与正类的相似度
            neg_sim = nn.functional.cosine_similarity(anchor_embedding,
                                                      negative_embedding)  # 锚点与负类的相似度
            label = torch.concat(
                [torch.ones(pos_sim.shape),
                 torch.zeros(neg_sim.shape)]).to(DEVICE)  # 生成ground truth, 正类和锚点相似度期望为1, 负类和锚点相似度期望为0
            similarity = torch.concat([pos_sim, neg_sim])
            loss = loss_fun(similarity, label)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print("Epoch: {}, Batch: {}, Loss: {}".format(i, batch, loss.item()))
    torch.save(model.state_dict(), cfg.checkpoint)  # 保存模型


if __name__ == "__main__":
    cfg = Box(toml.load('./config.toml'))  # 配置文件
    cfg.checkpoint = os.path.join(cfg.checkpoint, "test1.pth")  # 模型保存路径
    DEVICE = torch.device(cfg.device)
    root = cfg.dataset.office31.root  # 数据集根目录
    post = cfg.dataset.office31.post  # 数据集后缀

    source_folder = os.path.join(root, 'amazon', post)  # amazon数据集路径
    target_folder1 = os.path.join(root, 'dslr', post)  # dslr数据集路径
    target_folder2 = os.path.join(root, 'webcam', post)  # webcam数据集路径

    emb_epochs = cfg.emb.epoch  # 嵌入网络训练轮数
    emb_batch_size = cfg.emb.batch_size  # 嵌入网络训练批次
    emb_learning_rate = cfg.emb.lr  # 嵌入网络学习率
    source_set = ds.source_domain_dataset(source_folder)  # amazon数据集
    source_set_loader = DataLoader(source_set,
                                   batch_size=emb_batch_size,
                                   num_workers=4,
                                   shuffle=True)  # amazon数据集加载器
    train_FSCNN(epochs=emb_epochs,
                learning_rate=emb_learning_rate,
                train_loader=source_set_loader,
                cfg=cfg)  # 训练嵌入网络
