from proto import ProtoNet
from config import get_cfg
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import functional as F


def pretrain(model,
             feature, cls,
             loader,
             cfg=None):
    """
    有监督预训练
    Args:
        model:
        feature:
        cls:
        loader:
        cfg:

    Returns:

    """
    if cfg is None:
        cfg = get_cfg()
    model.to(cfg.device)
    model.train()
    fc = nn.Linear(feature, cls).to(cfg.device)
    criterion = F.cross_entropy
    opt = Adam(model.parameters(), lr=cfg.lr)
    with tqdm(total=cfg.epoch * len(loader)) as pbar:
        for epoch in range(cfg.epoch):
            for step, (x, y) in enumerate(loader):
                x, y = x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                out = model(x)
                out = fc(out)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                pbar.set_description(f'epoch:{epoch}, step:{step}, loss:{loss.item()}')
                pbar.update(1)
    return model
