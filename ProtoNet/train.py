from box import Box
from backbone import TS_ResNet50
from copy import deepcopy
from proto import ProtoNet, meta_loss, finetune_loss, get_acc
from config import get_cfg
from sampler import DataManager4Office31
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


def meta_train(proto: ProtoNet,
               loader: DataLoader,
               cfg: Box = None):
    """
    元学习训练
    Args:
        proto: 原型网络
        loader: 数据加载器
        cfg: 配置

    Returns:
    """
    if cfg is None:
        cfg = get_cfg()
    # proto.to_device(cfg.device)
    # proto.to_mode('train')
    opt = Adam(proto.parameters(), lr=cfg.lr)
    with tqdm(total=cfg.n_step * cfg.episode) as pbar:
        for task, (supp_x, query_x, y) in enumerate(loader):
            for step in range(cfg.n_step):
                supp_x, query_x, y = supp_x.to(cfg.device), query_x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                lpy = proto(supp_x, query_x)
                loss = meta_loss(supp_x, query_x, lpy)
                acc = get_acc(supp_x, query_x, lpy)
                loss.backward()
                opt.step()
                pbar.set_description(f'task:{task}, step:{step}, loss:{loss.item()}, acc:{acc.item()}')
                pbar.update(1)


def fine_tune(proto: ProtoNet,
              loader: DataLoader,
              cfg: Box = None):
    """
    微调
    Args:
        proto: 原型网络
        loader: 数据加载器
        cfg: 配置

    Returns:

    """
    if cfg is None:
        cfg = get_cfg()
    # proto.to_device(cfg.device)
    # proto.to_mode('train')
    opt = Adam(proto.parameters(), lr=cfg.lr)
    with tqdm(total=cfg.episode * cfg.n_step) as pbar:
        for task, (supp_x, query_x, y) in enumerate(loader):
            for step in range(cfg.n_step):
                supp_x, query_x, y = supp_x.to(cfg.device), query_x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                lpy = proto(supp_x, query_x)
                loss = finetune_loss(supp_x, query_x, lpy)
                acc = get_acc(supp_x, query_x, lpy)
                loss.backward()
                opt.step()
                pbar.set_description(f'task:{task}, step:{step}, loss:{loss.item()}, acc:{acc.item()}')
                pbar.update(1)


def test(proto: ProtoNet,
         loader: DataLoader,
         cfg: Box = None):
    """
    测试,计算平均准确率
    Args:
        proto: 原型网络
        loader: 数据加载器
        cfg: 配置

    Returns:

    """
    if cfg is None:
        cfg = get_cfg()
    # proto.to_device(cfg.device)
    # proto.to_mode('eval')
    total_acc = 0
    for task, (supp_x, query_x, y) in tqdm(enumerate(loader)):
        with torch.no_grad():
            supp_x, query_x, y = supp_x.to(cfg.device), query_x.to(cfg.device), y.to(cfg.device)
            lpy = proto(supp_x, query_x)
            acc = get_acc(supp_x, query_x, lpy)
            total_acc += acc
    print(f'Average accuracy: {total_acc / len(loader)}')
    return total_acc / len(loader)


if __name__ == "__main__":
    mgr = DataManager4Office31("/root/autodl-tmp/office_31")
    cfg = get_cfg()
    
    def train_bench(mgr, cfg, src_pretrain_path:str, tar:str, save_path:str):
        """
        辅助函数
        Args:
            mgr: 数据管理器
            cfg: 配置
            src_pretrain_path: {src}.pth
            tar: target in ["amazon", "dslr", "webcam"]
            save_path: {s_t}.pth

        Returns:
            None
        """

        finetune_loader, test_loader = mgr.get_finetune_and_test_loader(tar)

        foo = TS_ResNet50()
        foo.load_state_dict(torch.load(src_pretrain_path, map_location='cuda:0'))
        foo = deepcopy(foo.backbone)

        proto = ProtoNet(foo)
        proto = proto.to(cfg.device)

        acc1 = test(proto, test_loader)

        meta_train(proto, finetune_loader)
        acc2 = test(proto, test_loader)

        # fine_tune(proto, finetune_loader)
        # test(proto, test_loader)
        with open("result.txt", "a") as f:
            f.write(f'{acc1}, {acc2}\n')

        torch.save(proto.state_dict(), save_path)
    
    train_bench(mgr, cfg, "amazon.pth", "dslr", "a_d.pth")
    train_bench(mgr, cfg, "amazon.pth", "webcam", "a_w.pth")
    train_bench(mgr, cfg, "dslr.pth", "amazon", "d_a.pth")
    train_bench(mgr, cfg, "dslr.pth", "webcam", "d_w.pth")
    train_bench(mgr, cfg, "webcam.pth", "dslr", "w_d.pth")
    train_bench(mgr, cfg, "webcam.pth", "amazon", "w_a.pth")