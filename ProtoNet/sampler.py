import os
from copy import deepcopy
import torch
import torchvision.transforms as T
from box import Box
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import ImageFolder
import random
from einops import rearrange, pack
import pandas as pd
from torchvision.datasets.vision import StandardTransform 
from config import get_cfg


class DataManager4Office31(object):
    """
    数据管理器
    """
    def __init__(self, root, cfg: Box = None):
        self.root = root  # 数据集根目录
        self.domain = os.listdir(root)  # 数据集域
        self.domain.sort()
        self.dataset = {d: os.path.join(self.root, d, 'images') for d in self.domain}

        if cfg is None:
            cfg = get_cfg()
        self.cfg = cfg

    def get_train_loader(self, source: str, batch_size=64, num_workers=8,
                         transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
                         ):
        """
        获取训练数据加载器
        Args:
            source: 源域
            batch_size:
            num_workers: 线程数
            transform: 变换

        Returns:
            train_loader: 训练数据加载器
        """
        ds = ImageFolder(self.dataset[source], transform=transform)
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def get_finetune_and_test_loader(self, target,
                                     num_workers=8,
                                     transform=T.Compose([T.Resize((224, 224)), T.ToTensor()])
                                     ):
        """
        获取微调和测试数据加载器
        Args:
            target: 目标域
            num_workers: 线程数
            transform: 变换

        Returns:
            finetune_loader: 微调数据加载器
            test_loader: 测试数据加载器
        """
        ds = ImageFolder(self.dataset[target], transform=transform)
        epi = EpisodeSampler(k_way=self.cfg.k_way,
                             n_shot=self.cfg.n_shot,
                             query_shot=self.cfg.query_shot,
                             episode=self.cfg.episode,
                             ds=ds)
        collate_fn = EpisodeCollateFn(k_way=self.cfg.k_way,
                                      n_shot=self.cfg.n_shot,
                                      query_shot=self.cfg.query_shot)
        fds, tds = self.split(ds, epi)
        fds.transform = T.Compose([T.RandomResizedCrop(224, scale=(0.2, 1.)),
                                   T.RandomHorizontalFlip(),
                                   T.Resize([224, 224]),
                                   T.ToTensor()])
        fds.transforms = StandardTransform(fds.transform, fds.target_transform)
        fdl = DataLoader(fds, batch_sampler=epi, collate_fn=collate_fn, num_workers=num_workers)
        tdl = DataLoader(tds, batch_sampler=epi, collate_fn=collate_fn, num_workers=num_workers)
        return fdl, tdl

    @staticmethod
    def split(ds, epi, n=5):
        """
        划分数据集
        Args:
            ds: 数据集
            epi: 采样器
            n: 采样数，当frac为None时有效
        Returns:

        """
        db = epi.db
        fdb = db.groupby('label').apply(lambda x: x.sample(n=n)).reset_index(drop=True)
        tdb = db[~db['id'].isin(fdb['id'])].reset_index(drop=True)
        fds = deepcopy(ds)
        tds = deepcopy(ds)
        _ = fdb[['path', 'label']].values.tolist()
        fds.imgs = list(map(tuple, _))
        _ = tdb[['path', 'label']].values.tolist()
        tds.imgs = list(map(tuple, _))
        return fds, tds


class EpisodeSampler(Sampler):
    def __init__(self, k_way: int, n_shot: int, query_shot: int,
                 episode: int,
                 ds: ImageFolder):
        self.k_way = k_way
        self.n_shot = n_shot
        self.query_shot = query_shot

        self.cls_dict = ds.class_to_idx  # str -> int
        self.cls = ds.classes  # int -> str

        self.db = pd.DataFrame(ds.imgs, columns=['path', 'label'])
        self.db['id'] = list(range(len(ds.imgs)))
        self.db['class'] = self.db['label'].apply(lambda x: self.cls[x])

        self.episode = episode

    def __iter__(self):
        """
        采样器
        Returns:
            id : List[int]，K ways N shots
        """
        for _ in range(self.episode):
            k_cls = random.sample(range(len(self.cls)), self.k_way)
            sub_db = self.db[self.db['label'].isin(k_cls)]
            sub_db_sample = sub_db.groupby('label')\
                .apply(lambda x: x.sample(self.n_shot + self.query_shot))\
                .reset_index(drop=True)
            yield sub_db_sample['id'].tolist()

    def __len__(self):
        return self.episode


class EpisodeCollateFn(object):
    def __init__(self, k_way: int, n_shot: int, query_shot: int):
        self.k_way = k_way
        self.n_shot = n_shot
        self.query_shot = query_shot

    def __call__(self, data):
        """
        将datatloader给出的数据调整成episode的形状
        Args:
            data: dataloader给出的数据
        Returns:
            support_img: [k*n, ch, w, h]
            query_img: [k*q, ch, w, h]
            label: [k,]
        """
        x_list, y_list = list(zip(*data))
        x, _ = pack(x_list, '* ch w h')
        y = torch.tensor(y_list)
        kn = rearrange(x, '(k nq) ch w h->k nq ch w h',
                       k=self.k_way,
                       nq=self.n_shot + self.query_shot)
        kn_label = rearrange(y, '(k nq) ->k nq',
                             k=self.k_way,
                             nq=self.n_shot + self.query_shot)
        support_img = kn[:, 0:self.n_shot, :, :, :]
        label = kn_label[:, 0]
        query_img = kn[:, self.n_shot:, :, :, :]
        # support_img = rearrange(support_img, 'k n ch w h->(k n) ch w h')
        # query_img = rearrange(query_img, 'k q ch w h->(k q) ch w h')
        return support_img, query_img, label

