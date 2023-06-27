import os
import random
from typing import Tuple, Any

import torch
from torchvision.transforms import Resize, AutoAugment, AutoAugmentPolicy, Compose
from torchvision.io import image
from torch.utils.data import Dataset, DataLoader


class source_domain_dataset(Dataset):
    def __init__(self, source_folder,
                 trans=Compose([AutoAugment(), Resize((224, 224))])) -> None:
        super().__init__()
        self.trans = trans
        # 使用字典存储每个数据的标签
        self.dict = {}
        # 得到所有类别
        source_labels = os.listdir(source_folder)
        for i in range(len(source_labels)):
            label_folder = os.path.join(source_folder, source_labels[i])
            # 得到每个类别每个数据的路径
            item_list = os.listdir(label_folder)
            for item in item_list:
                item = os.path.join(label_folder, item)
                # 0~30给每一类数据进行标记
                self.dict[item] = i
        self.key_list = list(self.dict.keys())

    def __getitem__(self, index) -> Tuple:
        idx_range = range(len(self.key_list))
        anchor_path = self.key_list[index]
        positive_path = self.key_list[random.choice(idx_range)]
        negative_path = self.key_list[random.choice(idx_range)]
        while (self.dict[anchor_path] != self.dict[positive_path]
               or anchor_path == positive_path):
            positive_path = self.key_list[random.choice(idx_range)]
        while self.dict[anchor_path] == self.dict[negative_path]:
            negative_path = self.key_list[random.choice(idx_range)]

        anchor_img = image.read_image(anchor_path)
        positive_img = image.read_image(positive_path)
        negative_img = image.read_image(negative_path)

        # trans = Resize((224, 224))
        trans = self.trans
        anchor_img = trans(anchor_img).float()
        positive_img = trans(positive_img).float()
        negative_img = trans(negative_img).float()

        return anchor_img, positive_img, negative_img
        # return anchor_path, positive_path, negative_path

    def __len__(self):
        return len(self.key_list)


class target_domain_support_dataset(Dataset):
    def __init__(self, target_folder, num_shots=3,
                 trans=Compose([AutoAugment(), Resize((224, 224))])) -> None:
        super().__init__()
        self.trans = trans
        # 使用字典存储每个数据的标签
        self.dict = {}
        # 得到所有类别
        target_labels = os.listdir(target_folder)
        for i in range(len(target_labels)):
            label_folder = os.path.join(target_folder, target_labels[i])
            # 得到每个类别每个数据的路径
            item_list = os.listdir(label_folder)
            for j in range(num_shots):
                item = os.path.join(label_folder, item_list[j])
                # 0~30给每一类数据的前3shot进行标记
                self.dict[item] = i
        self.key_list = list(self.dict.keys())

    def __getitem__(self, index) -> tuple:
        feature_path = self.key_list[index]
        label = self.dict[feature_path]
        feature = image.read_image(feature_path)

        trans = self.trans
        feature = trans(feature).float()

        return feature, label

    def __len__(self):
        return len(self.key_list)


class target_domain_query_dataset(Dataset):
    def __init__(self, target_folder, num_shots=3,
                 trans=Compose([AutoAugment(), Resize((224, 224))])) -> None:
        super().__init__()
        self.trans = trans
        # 使用字典存储每个数据的标签
        self.dict = {}
        # 得到所有类别
        target_labels = os.listdir(target_folder)
        for i in range(len(target_labels)):
            label_folder = os.path.join(target_folder, target_labels[i])
            # 得到每个类别每个数据的路径
            item_list = os.listdir(label_folder)
            for j in range(num_shots, len(item_list)):
                item = os.path.join(label_folder, item_list[j])
                # 0~30给每一类数据的query数据标记
                self.dict[item] = i
        self.key_list = list(self.dict.keys())

    def __getitem__(self, index) -> tuple:
        feature_path = self.key_list[index]
        label = self.dict[feature_path]
        feature = image.read_image(feature_path)

        trans = self.trans
        feature = trans(feature).float()

        return feature, label

    def __len__(self):
        return len(self.key_list)


class target_query_dataset(Dataset):
    def __init__(self, query_dict,
                 trans=Compose([AutoAugment(), Resize((224, 224))])) -> None:
        super().__init__()
        self.dict = query_dict
        self.key_list = list(query_dict.keys())
        self.trans = trans

    def __getitem__(self, index):
        feature_path = self.key_list[index]
        label = self.dict[feature_path]
        feature = image.read_image(feature_path)

        trans = self.trans
        feature = trans(feature).float()

        return feature, label

    def __len__(self):
        return len(self.key_list)


class target_support_dataset(Dataset):
    def __init__(self, support_dict,
                 trans=Compose([AutoAugment(), Resize((224, 224))])) -> None:
        super().__init__()
        self.trans = trans
        self.dict = support_dict
        self.key_list = list(support_dict.keys())

    def __getitem__(self, index):
        feature_path = self.key_list[index]
        label = self.dict[feature_path]
        feature = image.read_image(feature_path)

        trans = self.trans
        feature = trans(feature).float()

        return feature, label

    def __len__(self):
        return len(self.key_list)


def split_support_query(target_folder, num_shots=3):
    support_dict = {}
    query_dict = {}
    target_labels = os.listdir(target_folder)
    for i in range(len(target_labels)):
        label_folder = os.path.join(target_folder, target_labels[i])
        item_list = os.listdir(label_folder)
        support_list = random.sample(item_list, num_shots)
        query_list = [item for item in item_list if item not in support_list]
        for item in support_list:
            item = os.path.join(label_folder, item)
            support_dict[item] = i
        for item in query_list:
            item = os.path.join(label_folder, item)
            query_dict[item] = i
    return support_dict, query_dict


if __name__ == "__main__":
    curr_path = os.getcwd()
    source_path = "office\\amazon\\images"
    # source_path = "office31\\dslr\\images"
    # source_path = "office31\\webcam\\images"
    source_folder = os.path.join(curr_path, source_path)
    source_labels = os.listdir(source_folder)
    source_dict = {}
    for i in range(len(source_labels)):
        label_folder = os.path.join(source_folder, source_labels[i])
        item_list = os.listdir(label_folder)
        for item in item_list:
            item = os.path.join(label_folder, item)
            source_dict[item] = i
    key_list = list(source_dict.keys())
    print(len(key_list))

    source_set = source_domain_dataset(source_folder)
    source_set_loader = DataLoader(source_set, batch_size=32, shuffle=True, num_workers=4)
    source_set_loader = DataLoader(source_set, batch_size=32, shuffle=True, num_workers=4)
    anchor, positive, negative = source_set[708]

    print(anchor.shape)
    print(positive.shape)
    print(negative.shape)
