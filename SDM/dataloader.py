import copy
from itertools import groupby
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from PIL import ImageFilter
import random
from random import sample
import os


class TwoCropsTransform:

    def __init__(self, base_transform, strong_transform=None):
        self.base_transform = base_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        base = self.base_transform(x)
        strong = self.strong_transform(x)
        return [base, strong]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Office:
    def __init__(self,root_dir,dataset) -> None:
        self.root_dir = root_dir
        self.dataset = dataset
        self.domain = os.listdir(root_dir)
        self.domain.sort()
        
        data = datasets.ImageFolder(os.path.join(self.root_dir, self.domain[0]))
        self.classes = data.classes
        self.class_to_idx = data.class_to_idx
        
    def get_dual_train_loader(self, source, batch_size=30, num_workers=8, weak=None, strong=None):
        traindir = os.path.join(self.root_dir, source)
        if source=='amazon':
            num = 20
        elif source=='dslr' or source=='webcam':
            num = 8
        else:
            num = None
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        weak_default = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize,
        ])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        strong_default = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
        
        if weak is None:
            weak = weak_default
            
        if strong is None:
            strong = strong_default
        
        train_dataset = datasets.ImageFolder(
                            traindir,
                            TwoCropsTransform(weak, strong))
        
        if num is not None:
            group = groupby(copy.deepcopy(train_dataset.imgs), key=lambda x: x[1])
            train_dataset.imgs.clear()
            for key, items in group:
                items = list(items)
                if len(items) < num:
                    sample_idx = range(len(items))
                else:
                    sample_idx = sample(range(len(items)), num)
                train_dataset.imgs.extend([items[i] for i in sample_idx])
        
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
        
        return train_loader
    
    
    def get_finetune_test_loader(self, target, shot=3, batch_size=30, num_workers=8):
        targetdir = os.path.join(self.root_dir, target)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        

        target_dataset = datasets.ImageFolder(
            targetdir,
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        )
        
        if shot == 0:
            test_dataloader = torch.utils.data.DataLoader(
            target_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
            return None, test_dataloader
        
        finetune_dataset = copy.deepcopy(target_dataset)
        finetune_dataset.imgs.clear()
        finetune_dataset.transform = transforms.Compose([
                transforms.RandomResizedCrop((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        test_dataset = copy.deepcopy(target_dataset)
        test_dataset.imgs.clear()
        
        group = groupby(target_dataset.imgs, key=lambda x: x[1])
        
        for key, items in group:
            items = list(items)
            sample_idx = sample(range(len(items)), shot)
            others = [i for i in range(len(items)) if i not in sample_idx]
            finetune_dataset.imgs.extend([items[i] for i in sample_idx])
            test_dataset.imgs.extend([items[i] for i in others])
            
        finetune_dataloader = torch.utils.data.DataLoader(
            finetune_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
            
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        
        return finetune_dataloader, test_dataloader
            
            
        
        
        