import  os
import torch
from torch.utils.data import Dataset
import  torchvision.transforms as transforms
import  numpy as np
from torchvision.datasets import ImageFolder
from    PIL import Image
import random

class Office:
    def __init__(self, root, name, batchsz, n_way, k_shot, k_query, resize, normalize, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        #print('shuffle DB: %s,')

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean = normalize[0], std = normalize[1])
                                                 ])
        self.path = os.path.join(root, name, 'images')
        self.data = []
        self.img2label = {}     # 用于存储
        label_list = os.listdir(self.path)
        for k, i in enumerate(label_list):
            temp_data = []
            for j in os.listdir(os.path.join(self.path, i)):
                temp_data.append(str.join('/', [i, j]))
            self.data.append(temp_data)           # 以amazon为例 [['back_pack/frame_0001.jpg' 'back_pack/frame_0002.jpg' ...],['bike/frame_001.jpg' ...] ...]
            self.img2label[i] = k + self.startidx   # 以amazon为例{'back_pack': 0, 'bike': 1, ...}
        
        self.label_num = len(self.data)             # 看分成几类
        self.create_batch(self.batchsz)             # 创建batch
    
    def create_batch(self, batchsz):
        """
        """
        self.support_batch = []     # support set batch
        self.query_batch = []       # query set batch
        for _ in range(batchsz):    # for each batch
            # 1.select n_way classes randomly
            selected_label = np.random.choice(self.label_num, self.n_way, False)    # 用于分类的类别
            np.random.shuffle(selected_label)                                       # 打乱顺序
            support_x = []
            qurey_x = []
            for idx in selected_label:
                # 2.select k_shot+k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[idx]), self.k_shot+self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])             # 用于训练的图
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])              # 用于测试的图
                support_x.append(np.array(self.data[idx])[indexDtrain].tolist())    # 得到测试集的文件名
                qurey_x.append(np.array(self.data[idx])[indexDtest].tolist())       # 得到验证集的文件名

            # 对图片做shuffle
            random.shuffle(support_x)
            random.shuffle(qurey_x)

            # 扩充测试集或者验证集的batch
            self.support_batch.append(support_x)
            self.query_batch.append(qurey_x)
    
    def __getitem__(self, index):
        """
        """
        # 测试集的dataset   [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # 验证集的dataset   [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int32)
        # 测试集的dataset   [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # 测试集的dataset   [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int32)

        # 支撑集
        flattern_support_x = [os.path.join(self.path, item)
                              for sublist in self.support_batch[index] for item in sublist]     # 遍历得到图片的路径地址
        support_y = np.array([self.img2label[item[:-15]]
                              for sublist in self.support_batch[index] for item in sublist]).astype(np.int32)
        # 验证集
        flattern_query_x = [os.path.join(self.path, item)
                            for sublist in self.query_batch[index] for item in sublist]   # 遍历得到图片的路径地址
        query_y = np.array([self.img2label[item[:-15]]
                            for sublist in self.query_batch[index] for item in sublist]).astype(np.int32)
        
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # 将标志归一化到[0,n-way]
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx
        
        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flattern_support_x):
            support_x[i] = self.transform(path)
        for i, path in enumerate(flattern_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # 当我们建立数据集时可以先让batch取一个较小值
        return self.batchsz



def getStat(path, name):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    img_trans = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor()])
    label_list = os.listdir(path)
    number = 0
    # for i in label_list:
    train_data = ImageFolder(root=os.path.join(path, name, 'images'), transform = img_trans)
    print('Compute mean and variance for training&valid data.')
    print(len(train_data))
    number = number + len(train_data)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:

        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(number)
    std.div_(number)
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    nomalize = [[],[]]
    path = '../data_set/office'
    nomalize[0], nomalize[1] = getStat(path, 'webcam')
    print(nomalize)

    # from torchvision.utils import make_grid
    # from matplotlib import pyplot as plt
    # import time

    # nomalize = [[0.54296356, 0.5387488, 0.53735834], [0.17181164, 0.17510022, 0.17613916]]
    # dataset = Office('/home/kdzhang/PC&ML/my_maml/office', 'amazon', 1, 2, 3, 4, 28, nomalize)
    # for i, set in enumerate(dataset):
    #     # support_x: [k_shot*n_way, 3, 84, 84]
    #     support_x, support_y, query_x, query_y = set

    #     support_x = make_grid(support_x, nrow=2)
    #     query_x = make_grid(query_x, nrow=2)

    #     time.sleep(5)