import torch
from torch import nn
from torchvision import models
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from config import device

from torchvision import utils as vutils

class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder.fc = nn.Sequential()

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Vgg16(nn.Module):

    def __init__(self):
        super(Vgg16, self).__init__()
        self.encoder = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        n_features = self.encoder.classifier[6].in_features 
        self.encoder.classifier[6] = nn.Linear(n_features, 1024)

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class AlexNet(nn.Module):
    
    def __init__(self):
        super(AlexNet, self).__init__()
        self.encoder = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.encoder.classifier[6] = nn.Sequential()
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class TS_ResNet50(nn.Module):

    def __init__(self,num_classes=31):
        super(TS_ResNet50, self).__init__()
        self.backbone = ResNet50()
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        inner = self.head(x)
        y = self.classifier(inner)
        return inner, y
    
class TS_Vgg16(nn.Module):

    def __init__(self,num_classes=31):
        super(TS_Vgg16, self).__init__()
        self.backbone = Vgg16()
        self.head = nn.Sequential(
            nn.Linear(1024, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        inner = self.backbone(x)
        y = self.head(inner)
        y = self.classifier(y)
        return inner, y
    
class TS_AlexNet(nn.Module):
    
    def __init__(self,num_classes=31):
        super(TS_AlexNet, self).__init__()
        self.backbone = AlexNet()
        self.head = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        inner = self.head(x)
        y = self.classifier(inner)
        return inner, y

def get_ts(num_classes=31, pretrain_path=None, cfg=None):
    if cfg.backbone == 'resnet50':
        ts = TS_ResNet50(num_classes=num_classes)   
    elif cfg.backbone == 'vgg16':
        ts = TS_Vgg16(num_classes=num_classes)
    elif cfg.backbone == 'alexnet':
        ts = TS_AlexNet(num_classes=num_classes)
    else:
        raise NotImplementedError
    
    if pretrain_path is not None:
        ts.load_state_dict(torch.load(pretrain_path))
    return ts

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                self.model.state_dict()[name].copy_(new_average)
    
def pretrain(dataloader, num_classes=31, pretrain_path=None, source='amazon', epochs=60, cfg=None):
    opt = cfg.pretrain.opt
    switch_epoch = cfg.pretrain.switch_epoch
    teacher = get_ts(num_classes,pretrain_path=pretrain_path, cfg=cfg).to(device)
    teacher = EMA(teacher, opt.ema)

    student = get_ts(num_classes,pretrain_path=pretrain_path, cfg=cfg).to(device)

    optimizer = torch.optim.SGD(student.parameters(), 
                                lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=False)
    criterion = nn.CrossEntropyLoss()
    consistency = nn.CosineEmbeddingLoss()
    
    student.train()
    teacher.model.eval()

    for epoch in range(epochs):

        with tqdm(total=len(dataloader), ascii=' =') as t:
        
            loss_mean = 0.
            correct = 0.
            total = 0.

            for batch_idx, (images, labels) in enumerate(dataloader):
                weak = images[0].to(device)
                strong = images[1].to(device)
                labels = labels.to(device)

                if epoch > switch_epoch:
                    inner_student,outputs_student = student(strong)
                    inner_teacher,outputs_teacher = teacher.model(weak)
                else:
                    _,outputs_student = student(weak)
                
                optimizer.zero_grad()
                loss_label = criterion(outputs_student, labels)
                
                if epoch > switch_epoch:
                    loss_c = consistency(inner_student, inner_teacher, torch.ones(inner_student.shape[0]).to(device))
                    loss = loss_label + opt.alpha*(epoch/epochs)*loss_c
                else:
                    loss = loss_label
                    
                loss.backward()
                optimizer.step()

                teacher.update(student)

                loss_mean += loss.item()

                _, predicted = torch.max(outputs_student.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).squeeze().cpu().sum().numpy()

                t.set_description(desc="Epoch %i"%epoch)
                t.set_postfix(loss=loss.data.item(),mean=loss_mean/total,acc=correct/total)
                t.update(1)

    torch.save(teacher.model.state_dict(), source+'.pth')
    

def dual_finetune(model, dataloader, epochs=60, alpha=0.5, opt=None):
    dataset = dataloader.dataset
    ran_dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers, pin_memory=True)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=False)
    criterion = nn.CrossEntropyLoss()
    
    model.train()

    for epoch in range(epochs):
        
        dual_dataloader = zip(dataloader, ran_dataloader)
        with tqdm(total=len(dataloader), ascii=' =') as t:
        
            loss_mean = 0.
            total = 0.

            for batch_idx, ((A_images,A_labels),(B_images,B_labels)) in enumerate(dual_dataloader):
                
                lam = np.random.beta(alpha, alpha)
                
                if np.random.rand() > 0.7:
                    images = lam*A_images + (1-lam)*B_images
                    mix = True
                else:
                    images = A_images
                    mix = False
                    
                images, A_labels, B_labels = images.to(device), A_labels.to(device), B_labels.to(device)
                
                _,outputs = model(images)
                
                optimizer.zero_grad()
                
                if mix:
                    loss = lam * criterion(outputs, A_labels) + (1 - lam) * criterion(outputs, B_labels)
                else:
                    loss = criterion(outputs, A_labels)
                    
                loss.backward()
                optimizer.step()

                loss_mean += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += A_labels.size(0)

                t.set_description(desc="Epoch %i"%epoch)
                t.set_postfix(loss=loss.data.item(),mean=loss_mean/total)
                t.update(1)


def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), ascii=' =') as t:
            for idx, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                _, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).squeeze().cpu().sum().numpy()

                t.set_postfix(acc=correct / total)
                t.update(1)
    print('Accuracy of the network on the test images: %.1f %%' % (100 * correct / total))
    return correct / total
