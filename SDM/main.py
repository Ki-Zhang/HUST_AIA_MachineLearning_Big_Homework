from dataloader import Office
from model import get_ts, pretrain, dual_finetune, test
from logger import Logger
from config import cfg, device
import numpy as np
import torch
import os

def main(dataset):
    data_path = os.path.join(cfg.root, dataset)
    data = Office(data_path, dataset)
    logger = Logger(dataset, cfg)
    source = data.domain
    print("Sources: ", source)
    if cfg.pretrain.enable:
        for s in source:
            print('Pretraining on', s)
            if s in cfg.pretrain.epochs.ex.keys():
                epochs = cfg.pretrain.epochs.ex[s]
            else:
                epochs = cfg.pretrain.epochs.default
            dataloader = data.get_dual_train_loader(s, batch_size=cfg.pretrain.batch)
            pretrain(dataloader=dataloader, source=s, epochs=epochs, 
                     num_classes=len(data.classes), cfg=cfg)
    
    target = [source.copy() for _ in range(len(source))]
    for i in range(len(source)):
        target[i].remove(source[i])
    print("Target: ", target)
    result = {}
    for i in range(len(source)):
        for t in target[i]:
            print('Testing '+source[i]+'->'+t)
            finetune_loader, test_loader = data.get_finetune_test_loader(t, batch_size=len(data.classes), shot=cfg.finetune.shot)
            model = get_ts(len(data.classes), source[i]+'.pth', cfg=cfg).to(device)
            if source[i] in cfg.finetune.epochs.ex.keys():
                epochs = cfg.finetune.epochs.ex[source[i]]
            else:
                epochs = cfg.finetune.epochs.default
            if cfg.finetune.enable:
                dual_finetune(model, finetune_loader, epochs=epochs, alpha=cfg.finetune.alpha, opt=cfg.finetune.opt)
            result[source[i]+'->'+t] = test(model, test_loader)
            
    result['mean'] = sum(result.values())/len(result)
            
    values = list(result.values())
    values = list(map(lambda x: '%.1f%%'%(x*100), values))
    logger.save_table(result.keys(), [values])
    
def same_seeds(seed):
    torch.manual_seed(seed) # 固定随机种子（CPU）
    if torch.cuda.is_available(): # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed) # 为当前GPU设置
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置
    np.random.seed(seed) # 保证后续使用random函数时，产生固定的随机数
    
if __name__ == '__main__':
    same_seeds(cfg.seed)
    for dataset in cfg.datasets:
        main(dataset)