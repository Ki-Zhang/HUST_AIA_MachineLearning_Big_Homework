from box import Box
from backbone import TS_ResNet50
from copy import deepcopy
from proto import ProtoNet, meta_loss, finetune_loss, get_acc
from config import get_cfg
from sampler import EpisodeSampler, EpisodeCollateFn, DataManager4Office31
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from train import test

if __name__ == "__main__":
    mgr = DataManager4Office31("/root/autodl-tmp/office_31")
    cfg = get_cfg()
    
    def preprocess(s:str):
        s_list = str.split(s)
        return ".".join(s_list[1:])

    def dummy(path:str, target:str):
        finetune_loader, test_loader = mgr.get_finetune_and_test_loader(target)

        # foo = TS_ResNet50()
        # foo.load_state_dict(torch.load(path, map_location='cuda:0'))
        # foo = deepcopy(foo.backbone)
        
        proto = ProtoNet(TS_ResNet50().backbone)
        st = torch.load(path, map_location='cuda:0')
        new_st = dict(zip(list(proto.state_dict().keys()), list(st.values())))
        proto.load_state_dict(new_st)
        proto = proto.cuda()

        test(proto, test_loader)

    dummy('a_w.pth', 'webcam')
    dummy('a_d.pth', 'dslr')
    dummy('d_a.pth', 'amazon')
    dummy('d_w.pth', 'webcam')
    dummy('w_a.pth', 'amazon')
    dummy('w_d.pth', 'dslr')
