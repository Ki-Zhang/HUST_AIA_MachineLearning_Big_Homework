import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from FSEmbNet import FSCNN, train_FSCNN
from FSPredictor import FSPredictor, fine_tune_FSPredictor, test_model
import my_dataset as ds


def main_pretrain(source_path):
    source_folder = os.path.join(curr_path, source_path)
    emb_epochs = 6
    emb_batch_size = 128
    emb_learning_rate = 5e-4
    source_set = ds.source_domain_dataset(source_folder)
    source_set_loader = DataLoader(source_set,
                                   batch_size=emb_batch_size,
                                   num_workers=4,
                                   shuffle=True)
    train_FSCNN(epochs=emb_epochs,
                learning_rate=emb_learning_rate,
                train_loader=source_set_loader)


def main_finetune(target_path):
    target_folder = os.path.join(curr_path, target_path)

    support_dict, query_dict = ds.split_support_query(target_folder)
    support_set = ds.target_support_dataset(support_dict, trans=Resize((224, 224)))
    test_set = ds.target_query_dataset(query_dict, trans=Resize((224, 224)))
    fine_tune_FSPredictor(epochs=80,
                          learning_rate=5e-4,
                          support_set=support_set,
                          use_entropy_regularization=True)
    score = test_model(support_set, test_set)
    return score


def main_pf(tbl, source_path, target_path):
    s, t = os.path.split(source_path)[0], os.path.split(target_path)[0]
    print(f"----------------------source: {s}, target: {t}, begin!----------------------")
    main_pretrain(source_path)
    score = main_finetune(target_path)
    tbl.append(f'{s}, {t}:{score * 100.0:>0.1f}')
    print(f"----------------------source: {s}, target: {t}, end!----------------------")


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_path = os.getcwd()
    tbl = []
    amazon_path = "office\\amazon\\images"
    dslr_path = "office\\dslr\\images"
    webcam_path = "office\\webcam\\images"

    main_pf(tbl, amazon_path, dslr_path)
    main_pf(tbl, amazon_path, webcam_path)
    main_pf(tbl, dslr_path, amazon_path)
    main_pf(tbl, dslr_path, webcam_path)
    main_pf(tbl, webcam_path, amazon_path)
    main_pf(tbl, webcam_path, dslr_path)

    print(tbl)
