#%%

# Imports

import argparse
import json

import torch
from torch.utils.data import DataLoader

# from new_dataset import Dictionary, VQAFeatureDataset
from dataset import Dictionary, VQAFeatureDataset
from aug_dataset import VQAAugFeatureDataset
import base_modelKD
from train import train
import utils

from vqa_debias_loss_functions import *
from tqdm import tqdm
from torch.autograd import Variable
import pickle
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import random


def parse_args():
    parser = argparse.ArgumentParser("Merge augmented dataset.")

    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset


if dataset == 'v2':
    with open('./aug_data/v2_color_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        color_high = pickle.load(f)
    with open('./aug_data/v2_number_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        number_high = pickle.load(f)
    with open('./aug_data/v2_other_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        other_high = pickle.load(f)
    with open('./aug_data/v2_paraphrasing_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        para_high = pickle.load(f)
else:
    with open('./aug_data/cpv2_color_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        color_high = pickle.load(f)
    with open('./aug_data/cpv2_number_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        number_high = pickle.load(f)
    with open('./aug_data/cpv2_other_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        other_high = pickle.load(f)
    with open('./aug_data/cpv2_paraphrasing_aug_dataset_high_quality_clean.pkl', 'rb') as f:
        para_high = pickle.load(f)

if dataset == 'v2':
    with open('./aug_data/v2_color_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        color_clean = pickle.load(f)
    with open('./aug_data/v2_number_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        number_clean = pickle.load(f)
    with open('./aug_data/v2_other_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        other_clean = pickle.load(f)
    # cpv2_paraphrasing_aug_dataset_low_quality_clean.pkl
    with open('./aug_data/v2_paraphrasing_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        para_clean = pickle.load(f)
else:
    with open('./aug_data/cpv2_color_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        color_clean = pickle.load(f)
    with open('./aug_data/cpv2_number_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        number_clean = pickle.load(f)
    with open('./aug_data/cpv2_other_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        other_clean = pickle.load(f)
    # cpv2_paraphrasing_aug_dataset_low_quality_clean.pkl
    with open('./aug_data/cpv2_paraphrasing_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        para_clean = pickle.load(f)

if dataset == 'cpv2':
    with open('./aug_data/cpv2_yesno_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        yesno_clean = pickle.load(f)
else:
    with open('./aug_data/v2_yesno_aug_dataset_low_quality_clean.pkl', 'rb') as f:
        yesno_clean = pickle.load(f)


aug_total = color_high + other_high + number_high + para_high
aug_total += color_clean + other_clean + number_clean + para_clean
aug_total = aug_total + yesno_clean


valid_keys = ['q_id', 'img_id', 'question', 'answer_text', 'scores', 'qtype', 'ori_answer_text', 'logits', 'nouns']
for entry in aug_total:
    delete_keys = []
    for key in entry.keys():
        if key in valid_keys:
            continue
        delete_keys.append(key)
    for key in delete_keys:
        entry.pop(key)

print("ALL DATASET: ", len(aug_total))
if dataset == 'cpv2':
    with open('./aug_data/cpv2_all_aug_dataset.pkl', 'wb') as f:
        pickle.dump(aug_total, f)
else:
    with open('./aug_data/v2_all_aug_dataset.pkl', 'wb') as f:
        pickle.dump(aug_total, f)