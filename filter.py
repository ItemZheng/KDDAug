# %%

# Imports

import argparse
import json

import torch
from torch.utils.data import DataLoader

# from new_dataset import Dictionary, VQAFeatureDataset
from dataset import Dictionary, VQAFeatureDataset
from aug_dataset import VQAAugFeatureDataset
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
    parser = argparse.ArgumentParser("CLIP-based filtering: filter out less-efficient augmented samples.")

    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )

    parser.add_argument(
        '--ratio', default=0.5, type=float,
        help="High quality data's ratio"
    )
    args = parser.parse_args()
    return args


args = parse_args()
dataset = args.dataset

# %%

import torch
import clip
from PIL import Image
import pickle

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# load dataset

if dataset == 'v2':
    print('1.Load from v2_all_aug_dataset.pkl')
    with open('./aug_data/v2_all_aug_dataset.pkl', 'rb') as f:
        all_dataset = pickle.load(f)
else:
    print('1.Load from cpv2_all_aug_dataset.pkl')
    with open('./aug_data/cpv2_all_aug_dataset.pkl', 'rb') as f:
        all_dataset = pickle.load(f)
print('Augmentated dataset size: ', len(all_dataset))

# load original dataset
if dataset == 'v2':
    print('2.Load from v2_original_dataset.pkl')
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    print('2.Load from original_dataset.pkl')
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
print('Original dataset size: ', len(original_dataset))

# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence

#%%

print('3. Collect Question Information')
from tqdm import tqdm
question_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    question = handle(entry['question'])
    if question_info.get(question, None) is not None:
        question_info[question]['entry_idxs'].append(i)
        continue
    info = {
        'nouns': entry['nouns'],
        'ori_nouns': entry['ori_nouns'],
        'entry_idxs': [i],
        'returned_imgs': [],
    }
    question_info[question] = info


# collect all image information
print('4. Collect Image Information')
image_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    img_id = entry['img_id']
    if image_info.get(img_id, None) is None:
        info = {
            # 'annotations': entry['annotations'],
            'objects': entry['objects'],
            'attributes': entry['attributes'],
            # 'img_path': entry['img_path']
        }
        if dataset == 'v2':
            info['img_path'] = entry['img_path']
        image_info[img_id] = info

# load img id to feature
print('5. Load CLIP image features')
if dataset == 'v2':
    with open('./aug_data/v2_imgId_to_clip_feature_dict.pkl', 'rb') as f:
        imgId_to_clip_feature_dict = pickle.load(f)
else:
    with open('./aug_data/imgId_to_clip_feature_dict.pkl', 'rb') as f:
        imgId_to_clip_feature_dict = pickle.load(f)

# collect all nouns
unique_statements = {}
for entry in tqdm(all_dataset, total=len(all_dataset), ncols=80):
    if 'nouns' not in entry.keys():
        nouns = question_info[entry['question']]['nouns']
    else:
        nouns = entry['nouns']
    for noun in nouns:
        unique_statements['a photo of a ' + noun] = True

#%%

# get text feature
print('6. Extra CLIP text feature for each question')
statement_feature_dict = {}
batch_size = 256
batch_statements = []
for statements in tqdm(list(unique_statements.keys()), total=len(unique_statements), ncols=80):
    # collect batch
    batch_statements.append(statements)
    if len(batch_statements) >= batch_size:
        batch_text_tokens = clip.tokenize(batch_statements).to(device)
        with torch.no_grad():
            text_features = model.encode_text(batch_text_tokens).cpu()
        for i in range(len(batch_statements)):
            key = batch_statements[i]
            feature = text_features[i]
            statement_feature_dict[key] = feature
        batch_statements = []

if len(batch_statements) > 0:
    batch_text_tokens = clip.tokenize(batch_statements).to(device)
    with torch.no_grad():
        text_features = model.encode_text(batch_text_tokens).cpu()
    for i in range(len(batch_statements)):
        key = batch_statements[i]
        feature = text_features[i]
        statement_feature_dict[key] = feature
    batch_statements = []

assert len(statement_feature_dict) == len(unique_statements)

import numpy as np

#%%
print('7. Calculate similarity between image and question')
for entry in tqdm(all_dataset, total=len(all_dataset), ncols=80):
    # get image feature
    img_id = entry['img_id']
    img_feature = imgId_to_clip_feature_dict[img_id]

    # get text feature
    sims = []
    if 'nouns' not in entry.keys():
        nouns = question_info[entry['question']]['nouns']
    else:
        nouns = entry['nouns']
    for noun in nouns:
        statement = 'a photo of a ' + noun
        text_feature = statement_feature_dict[statement]

        # sim
        sim = torch.nn.functional.cosine_similarity(text_feature.float(), img_feature.float(), dim=0).cpu().numpy()
        sims.append(sim)
    if len(nouns) == 0:
        sim_mean = 0
    else:
        sim_mean = np.array(sims).mean()
    entry['sim_mean'] = sim_mean



ratio = args.ratio
if ratio < 0:
    ratio = 0.5

sims = []
for entry in all_dataset:
    sims.append(entry['sim_mean'])
sorted_sims = sorted(sims)
threshold_idx = int((1 - ratio) * len(sims))
if threshold_idx >= len(sims) - 1:
    threshold_idx = len(sims) - 2
thre = sorted_sims[threshold_idx]
print('8. Determine thresh: ', thre)


print('9. Filter and save!')
high_entries = []
for entry in all_dataset:
    if entry['sim_mean'] > thre:
        high_entries.append(entry)
print('Filter ratio:', len(high_entries) / len(all_dataset))
if dataset == 'cpv2':
    with open('./aug_data/cpv2_total_aug_dataset.pkl', 'wb') as f:
        pickle.dump(high_entries, f)
else:
    with open('./aug_data/v2_total_aug_dataset.pkl', 'wb') as f:
        pickle.dump(high_entries, f)