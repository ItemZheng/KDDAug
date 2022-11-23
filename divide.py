#%%

# Imports
import argparse
import json

import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
from aug_dataset import VQAAugFeatureDataset
import utils

from vqa_debias_loss_functions import *
from tqdm import tqdm
from torch.autograd import Variable
import pickle
import random

import torch
import clip


def parse_args():
    parser = argparse.ArgumentParser("Divide Augmentated Samples to high quality or low quality")

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



#%%

args = parse_args()

# get param
dataset = args.dataset

# load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Load augmented dataset
if dataset == 'cpv2':
    # load data and generate statements
    with open('./aug_data/cpv2_other_aug_dataset.pkl', 'rb') as f:
        data_other = pickle.load(f)
    with open('./aug_data/cpv2_color_aug_dataset.pkl', 'rb') as f:
        data_color = pickle.load(f)
    with open('./aug_data/cpv2_number_aug_dataset.pkl', 'rb') as f:
        data_number = pickle.load(f)
    with open('./aug_data/cpv2_paraphrasing_aug_dataset.pkl', 'rb') as f:
        data_para = pickle.load(f)
else:
    # load data and generate statements
    with open('./aug_data/v2_other_aug_dataset.pkl', 'rb') as f:
        data_other = pickle.load(f)
    with open('./aug_data/v2_color_aug_dataset.pkl', 'rb') as f:
        data_color = pickle.load(f)
    with open('./aug_data/v2_number_aug_dataset.pkl', 'rb') as f:
        data_number = pickle.load(f)
    with open('./aug_data/v2_paraphrasing_aug_dataset.pkl', 'rb') as f:
        data_para = pickle.load(f)

if dataset == 'cpv2':
    cpv2_question_annotation = json.load(open('./data/vqacp_v2_train_annotations.json', 'r'))
    qid2qtype = {}
    qid2type = {}
    for anno in cpv2_question_annotation:
        qid = anno['question_id']
        qtype = anno['question_type'].lower()
        qid2qtype[qid] = qtype
        qid2type[qid] = anno['answer_type']
else:
    # get question type
    v2_question_annotation = json.load(open('./data/v2_mscoco_train2014_annotations.json', 'r'))['annotations']
    # %%
    qid2qtype = {}
    qid2type = {}
    for anno in v2_question_annotation:
        qid = anno['question_id']
        qtype = anno['question_type'].lower()
        qid2qtype[qid] = qtype
        qid2type[qid] = anno['answer_type']


# load original dataset
if dataset == 'cpv2':
    # Can Execute From Here
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    # Can Execute From Here
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)

# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence

#%%

# 2. collect question information
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
        'qtype': qid2qtype[entry['q_id']],
        'type': qid2type[entry['q_id']],
        'entry_idxs': [i],
        'returned_imgs': [],
    }
    question_info[question] = info


print("Generate statement for each vq pairs")
for entry in data_number:
    question = entry['question']
    qtype = question_info[question]['qtype']
    ori_noun = question_info[question]['ori_nouns'][0]
    ans = entry['answer_text'][0]
    statement = question.replace(qtype, '').strip().replace(ori_noun, ans + ' ' + ori_noun)
    entry['statement'] = statement


for entry in data_color:
    question = entry['question']
    qtype = entry['qtype']
    ori_noun = entry['ori_nouns'][0]
    ans = entry['answer_text'][0]
    statement = question.replace(qtype, '').strip().replace(ori_noun, ans + ' ' + ori_noun)
    entry['statement'] = statement

for entry in data_other:
    question = entry['question']
    qtype = question_info[question]['qtype']
    ori_noun = question_info[question]['ori_nouns'][0]
    ans = entry['answer_text'][0]
    statement = question.replace(qtype, '').strip().replace(ori_noun, ans)
    entry['statement'] = statement

for entry in data_para:
    qtype = entry['qtype']
    ans = ''
    if len(entry['answer_text']) != 0:
        ans = entry['answer_text'][0]
    question = entry['question']
    statement = question.replace(qtype, ans)
    entry['statement'] = statement


if dataset == 'cpv2':
    with open('./aug_data/cpv2_other_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_other, f)
    with open('./aug_data/cpv2_color_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_color, f)
    with open('./aug_data/cpv2_number_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_number, f)
    with open('./aug_data/cpv2_paraphrasing_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_para, f)
else:
    with open('./aug_data/v2_other_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_other, f)
    with open('./aug_data/v2_color_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_color, f)
    with open('./aug_data/v2_number_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_number, f)
    with open('./aug_data/v2_paraphrasing_aug_dataset.pkl', 'wb') as f:
        pickle.dump(data_para, f)

if dataset == 'v2':
    with open('./aug_data/v2_imgId_to_clip_feature_dict.pkl', 'rb') as f:
        imgId_to_clip_feature_dict = pickle.load(f)
else:
    with open('./aug_data/imgId_to_clip_feature_dict.pkl', 'rb') as f:
        imgId_to_clip_feature_dict = pickle.load(f)

names = ['other', 'color', 'number', 'paraphrasing']
for name in names:
    path = './aug_data/v2_color_aug_dataset.pkl'
    if name == 'color':
        path = './aug_data/v2_color_aug_dataset.pkl'
    elif name == 'other':
        path = './aug_data/v2_other_aug_dataset.pkl'
    elif name == 'number':
        path = './aug_data/v2_number_aug_dataset.pkl'
    elif name == 'paraphrasing':
        path = './aug_data/v2_paraphrasing_aug_dataset.pkl'

    path = path.replace('v2', dataset)
    high_quality_path = path.replace('dataset', 'dataset_high_quality')
    low_quality_path = path.replace('dataset', 'dataset_low_quality')


    #%%
    print('\n\nLoad augmented data from: ', path)
    print('Save results to: ', high_quality_path, 'and', low_quality_path)

    #%%

    with open(path, 'rb') as f:
        aug_data = pickle.load(f)

    # generate statements
    unique_statements = {}
    for entry in tqdm(aug_data, total=len(aug_data), ncols=80):
        statement = entry['statement']
        unique_statements[statement] = True

    print("Extra statements' CLIP feature.")
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

    print('Calculate similarity score')
    for entry in tqdm(aug_data, total=len(aug_data), ncols=80):
        # get image feature
        img_id = entry['img_id']
        img_feature = imgId_to_clip_feature_dict[img_id]

        # get text feature
        statement = entry['statement']
        text_feature = statement_feature_dict[statement]

        # sim
        sim = torch.nn.functional.cosine_similarity(text_feature.float(), img_feature.float(), dim=0)
        entry['sim'] = sim

    #%%
    print('Divide and save')
    high_quality_entry = []
    low_quality_entry = []

    # high quality ratio
    ratio = args.ratio
    if ratio < 0:
        if dataset == 'cpv2':
            if name == 'color':
                # threshold = 0.2065  # for cpv2 color
                threshold = 0.2115  # for cpv2 color no missing answer
            elif name == 'number':
                threshold = 0.2222 # for cpv2 number
            elif name == 'other':
                # threshold = 0.237 # for cpv2 other
                threshold = 0.2100   # for cpv2 other get rid of coco annotation
            elif name == 'paraphrasing':
                threshold = 0.2315
        else:
            if name == 'color':
                threshold = 0.2077  # for v2 color
            elif name == 'number':
                threshold = 0.2218 # for v2 number
            elif name == 'other':
                # threshold = 0.2275   # for v2 other
                threshold = 0.2120  # for v2 other get rid of coco annotation and no missing answer
            elif name == 'paraphrasing':
                threshold = 0.2295
    else:
        sims = []
        for entry in aug_data:
            sims.append(entry['sim'])
        sorted_sims = sorted(sims)
        threshold_idx = int((1 - ratio) * len(sims))
        if threshold_idx >= len(sims) - 1:
            threshold_idx = len(sims) - 2
        threshold = sorted_sims[threshold_idx]
    

    for entry in tqdm(aug_data, total=len(aug_data), ncols=80):
        if entry['sim'] > threshold:
            high_quality_entry.append(entry)
        else:
            low_quality_entry.append(entry)

    with open(high_quality_path, 'wb') as f:
        pickle.dump(high_quality_entry, f)
    with open(low_quality_path, 'wb') as f:
        pickle.dump(low_quality_entry, f)
    