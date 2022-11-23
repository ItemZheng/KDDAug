#%%

# Imports

import argparse
import json

import torch
from torch.utils.data import DataLoader
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
    parser = argparse.ArgumentParser("Assign Label For Low Quality/High Quality DATA")

    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2"],
        help="datset name"
    )
    parser.add_argument(
        '--name', default='other',
        choices=['number', 'color', 'other', 'paraphrasing', 'yesno', 'paired'],
        help="augmented dataset name"
    )
    parser.add_argument(
        '--split', default='low',
        choices=['low', 'high'],
        help="low quality or high quality dataset"
    )
    parser.add_argument(
        '--teacher_path', default='./logs/lmh_css/model.pth',
        type=str,
        help="Path of teacher model"
    )
    args = parser.parse_args()
    return args


args = parse_args()
dataset = args.dataset
name = args.name
split = args.split
if dataset == 'v2':
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
print('DATASET LEN', len(original_dataset))

# load label
if dataset == 'v2':
    cache_file = os.path.join('data', 'cache', 'trainval_label2ans.pkl')
    label2ans = pickle.load(open(cache_file, 'rb'))
else:
    cache_file = os.path.join('data', 'cp-cache', 'trainval_label2ans.pkl')
    label2ans = pickle.load(open(cache_file, 'rb'))

#load qid2type
qid2qtype = {}
qid2type = {}
if dataset == 'v2':
    # get question type
    v2_question_annotation = json.load(open('./data/v2_mscoco_train2014_annotations.json', 'r'))['annotations']
    for anno in v2_question_annotation:
        qid = anno['question_id']
        qtype = anno['question_type'].lower()
        qid2qtype[qid] = qtype
        qid2type[qid] = anno['answer_type']
else:
    # dataset = 'cpv2'
    cpv2_question_annotation = json.load(open('./data/vqacp_v2_train_annotations.json', 'r'))
    for anno in cpv2_question_annotation:
        qid = anno['question_id']
        qtype = anno['question_type'].lower()
        qid2qtype[qid] = qtype
        qid2type[qid] = anno['answer_type']

#%%
# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence

#%%

from tqdm import tqdm
# collect all image information
image_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    img_id = entry['img_id']
    if image_info.get(img_id, None) is None:
        info = {
            'objects': entry['objects'],
            'attributes': entry['attributes'],
        }
        image_info[img_id] = info

# Collect question information
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


# Get language bias, which is an input of CSS teacher model.
print('Get language bias, which is an input of CSS teacher model.')
dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary, dataset=dataset, cache_image_features=False)
# get bias
answer_voc_size = train_dset.num_ans_candidates

# question_type -> answer -> total score
question_type_to_probs = defaultdict(Counter)

# question_type -> num_occurances
question_type_to_count = Counter()
for ex in train_dset.entries:
    ans = ex["answer"]
    q_type = ans["question_type"]
    question_type_to_count[q_type] += 1
    if ans["labels"] is not None:
        for label, score in zip(ans["labels"], ans["scores"]):
            question_type_to_probs[q_type][label] += score
question_type_to_prob_array = {}

for q_type, count in question_type_to_count.items():
    prob_array = np.zeros(answer_voc_size, np.float32)
    for label, total_score in question_type_to_probs[q_type].items():
        prob_array[label] += total_score
    prob_array /= count
    question_type_to_prob_array[q_type] = prob_array


print('Load model from:', args.teacher_path)
constructor = 'build_baseline0_newatt'
ood_model = getattr(base_modelKD, constructor)(train_dset, 1024).cuda()
ood_model.debias_loss_fn = LearnedMixinKD()
model_state = torch.load(args.teacher_path)
ood_model.load_state_dict(model_state)
ood_model = ood_model.cuda().eval()

if split == 'high':
    load_name = 'high_' + name
    clean_name = 'high_clean_' + name
else:
    load_name = 'low_' + name
    clean_name = 'clean_' + name
low_assign_flag = (load_name.find('low') != -1)

print("LOAD PATH: ", load_name)
print("SAVE PATH: ", clean_name)

# dataset
aug_dset = VQAAugFeatureDataset(load_name, dictionary, cache_image_features=True, dataset=dataset)
eval_loader = DataLoader(aug_dset, 512, shuffle=False, num_workers=0)

#%%
for ex in aug_dset.entries:
    if name == 'color':
        q_type = ex["qtype"]
    else:
        q_type = question_info[ex['question']]['qtype']
    ex["bias"] = question_type_to_prob_array[q_type]

#%%

# laod chosen yesno aug dataset
with open('./aug_data/' + VQAAugFeatureDataset.path[dataset][load_name], 'rb') as f:
    low_quality_dataset = pickle.load(f)

#%%
for entry in low_quality_dataset:
    entry['ori_answer_text'] = entry['answer_text']
#%%

# begin assign label
print('Predict to get ood prediction and id prediction.')
begin_idx = 0
logsigmoid = torch.nn.LogSigmoid()
with torch.no_grad():
    for v, q, a, b in tqdm(eval_loader, ncols=100, total=len(eval_loader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        b = Variable(b, requires_grad=False).cuda()
        a = Variable(a, requires_grad=False).cuda()
        # id_pred, _, _ = id_model(v, q, None, None, None)
        ood_pred, id_pred, _, _ = ood_model(v, q, a, b, None)

        # calculate weight
        bb = a  # a/torch.clamp(a.sum(1, keepdim=True), min=1e-24)
        if low_assign_flag:
            bb = b
        s_id = 1 / (-bb * logsigmoid(id_pred) - (1 - bb) * logsigmoid(-id_pred)).sum(dim=1)
        s_ood = 1 / (-bb * logsigmoid(ood_pred) - (1 - bb) * logsigmoid(-ood_pred)).sum(dim=1)
        w_id = s_ood / (s_id + s_ood)
        w_ood = s_id / (s_id + s_ood)

        _, id_ans = torch.max(id_pred, dim=1)
        _, ood_ans = torch.max(ood_pred, dim=1)
        id_ans = id_ans.cpu().numpy()
        ood_ans = ood_ans.cpu().numpy()
        w_id = w_id.cpu().numpy()
        w_ood = w_ood.cpu().numpy()
        if not low_assign_flag:
            id_pred = a.cpu().numpy()
        else:
            id_pred = id_pred.sigmoid().cpu().numpy()

        ood_pred = ood_pred.sigmoid().cpu().numpy()
        for i in range(len(v)):
            idx = begin_idx + i
            low_quality_dataset[idx]['id_ans'] = label2ans[id_ans[i]]
            low_quality_dataset[idx]['ood_ans'] = label2ans[ood_ans[i]]
            low_quality_dataset[idx]['id_w'] = w_id[i]
            low_quality_dataset[idx]['ood_w'] = w_ood[i]
            low_quality_dataset[idx]['id_preds'] = id_pred[i]
            low_quality_dataset[idx]['ood_preds'] = ood_pred[i]
        begin_idx = begin_idx + len(v)

#%%
print('Assign answer')
for entry in low_quality_dataset:
    id_ans = entry['id_ans']
    ood_ans = entry['ood_ans']

    # by updn teacher
    # entry['answer_text'] = [ood_ans]
    # entry['scores'] = [1.0]

    # # by two teacher
    # if id_ans == ood_ans:
    #     entry['answer_text'] = [ood_ans]
    #     entry['scores'] = [1.0]
    # else:
    #     entry['answer_text'] = [ood_ans, id_ans]
    #     entry['scores'] = [entry['ood_w'], entry['id_w']]

    # soft version by two teacher
    entry['logits'] =  entry['id_w'] * entry['id_preds'] + entry['ood_w'] * entry['ood_preds']
    # entry['logits'] = 0.5 * entry['id_preds'] + 0.5 * entry['ood_preds']    # simple average
    # entry['logits'] = entry['id_preds']                                       # id weight 1
    # entry['logits'] = entry['ood_preds']  # ood weight 1

print('Save')
valid_keys = ['q_id', 'img_id', 'question', 'answer_text', 'scores', 'qtype', 'ori_answer_text', 'logits', 'nouns']
for entry in low_quality_dataset:
    delete_keys = []
    for key in entry.keys():
        if key in valid_keys:
            continue
        delete_keys.append(key)
    for key in delete_keys:
        entry.pop(key)

with open('./aug_data/' + VQAAugFeatureDataset.path[dataset][clean_name], 'wb') as f:
    pickle.dump(low_quality_dataset, f)