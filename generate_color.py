#%%

# imports
import os
import json
import pickle
import utils
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse

#%%

def parse_args():
    parser = argparse.ArgumentParser("Generate other augmented answers.")

    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset

#%%
# get question type
if dataset == 'cpv2':
    data_question_annotation = json.load(open('./data/vqacp_v2_train_annotations.json', 'r'))
else:
    data_question_annotation = json.load(open('./data/v2_mscoco_train2014_annotations.json', 'r'))['annotations']

qid2qtype = {}
qid2type = {}
for anno in data_question_annotation:
    qid = anno['question_id']
    qtype = anno['question_type'].lower()
    qid2qtype[qid] = qtype
    qid2type[qid] = anno['answer_type']

if dataset == 'cpv2':
    print('Load original data from: original_dataset.pkl')
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    print('Load original data from: v2_original_dataset.pkl')
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
print('Dataset size: ', len(original_dataset))


#%%

# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence

#%% md

### Speed Up Techs

#%%

# 1. create exist qa triplets dict
print('1. Create exist qa triplets dict')
exist_triplets_dict = defaultdict(dict)
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    exist_triplets_dict[handle(entry['question'])][entry['img_id']] = True

# 2. collect question information
print('2. Collect question information')
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

# 3. create obj_dict for every entry
print('3. Create obj_dict for every entry')
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    obj_dict = {}
    for obj in entry['objects']:
        obj_dict[obj] = True
    entry['obj_dict'] = obj_dict

# 4. collect image info
print('4. Collect image info')
image_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    img_id = entry['img_id']
    if image_info.get(img_id, None) is not None:
        image_info[img_id]['entry_idxs'].append(i)
    else:
        info = {
            'objects': entry['objects'],
            'attributes': entry['attributes'],
            'entry_idxs': [i],
            'obj_dict': entry['obj_dict']
        }
        image_info[img_id] = info


# 5. create object to image_id
print('5. Create object to image image id')
obj2imgIds = defaultdict(list)
for img_id, info in tqdm(image_info.items(), ncols=100, total=len(image_info)):
    obj_dict = info['obj_dict']
    for obj in list(obj_dict.keys()):
        obj2imgIds[obj].append(img_id)

#%% md

#### Simple AUG(Color Question)


#%%
# 6. find color questions
print('6. Find Color questions')
color_qs = {}
for question in list(question_info.keys()):
    qtype = question_info[question]['qtype']
    if 'color' in qtype.split():
        color_qs[qtype] = True
color_qs.keys()

#%%

# 1. find color questions and only 1 noun
questions = list(question_info.keys())
color_questions = []
for question in questions:
    # color question
    qtype = question_info[question]['qtype']
    if 'color' not in qtype.split():
        continue
    # only consider one noun
    if len(question_info[question]['nouns']) != 1:
        continue
    color_questions.append(question)

#%%

# 2. find color attributes
color_attr_count_dict = defaultdict(int)
for entry in original_dataset:
    question = handle(entry['question'])
    qtype = question_info[question]['qtype']
    if 'color' not in qtype.split():
        continue
    for ans in entry['answer_text']:
        color_attr_count_dict[ans] += 1
color_attr_dict = {}
for key, value in color_attr_count_dict.items():
    if value < 20:
        continue
    color_attr_dict[key] = True
color_attr_dict.keys()

valid_color_questions = []
for question in color_questions:
    idxs = question_info[question]['entry_idxs']
    nouns = question_info[question]['nouns']
    if len(nouns) == 0:
        continue
    noun = nouns[0]

    valid = True
    for idx in idxs:
        entry = original_dataset[idx]
        objs = entry['objects']
        attrs = entry['attributes']
        answer = entry['answer_text']

        colors = []
        for i in range(len(objs)):
            if objs[i] != noun:
                continue
            if attrs[i] == '':
                continue
            colors.append(attrs[i])
        flag = False
        for color in colors:
            if color in entry['answer_text']:
                flag = True
                break
        if not flag:
            valid = False
            break
    if valid:
        valid_color_questions.append(question)

print('Color Question Count: ', len(valid_color_questions))

#%%

print('6. Pair image-question pairs')
count = 0
for question in tqdm(valid_color_questions, total=len(valid_color_questions), ncols=80):
    info = question_info[question]
    nouns = info['nouns']

    returned_imgs = {}
    for noun in nouns:
        for img_id in obj2imgIds[noun]:
            if exist_triplets_dict[question].get(img_id, False):
                continue
            returned_imgs[img_id] = True
    info['returned_imgs'] = list(returned_imgs.keys())
    count = count + len(returned_imgs)
print('Color VQ Pairs:', count)

#%%

# Assign Answers
print('7. Assign initial answers and save')
color_aug_dataset = []
for question in tqdm(valid_color_questions, total=len(valid_color_questions), ncols=80):
    info = question_info[question]
    nouns = info['nouns']
    returned_imgs = info['returned_imgs']
    ori_noun = info['ori_nouns'][0]

    for img_id in returned_imgs:
        img_info = image_info[img_id]
        attrs = img_info['attributes']
        objs = img_info['objects']

        # assign answer
        noun = nouns[0]
        ans = ''
        for i in range(len(objs)):
            if objs[i] != noun:
                continue
            if attrs[i] != '' and color_attr_dict.get(attrs[i], False):
                ans = attrs[i]
                break
        if ans == '':
            continue

        newEntry = {
            'q_id': 'color_aug_' + str(len(valid_color_questions)),     # assign new question id
            'img_id': img_id,
            'question': question,
            'answer_text': [ans],
            'scores': [1.0],
            'objects': img_info['objects'],
            'nouns': nouns,
            'qtype': info['qtype'],
            'ori_nouns': info['ori_nouns']
        }

        # second step veriﬁcation
        valid = True
        for idx in img_info['entry_idxs']:
            entry_j = original_dataset[idx]
            qid_j = entry_j['q_id']
            if 'color' in qid2qtype[qid_j].split():
                nouns_j = entry_j['nouns']
                if not (len(nouns_j) == 1 and nouns_j[0] == noun):
                    continue
                ans = newEntry['answer_text'][0]
                if ans not in entry_j['answer_text']:
                    valid = False
                    break
        if not valid:
            continue
        color_aug_dataset.append(newEntry)

        # create more IQA triplets about colors
        for i in range(len(objs)):
            if objs[i] == noun or objs[i] == ori_noun:
                continue
            if attrs[i] == '' or not color_attr_dict.get(attrs[i], False):
                continue
            newEntry = {
                'q_id': 'color_aug_' + str(len(color_aug_dataset)),     # assign new question id
                'img_id': img_id,
                'question': question.replace(' ' + ori_noun, ' ' + objs[i]),
                'answer_text': [attrs[i]],
                'scores': [1.0],
                'objects': img_info['objects'],
                'nouns': nouns,
                'qtype': info['qtype'],
                'ori_nouns': [objs[i]]
            }
            color_aug_dataset.append(newEntry)
print('color augmented dataset:', len(color_aug_dataset))


#%%
if dataset == 'cpv2':
    with open('./aug_data/cpv2_color_aug_dataset.pkl', 'wb') as f:
        pickle.dump(color_aug_dataset, f)
else:
    with open('./aug_data/v2_color_aug_dataset.pkl', 'wb') as f:
        pickle.dump(color_aug_dataset, f)
