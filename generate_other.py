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

#%%

if dataset == 'cpv2':
    print('Load original data from: original_dataset.pkl')
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    print('Load original data from: v2_original_dataset.pkl')
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
print('Dataset size: ', len(original_dataset))


# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence


# 1. create exist qa triplets dict
print('1. Create exist qa triplets dict')
exist_triplets_dict = defaultdict(dict)
question_ans_dict = defaultdict(dict)
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    ans_texts = entry['answer_text']
    question = handle(entry['question'])
    img_id = entry['img_id']
    for ans in ans_texts:
        exist_triplets_dict[question + ans][img_id] = True
        question_ans_dict[question][ans] = True

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
        'q_id': entry['q_id'],
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

# 6. find other questions
print('6. Find What questions')
questions = list(question_info.keys())
other_questions = []
for question in questions:
    # what question
    qtype = question_info[question]['qtype']
    if qtype != 'what':
        continue
    # only consider one noun
    if len(question_info[question]['nouns']) != 1:
        continue
    other_questions.append(question)

# first step veriﬁcation
valid_other_questions = []
for question in other_questions:
    idxs = question_info[question]['entry_idxs']
    nouns = question_info[question]['nouns']
    if len(nouns) == 0:
        continue

    ans_dict = question_ans_dict[question]
    if len(ans_dict) == 0:
        continue

    valid = False
    for idx in idxs:
        entry = original_dataset[idx]
        obj_dict = entry['obj_dict']
        answer = entry['answer_text']
        if len(answer) == 0:
            continue
        
        # Slightly difference between cpv2 and v2, but there is little difference.
        if dataset == 'cpv2':
            for ans in answer:
                if obj_dict.get(ans, False):
                    valid = True
                    break
        else:
            ans = answer[0]
            if obj_dict.get(ans, False):
                valid = True
                break
    if valid:
        valid_other_questions.append(question)

print('Other Question Count: ', len(valid_other_questions))

#%%

print('6. Pair image-question pairs')
count = 0
for question in tqdm(valid_other_questions, total=len(valid_other_questions), ncols=80):
    info = question_info[question]
    noun = info['nouns'][0]

    returned_imgs = {}
    for img_id in obj2imgIds[noun]:
        returned_imgs[img_id] = True
    info['returned_imgs'] = list(returned_imgs.keys())
    count = count + len(returned_imgs)
print('Other VQ Pairs:', count)

#%%

# Assign Answers
print('7. Assign initial answers and save')
other_aug_dataset = []
count = 0
for question in tqdm(valid_other_questions, ncols=80, total=len(valid_other_questions)):
    info = question_info[question]
    returned_imgs = info['returned_imgs']
    ans_dict = question_ans_dict[question]
    nouns = info['nouns']

    for ans in list(ans_dict.keys()):
        for img_id in returned_imgs:
            count += 1
            img_info = image_info[img_id]
            # judge if exist
            if exist_triplets_dict[question + ans].get(img_id, False):
                continue

            # judge if has this annotation
            obj_dict = img_info['obj_dict']
            if not obj_dict.get(ans, False):
                continue

            # new Entry
            newEntry = {
                'q_id': 'other_aug_' + str(len(other_aug_dataset)),     # assign new question id
                'img_id': img_id,
                'question': question,
                'answer_text': [ans],
                'scores': [1.0],
                'objects': img_info['objects'],
                'attributes': img_info['attributes'],
                'nouns': nouns,
                'qtype': info['qtype'], 
            }

            # second step veriﬁcation
            valid = True
            for idx in img_info['entry_idxs']:
                entry_j = original_dataset[idx]
                qid_j = entry_j['q_id']
                if qid2qtype[qid_j] == info['qtype']:
                    nouns_j = entry_j['nouns']
                    if len(nouns_j) != 1:
                        continue
                    if nouns_j[0] != nouns[0]:
                        continue
                    ans = newEntry['answer_text'][0]
                    if not entry_j['obj_dict'].get(ans, False):
                        valid = False
                        break
            if valid:
                other_aug_dataset.append(newEntry)
print('other augmented dataset:', len(other_aug_dataset))


if dataset == 'cpv2':
    with open('./aug_data/cpv2_other_aug_dataset.pkl', 'wb') as f:
        pickle.dump(other_aug_dataset, f)
else:
    with open('./aug_data/v2_other_aug_dataset.pkl', 'wb') as f:
        pickle.dump(other_aug_dataset, f)

