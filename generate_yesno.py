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
    parser = argparse.ArgumentParser("Generate yes/no augmented answers.")

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
# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence



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
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    entry['handle_question'] = handle(entry['question'])

# 1. create exist qa triplets dict
print('1. Create exist qa triplets dict')
exist_triplets_dict = defaultdict(dict)
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    exist_triplets_dict[entry['handle_question']][entry['img_id']] = True

# 2. collect question information
print('2. Collect question information')
question_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    question = entry['handle_question']
    if question_info.get(question, None) is not None:
        question_info[question]['entry_idxs'].append(i)
        continue
    info = {
        'nouns': entry['nouns'],
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


# 6. find yes/no questions
print('6. Find yes/no questions')
questions = list(question_info.keys())
yes_no_questions = []
for question in questions:
    if question_info[question]['type'] != 'yes/no':
        continue
    if len(question_info[question]['nouns']) == 0:
        continue
    yes_no_questions.append(question)

# group yes/no questions by nouns
from collections import defaultdict
unique_question_dict = defaultdict(list)
for question in yes_no_questions:
    nouns = sorted(question_info[question]['nouns'])
    noun_key = ""
    for noun in nouns:
        noun_key = noun_key + "##########" + noun
    unique_question_dict[noun_key].append(question)

# randomly select 3 questions
valid_yes_no_questions = []
for key, value in unique_question_dict.items():
    valid_yes_no_questions.append(value[0])
    if len(value) > 1:
        valid_yes_no_questions.append(value[1])
    if len(value) > 2:
        valid_yes_no_questions.append(value[2])

#%%
print('Yes/No Question Count: ', len(valid_yes_no_questions))

print('6. Pair image-question pairs')
count = 0
for question in tqdm(valid_yes_no_questions, total=len(valid_yes_no_questions), ncols=80):
    info = question_info[question]
    nouns = info['nouns']

    returned_imgs = {}
    for noun in nouns:
        for img_id in obj2imgIds[noun]:
            if exist_triplets_dict[question].get(img_id, False):
                continue
            returned_imgs[img_id] = True

    # all nouns in questions must appears in image
    valid_returned_img = {}
    for img_id in list(returned_imgs.keys()):
        obj_dict = image_info[img_id]['obj_dict']
        miss_count = 0
        for noun in nouns:
            if not obj_dict.get(noun, False):
                miss_count += 1
                break
        if miss_count == 0:
            valid_returned_img[img_id] = True

    info['returned_imgs'] = list(valid_returned_img.keys())
    count = count + len(valid_returned_img)
print('Yes/No VQ Pairs:', count)


# 4. Assign Answers
print('7. Assign initial answers (Can be ignored for yes/no questions) and save')
yesno_aug_dataset = []
for question in tqdm(valid_yes_no_questions, total=len(valid_yes_no_questions), ncols=80):
    info = question_info[question]
    nouns = info['nouns']
    returned_imgs = info['returned_imgs']

    for img_id in returned_imgs:
        img_info = image_info[img_id]
        obj_dict = img_info['obj_dict']

        # assign answer
        YesAns = True
        for noun in nouns:
            if not obj_dict.get(noun, False):
                YesAns = False
                break
        newEntry = {
            'q_id': 'yesno_aug_' + str(len(yesno_aug_dataset)),     # assign new question id
            'img_id': img_id,
            'question': question,
            'answer_text': ['yes'] if YesAns else ['no'],   # fake ans for this
            'scores': [1.0],
            'objects': img_info['objects'],
            'attributes': img_info['attributes'],
            'nouns': nouns,
            'qtype': info['qtype'],
        }
        yesno_aug_dataset.append(newEntry)

print('yes/no augmented dataset:', len(yesno_aug_dataset))

if dataset == 'cpv2':
    with open('./aug_data/cpv2_yesno_aug_dataset_low_quality.pkl', 'wb') as f:
        pickle.dump(yesno_aug_dataset, f)
else:
    with open('./aug_data/v2_yesno_aug_dataset_low_quality.pkl', 'wb') as f:
        pickle.dump(yesno_aug_dataset, f)