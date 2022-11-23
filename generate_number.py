
# imports
import os
import json
import pickle
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

#%%

# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence

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

#%% md

### Speed Up Techs

#%%

# 1. create exist qa triplets dict
print('1. Create exist qa triplets dict')
exist_triplets_dict = defaultdict(dict)
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    question = handle(entry['question'])
    exist_triplets_dict[question][entry['img_id']] = True

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

#%% md

#### Simple AUG(Number Question)

#%%
# 6. find number questions
print('6. Find Number questions')
questions = list(question_info.keys())
number_questions = []
for question in questions:
    # number question
    qtype = question_info[question]['type']
    if qtype != 'number':
        continue
    # only consider one noun, except number
    number_nouns = []
    for noun in question_info[question]['nouns']:
        if noun == 'number':
            continue
        number_nouns.append(noun)
    if len(number_nouns) != 1:
        continue
    question_info[question]['number_noun'] = number_nouns[0]
    number_questions.append(question)

#%%

# 2. first step veriﬁcation
valid_number_questions = []
for question in number_questions:
    idxs = question_info[question]['entry_idxs']
    number_noun = question_info[question]['number_noun']
    valid = True
    for idx in idxs:
        entry = original_dataset[idx]
        objects = entry['objects']
        answer = entry['answer_text']

        count = 0
        for obj in objects:
            if obj == number_noun:
                count += 1
        ans = str(count)

        if ans not in answer:
            valid = False
            break
    if valid:
        valid_number_questions.append(question)
print('Number Question Count: ', len(valid_number_questions))


#%%

print('6. Pair image-question pairs')
count = 0
for question in tqdm(valid_number_questions, total=len(valid_number_questions), ncols=80):
    info = question_info[question]
    number_noun = info['number_noun']

    returned_imgs = {}
    for img_id in obj2imgIds[number_noun]:
        if exist_triplets_dict[question].get(img_id, False):
            continue
        returned_imgs[img_id] = True
    info['returned_imgs'] = list(returned_imgs.keys())
    count = count + len(returned_imgs)
print('Number VQ Pairs:', count)

#%%

if dataset == 'cpv2':
    # load answer to label file
    cache_file = os.path.join('data', 'cp-cache', 'trainval_ans2label.pkl')
else:
    # load answer to label file
    cache_file = os.path.join('data', 'cache', 'trainval_ans2label.pkl')
ans2label = pickle.load(open(cache_file, 'rb'))

#%%

# Assign Answers
print('7. Assign initial answers and save')
number_aug_dataset = []
for question in tqdm(valid_number_questions, total=len(valid_number_questions), ncols=80):
    info = question_info[question]
    number_noun = info['number_noun']
    returned_imgs = info['returned_imgs']

    for img_id in returned_imgs:
        img_info = image_info[img_id]
        objects = img_info['objects']

        # assign answer
        count = 0
        for obj in objects:
            if obj == number_noun:
                count += 1
        ans = str(count)

        newEntry = {
            'q_id': 'number_aug_' + str(len(number_aug_dataset)),     # assign new question id
            'img_id': img_id,
            'question': question,
            'answer_text': [ans],
            'scores': [1.0],
            'objects': img_info['objects'],
            'attributes': img_info['attributes'],
            'nouns': info['nouns'],
            'qtype': info['qtype'],
        }

        # second step veriﬁcation
        valid = True
        for idx in img_info['entry_idxs']:
            entry_j = original_dataset[idx]
            qid_j = entry_j['q_id']
            if qid2type[qid_j] == 'number':
                nouns_j = entry_j['nouns']
                number_nouns = []
                for noun in nouns_j:
                    if noun == 'number':
                        continue
                    number_nouns.append(noun)
                if len(number_nouns) != 1:
                    continue
                if number_nouns[0] != number_noun:
                    continue
                ans = newEntry['answer_text'][0]
                if ans not in entry_j['answer_text']:
                    valid = False
                    break
        if valid and ans2label.get(ans, False):
            number_aug_dataset.append(newEntry)

print('number augmented dataset:', len(number_aug_dataset))

#%%
if dataset == 'cpv2':
    with open('./aug_data/cpv2_number_aug_dataset.pkl', 'wb') as f:
        pickle.dump(number_aug_dataset, f)
else:
    with open('./aug_data/v2_number_aug_dataset.pkl', 'wb') as f:
        pickle.dump(number_aug_dataset, f)
