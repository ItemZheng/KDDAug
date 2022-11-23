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


if dataset == 'cpv2':
    with open('./aug_data/question_info.pkl', 'rb') as f:
        question_info = pickle.load(f)
else:
    with open('./aug_data/v2_question_info.pkl', 'rb') as f:
        question_info = pickle.load(f)


embeddings = []
for question in tqdm(list(question_info.keys())):
    info = question_info[question]
    embeddings.append(list(info['embed'].numpy()))

torch_embedding = torch.tensor(embeddings).cuda()
unique_question = list(question_info.keys())

print('1. Calulate cosine similarity of questions')
with torch.no_grad():
    for i, question in tqdm(enumerate(unique_question), total=len(unique_question)):
        self_embedding = torch_embedding[i]
        similarity = torch.nn.functional.cosine_similarity(self_embedding.expand_as(torch_embedding), torch_embedding)
        similarity[similarity<=0.95] = 0
        k = torch.nonzero(similarity).size(dim=0)
        top_k_idx = torch.topk(similarity, k=k)[1].cpu().numpy()

        sim_questions = []
        for idx in top_k_idx:
            if idx == i:
                continue
            sim_questions.append(unique_question[idx])
            if len(sim_questions) >= 3:
                break
        question_info[question]['sim_questions'] = sim_questions


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


#%%
# handle sentence function
def handle(sentence:str):
    sentence = sentence.lower()
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').\
        replace('-',' ').replace('.','').replace('"', '').replace('n\'t', ' not').\
        replace('$', ' dollar ')
    return sentence



# 2. create exist qa triplets dict
print('2. Create exist qa triplets dict')
exist_triplets_dict = defaultdict(dict)
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    question = handle(entry['question'])
    exist_triplets_dict[question][entry['img_id']] = True



# 3. create new qa triplets dict
print('3. Create new qa triplets')
paraphrasing_aug_dataset = []
for entry in tqdm(original_dataset, ncols=100, total=len(original_dataset)):
    question = handle(entry['question'])

    info = question_info[question]
    sim_questions = info['sim_questions']
    for sim_question in sim_questions:
        if exist_triplets_dict[sim_question].get(entry['img_id'], False):
            continue
        new_entry = {
            'q_id': 'paraphrasing_aug_' + str(len(paraphrasing_aug_dataset)),     # assign new question id
            'img_id': entry['img_id'],
            'question': sim_question,
            'answer_text': entry['answer_text'],
            'scores': entry['scores'],
            'objects': entry['objects'],
            'attributes': entry['attributes'],
            'nouns': info['nouns'],
            'type': question_info[sim_question]['type'],
            'qtype': question_info[sim_question]['qtype']
        }
        exist_triplets_dict[sim_question][entry['img_id']] = True
        paraphrasing_aug_dataset.append(new_entry)


print('number augmented dataset:', len(paraphrasing_aug_dataset))


if dataset == 'cpv2':
    # laod chosen yesno aug dataset
    with open('./aug_data/cpv2_paraphrasing_aug_dataset.pkl', 'wb') as f:
        pickle.dump(paraphrasing_aug_dataset, f)
else:
    # laod chosen yesno aug dataset
    with open('./aug_data/v2_paraphrasing_aug_dataset.pkl', 'wb') as f:
        pickle.dump(paraphrasing_aug_dataset, f)