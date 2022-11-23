# This code is to prepare Original IQA Triplets and extract meaningful nouns

# Original IQA Triplets:
#   <question_id, img_id, question, answer>
import os
import json
import pickle
import utils
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse
import spacy
import nltk
import nltk.stem as ns
from nltk.corpus import wordnet as wn

def parse_args():
    parser = argparse.ArgumentParser("Assign Label For Low Quality/High Quality DATA")

    parser.add_argument(
        '--dataset', default='cpv2',
        choices=["v2", "cpv2"],
        help="Run on VQA-2.0 instead of VQA-CP 2.0"
    )
    args = parser.parse_args()
    return args

args = parse_args()
dataset = args.dataset


#### Prepare Three Sources
#     - Original IQA Triplets
#     - MSCOCO Annotations
#     - Faster RCNN Detector Results 

print("Prepare Original IQA Triplets")


if dataset == 'cpv2':
    # open file of questions and answers
    answer_path = os.path.join('data', 'cp-cache', 'train_target.pkl')
    question_path = os.path.join('data', 'vqacp_v2_train_questions.json')
    with open(question_path) as f:
        questions = json.load(f)
    with open(answer_path, 'rb') as f:
        answers = pickle.load(f)
else:
    # open file of questions and answers
    answer_path = os.path.join('data', 'cache', 'train_target.pkl')
    question_path = os.path.join('data', 'v2_OpenEnded_mscoco_train2014_questions.json')
    with open(question_path) as f:
        questions = json.load(f)["questions"]
    with open(answer_path, 'rb') as f:
        answers = pickle.load(f)

# sort by question id
questions.sort(key=lambda x: x['question_id'])
answers.sort(key=lambda x: x['question_id'])
utils.assert_eq(len(questions), len(answers))

# load label to answer file
if dataset == 'cpv2':
    cache_file = os.path.join('data', 'cp-cache', 'trainval_label2ans.pkl')
    label2ans = pickle.load(open(cache_file, 'rb'))
else:
    cache_file = os.path.join('data', 'cache', 'trainval_label2ans.pkl')
    label2ans = pickle.load(open(cache_file, 'rb'))

original_dataset = []
# read data into memory
for question, answer in tqdm(zip(questions, answers), total=len(questions), ncols=80):
    if answer["labels"] is None:
        raise ValueError()
    utils.assert_eq(question['question_id'], answer['question_id'])
    utils.assert_eq(question['image_id'], answer['image_id'])
    q_id = question['question_id']
    img_id = question['image_id']
    q_text = question['question']
    ans = []
    score = []

    for i in range(len(answer['labels'])):
        ans.append(label2ans[answer['labels'][i]])
        score.append(answer['scores'][i])
    entry = {
        'q_id': q_id,
        'img_id': img_id,
        'question': q_text,
        'answer_text': ans,
        'scores': score
    }
    original_dataset.append(entry)

# show exsample
print('Example: ', original_dataset[1])



#####  Prepare Faster RCNN Detection Data (Filtered By Confidence of Detection)

print("\n\nPrepare Faster RCNN Detection Data")

# load vocabs of faster rcnn data result

data_path = './faster_rcnn_vocabs'

# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())

# prepare faster rcnn detection data
for entry in tqdm(original_dataset, total=len(original_dataset), ncols=100):
    img_id = entry['img_id']
    rcnn_feature = torch.load('data/rcnn_feature/' + str(img_id) + '.pth', encoding='iso-8859-1')
    roi = rcnn_feature['spatial_feature']
    bbox = roi[:, :4]
    cls_score = rcnn_feature['cls_score']
    attr_score = rcnn_feature['attr_score']

    attr_thresh = 0.4
    conf_thresh = 0.8

    object_texts = []
    attribute_texts = []
    for i in range(36):
        obj_idx = int(cls_score[i][0])
        if obj_idx > 0 and cls_score[i][1] > conf_thresh:
            object_texts.append(classes[obj_idx])
            attr_idx = int(attr_score[i][0])
            if attr_idx > 0:
                attribute_texts.append(attributes[attr_idx])
            else:
                attribute_texts.append('')
    entry['objects'] = object_texts
    entry['attributes'] = attribute_texts


# show example
print('Example with faster rcnn detection results: ', original_dataset[1])

print("\n\nExtract Nouns of Question")
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



nlp = spacy.load("en_core_web_sm")
lemmatizer = ns.WordNetLemmatizer()
for entry in tqdm(original_dataset, total=len(original_dataset), ncols=100):
    question = handle(entry['question'])
    qid = entry['q_id']
    qtype = qid2qtype[qid]
    meaningful_nouns = []
    ori_nouns = []

    # 1. remove question type
    question = question.strip()

    # 2. use spaCy POS to get meaningful words
    doc = nlp(question)
    qtype_doc = nlp(qtype)
    for i in range(len(qtype_doc), len(doc)):
        token = doc[i]
        if token.pos_ != 'NOUN' and token.pos_ != 'PROPN':
            continue
        # 3. tokenize singular and plural forms
        text = lemmatizer.lemmatize(token.text, pos=wn.NOUN)
        # 4. remove photo and picture
        if text == 'photo' or text == 'picture':
            continue
        meaningful_nouns.append(text)
        ori_nouns.append(token.text)
    entry['nouns'] = meaningful_nouns
    entry['ori_nouns'] = ori_nouns

# example
print('Example with meaningful nouns: ', original_dataset[1])
print('Dataset Size: ', len(original_dataset))

if dataset == 'cpv2':
    print('Saving to aug_data/original_dataset.pkl')
    with open('./aug_data/original_dataset.pkl', 'wb') as f:
        pickle.dump(original_dataset, f)
else:
    print('Saving to aug_data/v2_original_dataset.pkl')
    with open('./aug_data/v2_original_dataset.pkl', 'wb') as f:
        pickle.dump(original_dataset, f)
