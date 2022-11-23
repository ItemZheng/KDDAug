from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import pickle
from collections import Counter

import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset import Dictionary


def _load_dataset(dataroot, datapath):
    """
    Load entries
    """
    aug_datapath = os.path.join(dataroot, datapath)
    with open(aug_datapath, 'rb') as f:
        aug_dataset = pickle.load(f)
    # load dataset
    entries = []
    for data in aug_dataset:
        entry = {
            'question_id': data['q_id'],
            'image_id': data['img_id'],
            'question': data['question'],
            'answer': {
                "labels": data['answer_text'],
                "scores": data['scores']
            }
        }
        if 'qtype' in data.keys():
            entry['qtype'] = data['qtype']
        if 'logits' in data.keys():
            entry['logits'] = data['logits']
        entries.append(entry)
    return entries


class VQAAugFeatureDataset(Dataset):
    path = {
        "cpv2": {
            "ori": "original_dataset.pkl",

            "yesno": "cpv2_yesno_aug_dataset.pkl",
            "color": "cpv2_color_aug_dataset.pkl",
            "number": "cpv2_number_aug_dataset.pkl",
            "other": "cpv2_other_aug_dataset.pkl",
            "paraphrasing": "cpv2_paraphrasing_aug_dataset.pkl",

            "sub": "cpv2_sub_aug_dataset.pkl",
            "all": "cpv2_all_aug_dataset.pkl",
            "total": "cpv2_total_aug_dataset.pkl",
            "paired": "cpv2_paired_aug_dataset.pkl",

            "low_ori": "original_dataset.pkl",
            "low_paraphrasing": "cpv2_paraphrasing_aug_dataset_low_quality.pkl",
            "low_number": "cpv2_number_aug_dataset_low_quality.pkl",
            "low_color": "cpv2_color_aug_dataset_low_quality.pkl",
            "low_other": "cpv2_other_aug_dataset_low_quality.pkl",
            "low_yesno": "cpv2_yesno_aug_dataset_low_quality.pkl",
            # "low_paired": "cpv2_paired_aug_dataset_low_quality.pkl",

            "high_paraphrasing": "cpv2_paraphrasing_aug_dataset_high_quality.pkl",
            "high_number": "cpv2_number_aug_dataset_high_quality.pkl",
            "high_color": "cpv2_color_aug_dataset_high_quality.pkl",
            "high_other": "cpv2_other_aug_dataset_high_quality.pkl",
            "high_yesno": "cpv2_yesno_aug_dataset_high_quality.pkl",

            "clean_ori": "original_dataset_clean.pkl",
            "clean_paraphrasing": "cpv2_paraphrasing_aug_dataset_low_quality_clean.pkl",
            "clean_number": "cpv2_number_aug_dataset_low_quality_clean.pkl",
            "clean_color": "cpv2_color_aug_dataset_low_quality_clean.pkl",
            "clean_other": "cpv2_other_aug_dataset_low_quality_clean.pkl",
            "clean_yesno": "cpv2_yesno_aug_dataset_low_quality_clean.pkl",
            # "clean_paired": "cpv2_paired_aug_dataset_low_quality_clean.pkl",

            # "missing_paraphrasing": "cpv2_paraphrasing_aug_dataset_missing.pkl",
            # "missing_number": "cpv2_number_aug_dataset_missing.pkl",
            # "missing_color": "cpv2_color_aug_dataset_missing.pkl",
            # "missing_other": "cpv2_other_aug_dataset_missing.pkl",
            # "missing_yesno": "cpv2_yesno_aug_dataset_missing.pkl",

            # "very_low_paraphrasing": "cpv2_paraphrasing_aug_dataset_very_low_quality.pkl",
            # "very_low_number": "cpv2_number_aug_dataset_very_low_quality.pkl",
            # "very_low_color": "cpv2_color_aug_dataset_very_low_quality.pkl",
            # "very_low_other": "cpv2_other_aug_dataset_very_low_quality.pkl",

            "high_clean_paraphrasing": "cpv2_paraphrasing_aug_dataset_high_quality_clean.pkl",
            "high_clean_number": "cpv2_number_aug_dataset_high_quality_clean.pkl",
            "high_clean_color": "cpv2_color_aug_dataset_high_quality_clean.pkl",
            "high_clean_other": "cpv2_other_aug_dataset_high_quality_clean.pkl",
            "high_clean_yesno": "cpv2_yesno_aug_dataset_high_quality_clean.pkl",
        },
        "v2": {
            "ori": "v2_original_dataset.pkl",

            "yesno": "v2_yesno_aug_dataset.pkl",
            "color": "v2_color_aug_dataset.pkl",
            "number": "v2_number_aug_dataset.pkl",
            "other": "v2_other_aug_dataset.pkl",
            "paraphrasing": "v2_paraphrasing_aug_dataset.pkl",

            "all": "v2_all_aug_dataset.pkl",
            "total": "v2_total_aug_dataset.pkl",
            "paired": "v2_paired_aug_dataset.pkl",

            "low_ori": "v2_original_dataset.pkl",
            "low_paraphrasing": "v2_paraphrasing_aug_dataset_low_quality.pkl",
            "low_number": "v2_number_aug_dataset_low_quality.pkl",
            "low_color": "v2_color_aug_dataset_low_quality.pkl",
            "low_other": "v2_other_aug_dataset_low_quality.pkl",
            "low_yesno": "v2_yesno_aug_dataset_low_quality.pkl",
            "low_paired": "v2_paired_aug_dataset_low_quality.pkl",

            "high_paraphrasing": "v2_paraphrasing_aug_dataset_high_quality.pkl",
            "high_number": "v2_number_aug_dataset_high_quality.pkl",
            "high_color": "v2_color_aug_dataset_high_quality.pkl",
            "high_other": "v2_other_aug_dataset_high_quality.pkl",

            "clean_ori": "v2_original_dataset_clean.pkl",
            "clean_paraphrasing": "v2_paraphrasing_aug_dataset_low_quality_clean.pkl",
            "clean_number": "v2_number_aug_dataset_low_quality_clean.pkl",
            "clean_color": "v2_color_aug_dataset_low_quality_clean.pkl",
            "clean_other": "v2_other_aug_dataset_low_quality_clean.pkl",
            "clean_yesno": "v2_yesno_aug_dataset_low_quality_clean.pkl",
            "clean_paired": "v2_paired_aug_dataset_low_quality_clean.pkl",

            # "missing_paraphrasing": "v2_paraphrasing_aug_dataset_missing.pkl",
            # "missing_number": "v2_number_aug_dataset_missing.pkl",
            # "missing_color": "v2_color_aug_dataset_missing.pkl",
            # "missing_other": "v2_other_aug_dataset_missing.pkl",

            "high_clean_paraphrasing": "v2_paraphrasing_aug_dataset_high_quality_clean.pkl",
            "high_clean_number": "v2_number_aug_dataset_high_quality_clean.pkl",
            "high_clean_color": "v2_color_aug_dataset_high_quality_clean.pkl",
            "high_clean_other": "v2_other_aug_dataset_high_quality_clean.pkl",
        }
    }

    def __init__(self, name, dictionary, dataroot='aug_data', dataset='cpv2', cache_image_features=True, datapath=None):
        super(VQAAugFeatureDataset, self).__init__()
        assert name in self.path[dataset].keys()
        self.name = name
        if datapath is None:
            datapath = self.path[dataset][name]
            print("Load Data From: ", datapath)
        if dataset == 'cpv2':
            ans2label_path = os.path.join('data', 'cp-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join('data', 'cp-cache', 'trainval_label2ans.pkl')
        elif dataset == 'cpv1':
            ans2label_path = os.path.join('data', 'cp-v1-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join('data', 'cp-v1-cache', 'trainval_label2ans.pkl')
        elif dataset == 'v2':
            ans2label_path = os.path.join('data', 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join('data', 'cache', 'trainval_label2ans.pkl')
        else:
            ans2label_path = os.path.join('data', 'cp-cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join('data', 'cp-cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary

        self.entries = _load_dataset(dataroot, datapath)
        if cache_image_features:
            image_to_fe = {}
            for entry in tqdm(self.entries, ncols=100, desc="caching-features"):
                img_id = entry["image_id"]
                if img_id not in image_to_fe:
                    fe = torch.load('data/rcnn_feature/' + str(img_id) + '.pth', encoding='iso-8859-1')['image_feature']
                    image_to_fe[img_id] = fe
            self.image_to_fe = image_to_fe
        else:
            self.image_to_fe = None

        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries, ncols=100, desc="tokenize"):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in tqdm(self.entries, ncols=100, desc="tensorize"):
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if len(answer['labels']):
                labels = np.array([self.ans2label[answer['labels'][i]] for i in range(len(answer['labels']))])
                scores = np.array(answer['scores'], dtype=np.float32)
            else:
                labels = []
                scores = []
            if len(labels) > 0:
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
            
            if 'logits' in entry.keys():
                entry['logits'] = torch.from_numpy(entry['logits'])

    def __getitem__(self, index):
        entry = self.entries[index]
        if self.image_to_fe is not None:
            features = self.image_to_fe[entry["image_id"]]
        else:
            features = torch.load('data/rcnn_feature/' + str(entry["image_id"]) + '.pth',
                                  encoding='iso-8859-1')['image_feature']

        # load data
        ques = entry['q_token']

        if 'logits' in entry.keys():
            target = entry['logits']
        else:
            answer = entry['answer']
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)

        if "bias" in entry:
            return features, ques, target, entry["bias"]

        else:
            return features, ques, target, 0

    def __len__(self):
        return len(self.entries)
