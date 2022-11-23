import torch
import clip
from PIL import Image
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

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# load original dataset 
import pickle
if dataset == 'cpv2':
    print('Load original data from: original_dataset.pkl')
    with open('./aug_data/original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
else:
    print('Load original data from: v2_original_dataset.pkl')
    with open('./aug_data/v2_original_dataset.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
print('Dataset size: ', len(original_dataset))

from tqdm import tqdm
# collect all image information
print('1. Collect image information')
image_info = {}
for i in tqdm(range(len(original_dataset)), ncols=100, total=len(original_dataset)):
    entry = original_dataset[i]
    img_id = entry['img_id']
    if image_info.get(img_id, None) is None:
        info = {
            'objects': entry['objects'],
            'attributes': entry['attributes']
        }
        image_info[img_id] = info


# get image path
print('2. Collect image path')
import os, json
if dataset == 'cpv2':
    # open file of questions and answers
    question_path = os.path.join('data', 'vqacp_v2_train_questions.json')
    with open(question_path) as f:
        questions = json.load(f)
else:
    # open file of questions and answers
    question_path = os.path.join('data', 'v2_OpenEnded_mscoco_train2014_questions.json')
    with open(question_path) as f:
        questions = json.load(f)["questions"]

# construct image path
for question in questions:
    if dataset == 'cpv2':
        coco_split = question['coco_split']
    else:
        coco_split = 'train2014'
    img_id = question['image_id']
    path = os.path.join(coco_split, 'COCO_%s_%012d.jpg' % (coco_split, img_id))
    image_info[img_id]['img_path'] = path


print('3. Extract image clip feature')
imgId_to_clip_feature_dict = {}
batch_size = 256
batch_image = []
batch_imgIds = []
for img_id in tqdm(list(image_info.keys()), total=len(image_info), ncols=80):
    path = os.path.join('Images', image_info[img_id]['img_path'])
    image = preprocess(Image.open(path)).to(device)
    # collect batch
    batch_image.append(image)
    batch_imgIds.append(img_id)

    if len(batch_image) >= batch_size:
        imgs = torch.stack(batch_image, dim=0)
        assert imgs.size(0) == batch_size
        with torch.no_grad():
            image_features = model.encode_image(imgs).cpu()
        for i in range(len(batch_imgIds)):
            id = batch_imgIds[i]
            feature = image_features[i]
            imgId_to_clip_feature_dict[id] = feature
        batch_imgIds = []
        batch_image = []

if len(batch_image) > 0:
    imgs = torch.stack(batch_image, dim=0)
    with torch.no_grad():
        image_features = model.encode_image(imgs).cpu()
    for i in range(len(batch_imgIds)):
        id = batch_imgIds[i]
        feature = image_features[i]
        imgId_to_clip_feature_dict[id] = feature
    batch_imgIds = []
    batch_image = []

if dataset == 'cpv2':
    with open('./aug_data/imgId_to_clip_feature_dict.pkl', 'wb') as f:
        pickle.dump(imgId_to_clip_feature_dict, f)
else:
    with open('./aug_data/v2_imgId_to_clip_feature_dict.pkl', 'wb') as f:
        pickle.dump(imgId_to_clip_feature_dict, f)

assert len(imgId_to_clip_feature_dict) == len(image_info)