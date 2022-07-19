# [ECCV'22] Rethinking Data Augmentation for Robust Visual Question Answering

This repo contains codes for our paper ["Rethinking Data Augmentation for Robust Visual Question Answering"](https://arxiv.org/abs/2207.08739). 

We followed [CSS-VQA](https://github.com/yanxinzju/CSS-VQA) to finish our codes, many thanks!

## Prerequisites

```
Python 3.6 with 
h5py == 3.1.0
torch == 1.9.0
click == 7.1.2
numpy == 1.19.2
tqdm == 4.60.0
```

## Data Setup 

### Setup VQA v2 and VQA-CP v2
1. Download data
```
bash tools/download.sh
```

2. Download faster rcnn features

Download `feature1.zip` and `feature2.zip` from [Google Drive](https://drive.google.com/drive/folders/1v0alq1zD4DuMCwlvBkikC96LuYIGbTq_), then unzip and merge them into `data/rcnn_feature/`.

3. Process data
```
bash tools/process.sh
```

## KDDAug 

## Codes coming soon.
