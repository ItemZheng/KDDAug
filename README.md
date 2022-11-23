# [ECCV'22] Rethinking Data Augmentation for Robust Visual Question Answering

This repo contains codes for our paper ["Rethinking Data Augmentation for Robust Visual Question Answering"](https://arxiv.org/abs/2207.08739). 

We followed [CSS-VQA](https://github.com/yanxinzju/CSS-VQA) to finish our codes, many thanks!

## Prerequisites

Make sure you are on a machine with a NVIDIA GPU and about 100 GB disk space.
```
Python 3.6 with 
h5py == 3.1.0
torch == 1.9.0
click == 7.1.2
numpy == 1.19.2
tqdm == 4.60.0
transformers == 4.8.2
clip == 1.0
```

## Setup data: VQA v2 and VQA-CP v2

1. Download data

```
bash tools/download.sh
```

2. Download faster rcnn features

Download `feature1.zip` and `feature2.zip` from [Google Drive](https://drive.google.com/drive/folders/1v0alq1zD4DuMCwlvBkikC96LuYIGbTq_), then unzip and merge them into `data/rcnn_feature/`.

3. Download Images (For CLIP-based Filtering)

Create `Images` folder and download coco images.

train2014：[http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/train2014.zip)

val2014：[http://images.cocodataset.org/zips/train2014.zip](http://images.cocodataset.org/zips/val2014.zip)

4. Process data
```
bash tools/process.sh
```

**Data processing results may be in-consistent due to the inconsistency of python versions. To use our pretrained models, you can download the process results from [here](https://drive.google.com/drive/folders/137c1C9Msg-akDoA5PWlouYO1Qye05w5i?usp=share_link). Move them to folder `data`.**


4. Download extra data to train CSS (ID & OOD Teacher in KDDAug)

Download `*hintscore.json` files from [here](https://drive.google.com/drive/folders/1v0alq1zD4DuMCwlvBkikC96LuYIGbTq_), and move them to `data` folder. 


## KDDAug 

### Prepare

1. Create `aug_data` folder to save augmented data.

2. For convenience, process original dataset by following steps:

+ Prepare Original IQA Triplets
+ Prepare Faster RCNN Detection Data
+ Extract Nouns of Question.

Run command: 
```
python process_original_dataset.py --dataset cpv2
python process_original_dataset.py --dataset v2
```

Example data after processing:

```
{   
    # IQA triplets
    'q_id': 9001, 
    'img_id': 9, 
    'question': 'What color are the dishes?', 
    'answer_text': ['pink and yellow'], 
    'scores': [0.9], 

    # Faster RCNN Detection Results 
    'objects': ['broccoli', 'donut', 'container', 'meat', 'container', 'bowl', 'food'], 
    'attributes': ['green', '', 'plastic', '', '', '', ''], 

    # Meaningful nouns in Question
    'nouns': ['dish']
}
```

3. Extract question features (For generate Paraphrasing questions).

```
CUDA_VISIBLE_DEVICES=0 python extract_question_feature.py --dataset cpv2
CUDA_VISIBLE_DEVICES=0 python extract_question_feature.py --dataset v2
```

4. Extract CLIP features for images (For CLIP-based Filtering).

```
CUDA_VISIBLE_DEVICES=0 python extract_clip_feature.py --dataset cpv2
CUDA_VISIBLE_DEVICES=0 python extract_clip_feature.py --dataset v2
```


### Image-Question Composition with Initial Answer
1. Yes/No Questions.
```
python generate_yesno.py --dataset cpv2
python generate_yesno.py --dataset v2
```

2. Other Questions 

```
python generate_other.py --dataset cpv2
python generate_other.py --dataset v2
```

3. Color Questions

```
python generate_color.py --dataset cpv2
python generate_color.py --dataset v2
```

4. Number Questions

```
python generate_number.py --dataset cpv2
python generate_number.py --dataset v2
```

5. Paraphrasing Questions
```
CUDA_VISIBLE_DEVICES=0 python generate_paraphrasing.py --dataset cpv2
CUDA_VISIBLE_DEVICES=0 python generate_paraphrasing.py --dataset v2
```

### Divide initial answer to High-Quality or Low-Quality(Mentioned in section 4.5)
```
CUDA_VISIBLE_DEVICES=0 python divide.py --dataset cpv2 --ratio 1.0
CUDA_VISIBLE_DEVICES=0 python divide.py --dataset v2 --ratio 1.0
```
`ratio` denotes high-quality ratio.

**Notice**: even if `ratio` set to `1.0`, the code still generate `low_quality_dataset.pkl` file.


### KD-based Answer Assignment

1. Pretrain a teacher model (CSS)
Download from [CSS-VQA](https://github.com/yanxinzju/CSS-VQA) or train a new LMH-CSS model using the command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset [cpv2/v2] --mode q_v_debias --debias learned_mixin --topq 1 --topv -1 --qvp 5 --output lmh_css --seed 2048
```

2. Assign new answer.
```
# number
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name number --split high --teacher_path []
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name number --split low --teacher_path []
# other
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name other --split high --teacher_path []
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name other --split low --teacher_path []
# color
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name color --split high --teacher_path []
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name color --split low --teacher_path []
# paraphrasing
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name paraphrasing --split high --teacher_path []
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name paraphrasing --split low --teacher_path []
# yesno
CUDA_VISIBLE_DEVICES=0 python assign_answer.py --dataset [cpv2/v2] --name yesno --split low --teacher_path []
```

### Merge all dataset
Merge all augmented data and save to `[cpv2/v2]_all_aug_dataset.pkl`.
```
python merge.py --dataset [cpv2/v2]
```

### CLIP-based Filtering
CLIP-based filtering and save to `[cpv2/v2]_total_aug_dataset.pkl`
```
CUDA_VISIBLE_DEVICES=0 python filter.py --ratio 0.1 --dataset [cpv2/v2]
```

## Train
1. Train Backbone models

**UpDn**

Run command:
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cpv2 --mode updn --debias none --output updn --seed 0
```
or download our pretrained `UpDn` model from [here](https://drive.google.com/drive/folders/1orw_hCdCmp69LUBdPBMR9VJTSmbVF8OV?usp=sharing)


**LMH-CSS+**

Download our pretrained `LMH-CSS+` model from [here](https://drive.google.com/drive/folders/1orw_hCdCmp69LUBdPBMR9VJTSmbVF8OV?usp=sharing)

2. Finetune on Augmented dataset

Use `[cpv2/v2]_all_aug_dataset.pkl` if `aug_name` set to `all`.

Use `[cpv2/v2]_total_aug_dataset.pkl` (after clip-based filtering) if `aug_name` set to `total`.

```
CUDA_VISIBLE_DEVICES=0 python aug_main.py --backbone ./path/to/model --aug_name all --dataset cpv2 --output [] --seed 0
```

Our KDDAug model is available [here](https://drive.google.com/drive/folders/1orw_hCdCmp69LUBdPBMR9VJTSmbVF8OV?usp=sharing)

