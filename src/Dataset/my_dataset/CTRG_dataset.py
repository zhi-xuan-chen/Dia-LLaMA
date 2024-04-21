import csv
import json
import logging
import os
import re
import difflib
import sys
import cv2
import torch
import random
from abc import abstractmethod
from itertools import islice
from scipy import ndimage
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections.abc import Mapping
from torch.utils.data import DataLoader
import PIL
import SimpleITK as sitk
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import math
from monai.transforms import RandSpatialCrop

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'<BLA>',
'<POS>',
'<NEG>',
'<UNC>'
]

Token_to_Text = {
    '<BLA>': 'blank',
    '<POS>': 'positive',
    '<NEG>': 'negative',
    '<UNC>': 'uncertain'
}


class CTRG_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of caption prompts
            "answer":answer, # caption
            }
    """

    def __init__(self, data_dir, data_json, split, prompt_json_file, down_sample_ratio=5):
        self.data_dir = data_dir
        self.split = split

        with open(data_json, 'r') as f:
            data_dict = json.load(f)
        self.data_list = data_dict[split]
        self.down_sample_ratio = down_sample_ratio

        with open(prompt_json_file, 'r') as f:
            self.caption_prompts = json.load(f)['caption_prompt']

        self.label = self._load_label('/home/chenzhixuan/Workspace/M2KT/data_csv/CTRG_finding_labels.csv')
    
    def _load_label(self, label_path):
        label_dict = {}

        data = pd.read_csv(label_path)
        for index, row in data.iterrows():
            idx = row['id']
            label = row[1:].to_list()

            label_dict[idx] = label

        return label_dict
        
    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] < image.shape[2]:
                image = image.transpose(1, 2, 0)
            # print('before resize',image.shape)
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis, :, :, :]
            image = np.concatenate([image, image, image], axis=0)
        else:
            print('image shape',image.shape)

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            print('image shape',image.shape)
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
            
        return image

    def __len__(self):
        return math.ceil(len(self.data_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.data_list)
        image_id = self.data_list[index]['id'] 
        cls_labels = torch.tensor(self.label[int(image_id)], dtype=torch.float32).long()
        img_path = os.path.normpath(os.path.join(self.data_dir, str(image_id)+'.nii.gz'))

        itk_image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(itk_image)
        image = self.resize_image(image)

        image = np.array(image)

        image = (image-image.min())/(image.max()-image.min())
        image = torch.from_numpy(image).float()

        answer = self.data_list[index]['finding']
        cls_labels = torch.where(cls_labels == 1, cls_labels, torch.tensor(0))

        prompt = ""
        for i, l in enumerate(cls_labels):
            disease = CONDITIONS[i]
            if l == 1:
                state = 'positive'
                prompt += f"The \"{disease}\" is {state}. "
            else:
                state = 'negative'
                prompt += f"The \"{disease}\" is {state}. "
        
        guide = "Based on the above visual information and some key abnormal information related to specific diseases, please generate a complete medical report corresponding to this CT image. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential. "
        question = prompt + guide
        
        image_dict = {
                "image": image,
                "position": {
                    "question": 0
                }
            }

        return {
            "image_id": image_id,
            "image_dict": [image_dict],
            "guide": guide,
            "question": question,
            "answer": answer,
            "cls_labels": cls_labels,
        }