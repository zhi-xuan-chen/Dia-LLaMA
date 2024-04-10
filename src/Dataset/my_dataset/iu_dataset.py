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


class IU_Dataset(Dataset):
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

        self.label = self._load_label(data_json)
    
    def _load_label(self, label_path):
        label_dict = {}

        with open(label_path, 'r') as f:
            data = json.load(f)
            for idx in data['train']:
                label_dict[idx['id']] = idx['labels']
            for idx in data['val']:
                label_dict[idx['id']] = idx['labels']
            for idx in data['test']:
                label_dict[idx['id']] = idx['labels']

        return label_dict
        
    def resize_image(self, image):
        # print('before resize',image.shape)
        image = cv2.resize(image, (224, 224),
                            interpolation=cv2.INTER_LINEAR)
        image = image.transpose(2, 0, 1)
        # print('after resize',image.shape)
        image = image[:, :, :, np.newaxis]
        image = np.concatenate([image, image, image, image], axis=-1)
            
        return image

    def __len__(self):
        return math.ceil(len(self.data_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.data_list)
        image_id = self.data_list[index]['id'] 
        cls_labels = torch.tensor(self.label[image_id], dtype=torch.float32).long()
        path0 = self.data_list[index]['image_path'][0]
        path1 = self.data_list[index]['image_path'][1]
        img_path0 = os.path.join(self.data_dir, path0)
        img_path1 = os.path.join(self.data_dir, path1)

        # 读取图像
        image0 = PIL.Image.open(img_path0)
        image1 = PIL.Image.open(img_path1)

        # 转换为 NumPy 数组
        image0 = np.array(image0)
        image0 = self.resize_image(image0)

        image1 = np.array(image1)
        image1 = self.resize_image(image1)

        # image = np.load(img_path) # c,w,h,d
        image0 = (image0-image0.min())/(image0.max()-image0.min())
        image0 = torch.from_numpy(image0).float()

        image1 = (image1-image1.min())/(image1.max()-image1.min())
        image1 = torch.from_numpy(image1).float()

        answer = self.data_list[index]['report']
        # question = random.choice(self.caption_prompts)
        cls_labels = torch.where(cls_labels == 1, cls_labels, torch.tensor(0))

        prompt = ""
        # for i, l in enumerate(cls_labels):
        #     disease = CONDITIONS[i]
        #     if l == 1:
        #         state = 'positive'
        #         prompt += f"The \"{disease}\" is {state}. "
        #     else:
        #         state = 'negative'
        #         prompt += f"The \"{disease}\" is {state}. "
        
        # guide = "Based on the above image embedding information and some significant key information related to specific diseases, please generate a complete medical report corresponding to CT images. The generated report needs to meet the following requirements. A thorough examination of the anatomical structures is essential. Begin with the thoracic region, assessing the lungs for any abnormalities such as lung parenchyma changes, bronchial abnormalities, or the presence of nodules, masses, or lung lesions. Evaluate the thoracic cavity and chest wall for signs of pleural effusion, pneumothorax, pleural thickening, or rib fractures. Moving to the heart and major blood vessels, assess their morphology and function, noting any abnormalities in cardiac chambers, wall thickness, pericardial effusion, or the diameter and morphology of the aorta and pulmonary arteries. Evaluate the mediastinum for lymph node size, shape, and enlargement, paying attention to any abnormal lymph node enlargement or metastatic involvement. In the abdomen, examine the liver for size, shape, and density, observing for any focal lesions or masses. Assess the kidneys for symmetry, looking for signs of hydronephrosis or renal calculi. Evaluate the thyroid gland for size, shape, and the presence of nodules or other abnormalities. Examine the breasts for any masses or architectural distortions. Lastly, evaluate the brain for any intracranial abnormalities, such as hemorrhages or masses. Examine the brain parenchyma for signs of edema or infarction. In summary, you should examine the anatomical structures of the chest, abdomen, neck, and brain. Document any abnormalities, including the location, size, shape, and density characteristics of lesions or pathologies. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential. "
        guide = "Based on the above visual information and some key abnormal information related to specific diseases, please generate a complete medical report corresponding to these Chest X-ray images. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential. "
        question = prompt + guide
        
        image_dict0 = {
                "image": image0,
                "position": {
                    "question": 0
                }
            }
        
        image_dict1 = {
                "image": image1,
                "position": {
                    "question": 0
                }
            }

        # if random.random() < 0.5:
        #     image_dict = {
        #         "image": image,
        #         "position": {
        #             "question": 0
        #         }
        #     }
        # else:
        #     image_dict = {
        #         "image": image,
        #         "position": {
        #             "question": len(question)
        #         }
        #     }

        return {
            "image_id": image_id,
            "image_dict": [image_dict0, image_dict1],
            "guide": guide,
            "question": question,
            "answer": answer,
            "cls_labels": cls_labels,
        }