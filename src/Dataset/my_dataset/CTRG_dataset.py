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

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        image = torch.from_numpy(image).float()

        answer = self.data_list[index]['finding']
        # question = random.choice(self.caption_prompts)
        cls_labels = torch.where(cls_labels == 1, cls_labels, torch.tensor(0))

        prompt = ""
        for i, l in enumerate(cls_labels):
            disease = CONDITIONS[i]
            if l == 1:
                state = 'positive'
                prompt += f"The \"{disease}\" is {state}. "
                # prompt += "<POS>"
            else:
                state = 'negative'
                prompt += f"The \"{disease}\" is {state}. "
                # prompt += "<NEG>"
        
        # guide = "Based on the above image embedding information and some significant key information related to specific diseases, please generate a complete medical report corresponding to CT images. The generated report needs to meet the following requirements. A thorough examination of the anatomical structures is essential. Begin with the thoracic region, assessing the lungs for any abnormalities such as lung parenchyma changes, bronchial abnormalities, or the presence of nodules, masses, or lung lesions. Evaluate the thoracic cavity and chest wall for signs of pleural effusion, pneumothorax, pleural thickening, or rib fractures. Moving to the heart and major blood vessels, assess their morphology and function, noting any abnormalities in cardiac chambers, wall thickness, pericardial effusion, or the diameter and morphology of the aorta and pulmonary arteries. Evaluate the mediastinum for lymph node size, shape, and enlargement, paying attention to any abnormal lymph node enlargement or metastatic involvement. In the abdomen, examine the liver for size, shape, and density, observing for any focal lesions or masses. Assess the kidneys for symmetry, looking for signs of hydronephrosis or renal calculi. Evaluate the thyroid gland for size, shape, and the presence of nodules or other abnormalities. Examine the breasts for any masses or architectural distortions. Lastly, evaluate the brain for any intracranial abnormalities, such as hemorrhages or masses. Examine the brain parenchyma for signs of edema or infarction. In summary, you should examine the anatomical structures of the chest, abdomen, neck, and brain. Document any abnormalities, including the location, size, shape, and density characteristics of lesions or pathologies. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential. "
        guide = "Based on the above visual information and some key abnormal information related to specific diseases, please generate a complete medical report corresponding to this CT image. Providing a comprehensive report which contains detailed information about the anatomical structures and any abnormalities is essential. "
        question = prompt + guide
        
        image_dict = {
                "image": image,
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
            "image_dict": [image_dict],
            "guide": guide,
            "question": question,
            "answer": answer,
            "cls_labels": cls_labels,
        }

#TODO: add a new dataset class for CTRG dataset with report (combine finding and impression)
class CTRG_Report_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question": question, # random choice of report prompts
            "answer":answer, # caption
            }
    """

    def __init__(self, data_dir, csv_path, prompt_json_file, down_sample_ratio=5):
        self.data_dir = data_dir
        data_info = pd.read_csv(csv_path)
        self.down_sample_ratio = down_sample_ratio
        self.img_path_list = np.asarray(data_info['image_path'])
        self.impression_list = np.asarray(data_info['impression'])
        self.finding_list = np.asarray(data_info['finding'])
        self.id_list = np.asarray(data_info['id'])
        with open(prompt_json_file, 'r') as f:
            self.report_prompts = json.load(f)['caption_prompt']

    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] < image.shape[2]:
                image = image.transpose(1, 2, 0)
            # print('before resize',image.shape)
            image = cv2.resize(image, (256, 256),
                               interpolation=cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis, :, :, :]
            image = np.concatenate([image, image, image], axis=0)

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 256/image.shape[1], 256/image.shape[2], 64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(
                image, (3/image.shape[0], 256/image.shape[1], 256/image.shape[2], 1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.img_path_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.img_path_list)
        img_path = self.img_path_list[index]
        img_path = os.path.normpath(os.path.join(self.data_dir, img_path))
        image_id = self.id_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            image = np.random.randn(3, 256, 256, 64)

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3, 256, 256, 64)
        image = torch.from_numpy(image).float()

        answer = 'Impression: ' + self.impression_list[index] + ' Finding: ' + self.finding_list[index]
        question = random.choice(self.report_prompts)

        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question": len(question)
                }
            }
        return {
            "image_dict": [image_dict],
            "question": question,
            "answer": answer,
        }, image_id

class CTRG_Impression_Finding_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question1": question, # random choice of impression prompts
            impression":impression, # impression, used as answer1 and instruction
            "question2": question, # random choice of finding prompts
            "finding":finding, # finding, used as answer2
            }
    """

    def __init__(self, data_dir, csv_path, prompt_json_file, down_sample_ratio=5):
        self.data_dir = data_dir
        data_info = pd.read_csv(csv_path)
        self.down_sample_ratio = down_sample_ratio
        self.img_path_list = np.asarray(data_info['image_path'])
        self.impression_list = np.asarray(data_info['impression'])
        self.finding_list = np.asarray(data_info['finding'])
        self.id_list = np.asarray(data_info['id'])
        with open(prompt_json_file, 'r') as f:
            prompts = json.load(f)
        self.impression_prompts = prompts['impression_prompt']
        self.finding_prompts = prompts['finding_prompt']

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

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.img_path_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.img_path_list)
        img_path = self.img_path_list[index]
        img_path = os.path.normpath(os.path.join(self.data_dir, img_path))
        image_id = self.id_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            image = np.random.randn(3, 512, 512, 4)

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3, 512, 512, 4)
        image = torch.from_numpy(image).float()

        finding = self.finding_list[index]
        impression = self.impression_list[index]
        question1 = random.choice(self.impression_prompts)
        question2 = random.choice(self.finding_prompts)

        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question1": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question1": len(question1)
                }
            }

        if random.random() < 0.5:
            image_dict['position'].update({"question2": 0})
        else:
            image_dict['position'].update(
                {"question2": len(question2)})

        return {
            "image_dict": [image_dict],
            "question1": question1,
            "impression": impression,
            "question2": question2,
            "finding": finding,
        }, image_id

class CTRG_Impression_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question2": question, # random choice of finding prompts
            "finding":finding, # finding, used as answer2
            }
    """

    def __init__(self, data_dir, csv_path, prompt_json_file, down_sample_ratio=5):
        self.data_dir = data_dir
        data_info = pd.read_csv(csv_path)
        self.down_sample_ratio = down_sample_ratio
        self.img_path_list = np.asarray(data_info['image_path'])
        self.impression_list = np.asarray(data_info['impression'])
        self.id_list = np.asarray(data_info['id'])
        with open(prompt_json_file, 'r') as f:
            prompts = json.load(f)
        self.impression_prompts = prompts['impression_prompt']

    def resize_image(self, image):
        if len(image.shape) == 3:
            if image.shape[0] < image.shape[2]:
                image = image.transpose(1, 2, 0)
            # print('before resize',image.shape)
            # image = cv2.resize(image, (512, 512),
            #                    interpolation=cv2.INTER_LINEAR)
            # print('after resize',image.shape)
            image = image[np.newaxis, :, :, :]
            image = np.concatenate([image, image, image], axis=0)

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.img_path_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.img_path_list)
        img_path = self.img_path_list[index]
        img_path = os.path.normpath(os.path.join(self.data_dir, img_path))
        image_id = self.id_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            print('load image error')

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            print('image contain nan')
        image = torch.from_numpy(image).float()

        impression = self.impression_list[index]
        question1 = random.choice(self.impression_prompts)

        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question1": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question1": len(question1)
                }
            }

        return {
            "image_dict": [image_dict],
            "question1": question1,
            "impression": impression,
        }, image_id

class CTRG_Finding_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question2": question, # random choice of finding prompts
            "impression": impression, # impression, used as instruction
            "finding":finding, # finding, used as answer2
            }
    """

    def __init__(self, data_dir, csv_path, prompt_json_file, impression_csv=None, down_sample_ratio=5):
        self.data_dir = data_dir
        data_info = pd.read_csv(csv_path)
        self.down_sample_ratio = down_sample_ratio
        self.img_path_list = np.asarray(data_info['image_path'])
        self.finding_list = np.asarray(data_info['finding'])
        if impression_csv is None:
            self.impression_list = np.asarray(data_info['impression'])
        else:
            impression_info = pd.read_csv(impression_csv)
            self.impression_list = np.asarray(impression_info['Impression_Pred'])
        self.id_list = np.asarray(data_info['id'])
        with open(prompt_json_file, 'r') as f:
            prompts = json.load(f)
        self.finding_prompts = prompts['finding_prompt']

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

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.img_path_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.img_path_list)
        img_path = self.img_path_list[index]
        img_path = os.path.normpath(os.path.join(self.data_dir, img_path))
        image_id = self.id_list[index]
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            image = np.random.randn(3, 512, 512, 4)

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3, 512, 512, 4)
        image = torch.from_numpy(image).float()

        finding = self.finding_list[index]
        question2 = random.choice(self.finding_prompts)
        impression = self.impression_list[index]

        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question2": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question2": len(question2)
                }
            }

        return {
            "image_dict": [image_dict],
            "question2": question2,
            "impression": impression,
            "finding": finding,
        }, image_id

class CTRG_All_Dataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_: modality asked task formulated as vqa task for Radiopaedia dataset
        csv_path (_type_): path to csv file
        prompt_json_file (_type_): path to json file containing caption prompts
    Output:
        Dict: {
             "image_dict": [{"image": image, "position": {"question": 0}}], # image is a tensor of shape [c,w,h,d] like, [3,512,512,4], position is a dict, random choice of 0 or len(question)
            "question1": question, # random choice of impression prompts
            "impression":impression, # impression, used as answer1 and instruction
            "label_impression": labeled_impression, # impression with label
            "question2": question, # random choice of finding prompts
            "finding":finding, # finding, used as answer2
            }
    """

    def __init__(self, data_dir, data_json, prompt_json_file, split, down_sample_ratio=5):
        self.data_dir = data_dir
        self.down_sample_ratio = down_sample_ratio
        # self.img_path_list = np.asarray(data_info['image_path'])
        # self.impression_list = np.asarray(data_info['impression'])
        # self.finding_list = np.asarray(data_info['finding'])
        # self.id_list = np.asarray(data_info['id'])
        with open(data_json, 'r') as f:
            data_dict = json.load(f)
        self.data_list = data_dict[split]
        with open(prompt_json_file, 'r') as f:
            prompts = json.load(f)
        self.impression_prompts = prompts['impression_prompt']
        self.finding_prompts = prompts['finding_prompt']

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

        if image.shape[-1] > 64:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 64/image.shape[3]), order=0)
        else:
            image = ndimage.zoom(
                image, (3/image.shape[0], 512/image.shape[1], 512/image.shape[2], 1), order=0)
        return image

    def __len__(self):
        return math.ceil(len(self.data_list)/self.down_sample_ratio)

    def __getitem__(self, index):
        index = (self.down_sample_ratio*index + random.randint(0,
                 self.down_sample_ratio-1)) % len(self.data_list)
        img_path = self.data_list[index]['volume_path']
        img_path = os.path.normpath(os.path.join(self.data_dir, img_path))
        image_id = self.data_list[index]['id']
        try:
            itk_image = sitk.ReadImage(img_path)
            image = sitk.GetArrayFromImage(itk_image)
            image = self.resize_image(image)
        except:
            raise Exception('load image error')

        # image = np.load(img_path) # c,w,h,d
        image = (image-image.min())/(image.max()-image.min())
        contain_nan = (True in np.isnan(image))
        if contain_nan:
            image = np.random.randn(3, 512, 512, 4)
        image = torch.from_numpy(image).float()

        finding = self.data_list[index]['finding']
        impression = self.data_list[index]['impression']
        label_impression = self.data_list[index]['label_impression']
        question1 = random.choice(self.impression_prompts)
        question2 = random.choice(self.finding_prompts)

        if random.random() < 0.5:
            image_dict = {
                "image": image,
                "position": {
                    "question1": 0
                }
            }
        else:
            image_dict = {
                "image": image,
                "position": {
                    "question1": len(question1)
                }
            }

        if random.random() < 0.5:
            image_dict['position'].update({"question2": 0})
        else:
            image_dict['position'].update(
                {"question2": len(question2)})

        return {
            "id": image_id,
            "image_dict": [image_dict],
            "question1": question1,
            "impression": impression,
            "label_impression": label_impression,
            "question2": question2,
            "finding": finding,
        }
