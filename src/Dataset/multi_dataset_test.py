from torch.utils.data import Dataset
import numpy as np
import transformers
import pandas as pd
import copy
import random
import os
import numpy as np
import tqdm
import torch
import json
from PIL import Image
import math
import torchvision
from .my_dataset import *
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def stack_images(images):

    target_H = 256
    target_W = 256
    target_D = 4
    if len(images) == 0:
        return torch.zeros((1, 3, target_H, target_W, target_D))
    MAX_D = 4
    D_list = list(range(4, 65, 4))

    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
            target_D = temp_D

    stack_images = []
    for s in images:
        s = torch.tensor(s)
        if len(s.shape) == 3:
            # print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(
                s.unsqueeze(0).unsqueeze(-1), size=(target_H, target_W, target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(
                s.unsqueeze(0), size=(target_H, target_W, target_D)))
    images = torch.cat(stack_images, dim=0)
    return images


class multi_dataset_test(Dataset):
    def __init__(self, text_tokenizer, max_seq=2048, max_img_size=10, image_num=32, voc_size=32000):

        self.text_tokenizer = text_tokenizer
        self.max_img_size = max_img_size
        self.image_num = image_num
        self.max_seq = max_seq
        self.voc_size = voc_size
        self.H = 512
        self.W = 512
        self.image_padding_tokens = []
        if isinstance(self.text_tokenizer, str):
            self.text_tokenizer = LlamaTokenizer.from_pretrained(
                self.text_tokenizer,
            )
            special_token = {
                "additional_special_tokens": ["<image>", "</image>"]}
            for i in range(max_img_size):
                image_padding_token = ""
                for j in range(image_num):
                    image_token = "<image"+str(i*image_num+j)+">"
                    image_padding_token = image_padding_token + image_token
                    special_token["additional_special_tokens"].append(
                        "<image"+str(i*image_num+j)+">")
                self.image_padding_tokens.append(image_padding_token)
            self.text_tokenizer.add_special_tokens(
                special_token
            )
            self.text_tokenizer.pad_token_id = 0
            self.text_tokenizer.bos_token_id = 1
            self.text_tokenizer.eos_token_id = 2

        self.data_whole_2D = []
        self.data_whole_3D = []
        self.dataset_reflect = {}

        # CTRG
        ctrg_dataset = CTRG_Dataset(
            data_dir='/data/chenzhixuan/data/CTRG/Chest_new_1_volume/',
            data_json='/jhcnas1/chenzhixuan/CTRG/Chest_new_1/annotation2_new.json',
            split='test',
            prompt_json_file='/home/chenzhixuan/Workspace/LLM4CTRG/src/Dataset/my_dataset/report_prompt.json', down_sample_ratio=1)
        self.dataset_reflect['ctrg_dataset'] = ctrg_dataset
        self.data_whole_3D = self.data_whole_3D + \
            [{'ctrg_dataset': i} for i in range(len(ctrg_dataset))]
        print('ctrg_dataset loaded')

        self.data_whole = self.data_whole_2D + self.data_whole_3D

    def __len__(self):
        return len(self.data_whole)

    def __getitem__(self, idx):
        # vision_x, lang_x, attention_mask, labels
        sample = list(self.data_whole[idx].items())[0]
        # print(sample)
        belong_to = sample[0]
        sample = self.dataset_reflect[sample[0]][sample[1]]

        '''
        Dict: {
            "image_dict": [
                            {"image": image, # image is a tensor of shape [c,w,h,d], c is channel=3, w is width, h is height, d is depth(1 for chestxray,pmcoa,pmcvqa)
                            "position": {"question": 0}}, position is a dict, random choice of 0 or len(question)
                        ]
            "question": question, 
            "answer":answer,  
            }
        '''
        image_id = sample["image_id"]
        images = sample["image_dict"]
        if len(images) > 8:
            images = random.sample(images, 8)
        question = " "
        guide = str(sample["guide"])
        answer = str(sample["answer"])
        cls_labels = sample["cls_labels"]
        images, question, answer = self.text_add_image(
            images, question, answer)
        # print(question,answer)
        # make vision_x
        try:
            vision_x = stack_images(images)
            # vision_x = torch.stack(images)
        except:
            print(self.data_whole[idx].items())
            input()
        # print(vision_x.shape,question,answer)

        return {'id': image_id, 'vision_x': vision_x, 'question': question, 'guide': guide, 'cls_labels': cls_labels, 'answer': answer, 'belong_to': belong_to}

    def text_add_image(self, images, question, answer):
        ref_image = []
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]
        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]
            if position[0] == 'question':
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == 'answer':
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)
        new_question = ''
        new_answer = ''
        for char_i in range(len(question)):
            if question_list[char_i] == []:
                new_question = new_question + question[char_i]
            if question_list[char_i] != []:
                for img_index in question_list[char_i]:
                    try:
                        new_question = new_question + '<image>' + \
                            self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_question = new_question + question[char_i]

        for char_i in range(len(answer)):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            if answer_list[char_i] != []:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + '<image>' + \
                            self.image_padding_tokens[img_index] + '</image>'
                    except:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]

        new_answer = new_answer.replace('•', '')
        return ref_image, new_question, new_answer
