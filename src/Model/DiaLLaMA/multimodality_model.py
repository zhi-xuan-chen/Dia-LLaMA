from torch import nn
from transformers.models.llama import LlamaForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from .my_embedding_layer import MyEmbedding
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import tqdm.auto as tqdm
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

DISEASES = [
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


class MultiLLaMAForCausalLM(nn.Module):
    def __init__(self, text_tokenizer_path, lang_model_path):
        super(MultiLLaMAForCausalLM, self).__init__()
        # tokenizer
        self.image_padding_tokens = []
        self.text_tokenizer = LlamaTokenizer.from_pretrained(
            text_tokenizer_path,
        )
        special_token = {
            "additional_special_tokens": ["<image>", "</image>"]}
        max_img_size=100
        image_num=32
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

        self.lang_model = LlamaForCausalLM.from_pretrained(
            lang_model_path,
        )
        self.lang_model = self.lang_model.half()

        # use lora to wrap the model
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        self.lang_model = get_peft_model(self.lang_model, peft_config)
        self.lang_model.print_trainable_parameters()

        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()
        # self.lang_model.requires_grad_(False)
        # # frozen the lang model
        # for param in self.lang_model.parameters():
        #     param.requires_grad = False

        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.loss_function = nn.CrossEntropyLoss()

        self.hidden_dim = 4096
        self.voc_size = 32000

    def forward(self, lang_x, vision_x, cls_labels, attention_mask, labels, loss_reweight, key_words_query):
        B = vision_x.shape[0]

        input_embedding, sim = self.embedding_layer(
                vision_x, cls_labels, lang_x,  key_words_query, mode='train')  
        output = self.lang_model(
            inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels)
        logits = output['logits']

        targets = torch.zeros(B*14).to(sim.device)
        ctr_loss = F.cross_entropy(sim, targets.long())

        # only rank 0 print
        if torch.distributed.get_rank() == 0:
            print('lm_oss:', output['loss'].item(), 'ctr_loss:', ctr_loss.item())

        return dict(
            loss=output['loss']+ctr_loss*4,
        )

    def generate(self, question, guide, vision_x):
        with torch.no_grad():
            all_sim = self.embedding_layer(vision_x=vision_x, mode='cls')

            all_sim_sort, indice = torch.sort(all_sim, dim=2, descending=True)
            indice = indice.to(all_sim.device)
            ctr_labels = torch.where(indice[:,:,0]==0, torch.ones(1,14).to(all_sim.device), torch.zeros(1,14).to(all_sim.device))

            ctr_labels = ctr_labels.long().cpu().numpy().tolist()

            prompts = []
            for i in range(len(ctr_labels)):
                prompt = ""
                for j, l in enumerate(ctr_labels[i]):
                    disease = DISEASES[j]
                    if l == 1:
                        state = 'positive'
                        prompt += f"The \"{disease}\" is {state}. "
                    else:
                        state = 'negative'
                        prompt += f"The \"{disease}\" is {state}. "

                prompts.append(prompt)
            
            questions = []
            for i in range(len(question)):
                q = question[i].strip() + prompts[i] + guide
                questions.append(q)
            
            lang_x = self.text_tokenizer(
                questions, max_length=2048, truncation=True, return_tensors="pt"
            )['input_ids'].to('cuda')

            input_embedding = self.embedding_layer(vision_x=vision_x, text_input=lang_x, key_words_query=None)
            
            generation = self.lang_model.generate(
                inputs_embeds=input_embedding, max_new_tokens=300, top_k=50)
            report = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True)
            
        return questions[0], report[0]
