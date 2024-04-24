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
        # # load partial weights from llava-med
        # lang_ckpt = torch.load(
        #     '/jhcnas1/chenzhixuan/checkpoints/LLM4CTRG/llava_partial_weights.pth', map_location='cpu')
        # self.lang_model.load_state_dict(lang_ckpt, strict=False)
        # print('load partial weights from llava-med')

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
        if labels.shape == lang_x.shape:
            B = vision_x.shape[0]
            self.embedding_layer.flag = 'Text'
            # lang_x = lang_x.to(vision_x.dtype)
            # lang_x = lang_x + torch.zeros(1, dtype=lang_x.dtype, device=lang_x.device, requires_grad=True)
            # vision_x = vision_x + torch.zeros(1, dtype=vision_x.dtype, device=vision_x.device, requires_grad=True)
            # input_embedding = checkpoint(self.embedding_layer, lang_x, vision_x)
            input_embedding, cls_logits, loss_match, sim = self.embedding_layer(
                 vision_x, cls_labels, lang_x,  key_words_query, mode='train')   # ,loss_matching
            output = self.lang_model(
                inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels)
            logits = output['logits']

            targets = torch.zeros(B*14).to(sim.device)
            ctr_loss = F.cross_entropy(sim, targets.long())

            cls_loss = self.loss_function(cls_logits, cls_labels)

            loss_reg = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(
                    shift_logits.device)
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(
                    shift_loss_reweight*loss_reg)/torch.sum(shift_loss_reweight)
            loss = loss_reg
            if loss_match != None:
                loss = 0.8*loss + 0.2*loss_match

            logits = output['logits'][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            Acc = torch.sum(torch.all(torch.logical_or(
                predictions == labels, labels == -100), dim=-1))
            Accuracy = Acc / total

            # only rank 0 print
            if torch.distributed.get_rank() == 0:
                print('lm_oss:', output['loss'].item(), 'cls_loss:', cls_loss.item(), 'ctr_loss:', ctr_loss.item())

            return dict(
                # loss_reg = loss_reg,
                # loss_matching = loss_matching,
                logits=Accuracy,
                loss=output['loss']+ctr_loss*4,
            )

    def generate(self, question, guide, vision_x):
        self.embedding_layer.flag = 'Text'
        with torch.no_grad():
            cls_preds, all_sim = self.embedding_layer(vision_x=vision_x, mode='cls')
            cls_preds = F.softmax(cls_preds, dim=1)
            cls_preds_logits = cls_preds[:, 1, :14]
            cls_labels = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist() # B, 14

            all_sim_sort, indice = torch.sort(all_sim, dim=2, descending=True)
            indice = indice.to(all_sim.device)
            ctr_labels = torch.where(indice[:,:,0]==0, torch.ones(1,14).to(all_sim.device), torch.zeros(1,14).to(all_sim.device))
            # all_sim = all_sim.reshape(-1, 14, 2, 10)
            # all_sim = all_sim.sum(dim=-1)
            # ctr_labels = torch.where(all_sim[:,:,0] > all_sim[:,:,1], torch.ones(1,14).to(all_sim.device), torch.zeros(1,14).to(all_sim.device))

            ctr_labels = ctr_labels.long().cpu().numpy().tolist()

            prompts = []
            for i in range(len(ctr_labels)):
                prompt = ""
                for j, l in enumerate(ctr_labels[i]):
                    disease = CONDITIONS[j]
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
            
        return questions[0], report[0], cls_labels[0], ctr_labels[0]
