import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Dataset.multi_dataset_test import multi_dataset_test
from Model.DiaLLaMA.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
import torch
from torch.utils.data import DataLoader
import csv
import random
import numpy as np
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)
# 预处理数据以及训练模型


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})
    # vision_encoder_path: str = field(default='/home/cs/leijiayu/wuchaoyi/multi_modal/src/PMC-CLIP/checkpoint.pt', metadata={"help": "Path to the vision_encoder."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    test_split: Optional[str] = field(default="open")


@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances)
        vision_xs, lang_xs, attention_masks, labels = tuple(
            [instance[key] for instance in instances] for key in ('vision_x', 'lang_x', 'attention_mask', 'labels'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0)
                                    for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        # print(lang_xs.shape,attention_masks.shape,labels.shape)
        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0

        if len(vision_xs) == 1:
            target_H = 256
            target_W = 256

        D_list = list(range(4, 65, 4))
        if len(vision_xs) == 1:
            if vision_xs[0].shape[0] > 6:
                D_list = list(range(4, 33, 4))

        for ii in vision_xs:
            try:
                D = ii.shape[-1]
                if D > MAX_D:
                    MAX_D = D
            except:
                continue
        for temp_D in D_list:
            if abs(temp_D - MAX_D) < abs(target_D - MAX_D):
                target_D = temp_D

        vision_xs = [torch.nn.functional.interpolate(
            s, size=(target_H, target_W, target_D)) for s in vision_xs]

        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        print(vision_xs.shape)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            attention_mask=attention_masks,
            labels=labels,
        )

def main():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    Test_dataset = multi_dataset_test(
        text_tokenizer=model_args.tokenizer_path
    )

    Test_dataloader = DataLoader(
        Test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    print("Setup Model")

    model = MultiLLaMAForCausalLM(
        text_tokenizer_path=model_args.tokenizer_path,
        lang_model_path=model_args.lang_encoder_path,
    )
    ckpt = torch.load(
        '/data/chenzhixuan/checkpoints/Dia-LLaMA/dia-llama/checkpoint-2000/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model = model.cuda()

    print("load ckpt")

    model.eval()
    with open('/home/chenzhixuan/Workspace/Dia-LLaMA/results/' + 'dia-llama'+'.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(
            ["ID", "Question", "Ground Truth", "Pred", "GT_labels", "Pred_ctr_labels", "Pred_cls_labels", 'belong_to'])

        for sample in tqdm.tqdm(Test_dataloader):
            image_id = int(sample['id'][0])
            question = sample["question"]
            guide = sample['guide'][0]
            cls_labels = sample['cls_labels'][0]
            belong_to = sample['belong_to'][0]

            vision_x = sample["vision_x"].to('cuda')

            answer = sample['answer'][0]
            
            question, pred, pred_cls_labels, pred_ctr_labels = model.generate(question, guide, vision_x)

            print('finding_gt: ', answer)
            print('finding_pred: ', pred)

            writer.writerow([image_id, question, answer,
                            pred, cls_labels, pred_ctr_labels, pred_cls_labels, belong_to])



if __name__ == "__main__":
    main()
