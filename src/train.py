import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
from Model.DiaLLaMA.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
from Dataset.dataset_train import dataset_train
from Dataset.dataset_val import dataset_val
import numpy as np
import torch
import random


def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs, axis=-1)}


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf")
    tokenizer_path: str = field(default="/data/chenzhixuan/checkpoints/Llama-2-7b-chat-hf",
                                metadata={"help": "Path to the tokenizer data."})


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(
        default="/data/chenzhixuan/checkpoints/LLM4CTRG/outputs/")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


@dataclass
class DataCollator(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances) 'loss_reweight': reweight_tensor, 'key_words_query': emphasize_words
        vision_xs, lang_xs, cls_labels, attention_masks, labels, loss_reweight, key_words_query = tuple([instance[key] for instance in instances] for key in (
            'vision_x', 'lang_x', 'cls_labels', 'attention_mask', 'labels', 'loss_reweight', 'key_words_query'))

        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        cls_labels = torch.cat([_.unsqueeze(0) for _ in cls_labels], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0)
                                    for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        loss_reweight = torch.cat([_.unsqueeze(0)
                                  for _ in loss_reweight], dim=0)
        # print(lang_xs.shape,attention_masks.shape,labels.shape)

        target_H = 512
        target_W = 512
        target_D = 4
        MAX_D = 0

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

        if len(vision_xs) == 1 and target_D > 4:
            target_H = 256
            target_W = 256

        vision_xs = [torch.nn.functional.interpolate(
            s, size=(target_H, target_W, target_D)) for s in vision_xs]

        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        # print(vision_xs.shape,vision_xs.dtype)
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            cls_labels=cls_labels,
            attention_mask=attention_masks,
            labels=labels,
            loss_reweight=loss_reweight,
            key_words_query=key_words_query
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.data_sampler = My_DistributedBatchSampler

    print("Setup Data")
    Train_dataset = dataset_train(
        text_tokenizer=model_args.tokenizer_path,
    )

    Eval_dataset = dataset_val(
        text_tokenizer=model_args.tokenizer_path,
    )
    print("Setup Model")

    model = MultiLLaMAForCausalLM(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
    )

    # load model weights
    ckpt = torch.load('/jhcnas1/chenzhixuan/checkpoints/Dia-LLaMA/Dia-LLaMA/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    print("Model weights loaded successfully")

    trainer = Trainer(model=model,
                      train_dataset=Train_dataset,
                      eval_dataset=Eval_dataset,
                      args=training_args,
                      data_collator=DataCollator(),
                      compute_metrics=compute_metrics
                      )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
