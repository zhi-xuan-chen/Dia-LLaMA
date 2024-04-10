import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from My_Trainer.trainer import Trainer
from dataclasses import dataclass, field
# from Model.RadFM_ctr_prompt.multimodality_model import MultiLLaMAForCausalLM
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from datasampler import My_DistributedBatchSampler
from Dataset.iu_dataset_train import dataset_train
from Dataset.multi_dataset_val import multi_dataset_val
import numpy as np
import torch
import random


def compute_metrics(eval_preds):
    # metric = load_metric("glue", "mrpc")
    ACCs = eval_preds.predictions
    # print(ACCs)
    return {"accuracy": np.mean(ACCs,axis=-1)}

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

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                 
def main():
    set_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.data_sampler = My_DistributedBatchSampler
    
    print("Setup Data")
    Train_dataset = dataset_train(
        text_tokenizer=model_args.tokenizer_path,
        )
    
    Eval_dataset = multi_dataset_val(
        text_tokenizer=model_args.tokenizer_path,
        )
    print("Setup Model")

    model = MultiLLaMAForCausalLM(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
    )

    # loader = torch.utils.data.DataLoader(
    #     Train_dataset, batch_size=1, shuffle=False, num_workers=0)
    # for batch in tqdm.tqdm(loader):
    #     # 将张量保存为 NIfTI 文件
    #     model(**batch)
    #     pass
    
    trainer = Trainer(model=model, 
                      train_dataset = Train_dataset, 
                      eval_dataset = Eval_dataset,
                      args = training_args,
                      compute_metrics= compute_metrics
                      )

    trainer.train()
    trainer.save_state()
      
if __name__ == "__main__":
    main()