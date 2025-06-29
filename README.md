# Dia-LLaMA: Towards Large Language Model-driven CT Report Generation

This is the official GitHub repository of the paper ["Dia-LLaMA: Towards Large Language Model-driven CT Report Generation"](https://arxiv.org/pdf/2403.16386)

## Getting Started

1. Before you run the code, you need to create a virtual environment and activate it via the following command:

```bash
# create environment
conda create --name dia-llama python=3.9
# install requirements
pip install -r requirements
conda activate dia-llama
```

2. Once the virtual environment is created, you should download the model weight `pytorch_model.bin` from the huggingface repository [Dia-LLaMA](https://huggingface.co/Trusure/Dia-LLaMA/tree/main). Additionally, you need prepare the model and tokenizer of LLaMA2-7B from [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

3. And then, you should prepare the `CTRG-Chest-548K_volume` dataset. This involves converting the original JPG format images into the NII.GZ format. The original dataset can be found in [CTRG](https://github.com/tangyuhao2016/CTRG). You can also download the converted dataset from our huggingface repository [CTRG-Chest-548K_volume](https://huggingface.co/datasets/Trusure/CTRG-Chest-548K_volume).

## Training

If you want to fine-tune the model, you can run the following script:

```bash
sh scripts/train.sh
```

Please remember to adjust the model path specified in the `src/train.py` file, the dataset path in the `src/Dataset/dataset_train.py` file, and the paths mentioned in the `train.sh` script to match your environment.

## Inference

When you're ready to leverage the model for inference, simply execute the following script:

```bash
sh scripts/test.sh
```

Ensure that you tailor the model and result paths within the `src/test.py` file, as well as the dataset path in the `src/Dataset/dataset_test.py` file, to suit the specifics of your environment.

## Evaluation

Upon completion of your training and inference processes, the outcomes will be stored in the `results` directory. 
1. To assess the NLG performance, you can utilize the script found at `utils/nlg_eval.py`. 
2. For evaluating the Clinical Emotion (CE) metrics, run `utils/chexbert_ce_eval.py`.
    
> Before executing `utils/chexbert_ce_eval.py`, ensure that you have the CheXbert model downloaded from the official source. For your convenience, we have also made the CheXbert model available in our huggingface repository [Dia-LLaMA](https://huggingface.co/Trusure/Dia-LLaMA/tree/main).

## Acknowledgement

This repository is based on the [RadFM](https://github.com/chaoyi-wu/RadFM). We gratefully acknowledge the authors for their contributions to the open-source community.
