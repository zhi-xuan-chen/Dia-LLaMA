import pandas as pd
import numpy as np
import re

df = pd.read_csv('/home/chenzhixuan/Workspace/LLM4CTRG/results/vit3d-radfm_perceiver-radfm_llama2-7b-chat-hf_ctr_all_prompt.csv')

gts_chexbert = df['GT_labels']
res_chexbert = df['Pred_ctr_labels']

def extract_numbers(string):
    pattern = r'\d+'  # 匹配一个或多个数字
    numbers = re.findall(pattern, string)  # 找到所有匹配的数字并返回列表
    numbers = [int(num) for num in numbers]  # 将数字部分转换为整数
    return numbers

gts_chexbert = [extract_numbers(gt) for gt in gts_chexbert]
res_chexbert = [extract_numbers(res) for res in res_chexbert]

gts_chexbert = np.array(gts_chexbert)
res_chexbert = np.array(res_chexbert)

res_chexbert_cvt = (res_chexbert == 1)
gts_chexbert_cvt = (gts_chexbert == 1)

tp = (res_chexbert_cvt * gts_chexbert_cvt).astype(float)

fp = (res_chexbert_cvt * ~gts_chexbert_cvt).astype(float)
fn = (~res_chexbert_cvt * gts_chexbert_cvt).astype(float)

tp_cls = tp.sum(0)
fp_cls = fp.sum(0)
fn_cls = fn.sum(0)

tp_eg = tp.sum(1)
fp_eg = fp.sum(1)
fn_eg = fn.sum(1)

precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls + 1e-6))
recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls + 1e-6))
f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls) + 1e-6))

scores_cvt = {
    'ce_precision_macro': precision_class.mean(),
    'ce_recall_macro': recall_class.mean(),
    'ce_f1_macro': f1_class.mean(),
    'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum() + 1e-6),
    'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum() + 1e-6),
    'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum()) + 1e-6),
    'ce_precision_example': np.nan_to_num(tp_eg / (tp_eg + fp_eg + 1e-6)).mean(),
    'ce_recall_example': np.nan_to_num(tp_eg / (tp_eg + fn_eg + 1e-6)).mean(),
    'ce_f1_example': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg) + 1e-6)).mean(),
    'ce_num_examples': float(len(res_chexbert)),
} 

for key, value in scores_cvt.items():
    print(f"{key}: {value}")