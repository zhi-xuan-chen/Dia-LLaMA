from pprint import pprint

import pandas as pd

from metrics_clinical import CheXbertMetrics


def main():
    chexbert_metrics = CheXbertMetrics('/jhcnas1/chenzhixuan/checkpoints/Dia-LLaMA/Dia-LLaMA/chexbert.pth', 16, 'cuda:4') #NOTE: change with your chexbert model path
    report_path = "/home/chenzhixuan/Workspace/Dia-LLaMA/results/dia-llama.csv"
    report_data = pd.read_csv(report_path)

    gt_list = report_data['Ground Truth'].tolist()
    res_list = report_data['Pred'].tolist()

    ce_scores, f1_class = chexbert_metrics.compute(gt_list, res_list)
    for k, v in ce_scores.items():
        print(f"{k}: {v}")
    
    # # save the f1 score of each class to csv
    # f1_class = pd.DataFrame(f1_class).transpose()
    # f1_class.to_csv('/home/chenzhixuan/Workspace/LLM4CTRG/experiment/disease_f1_distribution/DPB+DTP.csv', index=False)




if __name__ == '__main__':
    main()
