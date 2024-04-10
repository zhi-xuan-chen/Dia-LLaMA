from pprint import pprint

import pandas as pd

from metrics_clinical import CheXbertMetrics


# def main():
#     res_path = "/home/chenzhixuan/Workspace/R2Gen/results/swin3d-T3D_R2Gen/test_res_chexbert.csv"
#     gts_path = "/home/chenzhixuan/Workspace/R2Gen/results/swin3d-T3D_R2Gen/test_gts_chexbert.csv"
#     res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
#     res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

#     label_set = res_data.columns[1:].tolist()
#     res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
#     res_data[res_data == -1] = 0
#     gts_data[gts_data == -1] = 0

#     metrics = compute_mlc(gts_data, res_data, label_set)
#     pprint(metrics)

def main():
    chexbert_metrics = CheXbertMetrics('/home/chenzhixuan/Workspace/MRG_baseline/checkpoints/stanford/chexbert/chexbert.pth', 16, 'cpu')
    report_path = "/home/chenzhixuan/Workspace/LLM4CTRG/results/radfm_llama7B_mimic.csv"
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
