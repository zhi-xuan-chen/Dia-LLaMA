import csv
from metrics_nlg import compute_scores

# 定义两个空字典，用于存储提取的信息
gt_dict = {}
pred_dict = {}

# 打开CSV文件
with open('/home/chenzhixuan/Workspace/Dia-LLaMA/results/dia-llama.csv', 'r') as file:
    # 创建CSV读取器
    reader = csv.reader(file)

    # 跳过标题行（如果有的话）
    next(reader)

    # 遍历每一行数据
    for row in reader:
        # 提取ID、GT和Pred
        id_value = row[0]
        gt_value = row[2]
        pred_value = row[3]

        # 将ID作为键，GT作为值存储在gt_dict中
        gt_dict[id_value] = [gt_value]

        # 将ID作为键，Pred作为值存储在pred_dict中
        pred_dict[id_value] = [pred_value]

# 计算评价指标
metrics = compute_scores(gt_dict, pred_dict)
print(metrics)
