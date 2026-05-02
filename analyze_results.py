import json
import os

import numpy as np


class ResultAnalyzer:
    def __init__(self, root_dir='outputs/'):
        self.root_dir = root_dir

    def _load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_summary(self, target_model, target_dataset, target_lr, target_wd):
        results = {}

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            config = self._load_json(os.path.join(folder_path, 'config.json'))
            metrics = self._load_json(os.path.join(folder_path, 'metrics.json'))

            if not config or not metrics:
                continue

            # 提取配置信息
            model_name = config.get('model_name')
            dataset_name = config.get('dataset')
            lr = config.get('lr')
            wd = config.get('wd')
            subject_id = config.get('sub')

            # 多维度条件过滤：模型 + 数据集 + 学习率 + 权重衰减
            if (model_name == target_model and
                    dataset_name == target_dataset and
                    lr == target_lr and
                    wd == target_wd):

                test_acc_list = metrics.get('test_acc', [])

                valid_accs = [acc for acc in test_acc_list if acc is not None and acc > 0]
                if valid_accs:
                    final_acc = valid_accs[-1]
                    # 如果同一个受试者跑了多次，这里可以根据需要保留最新的一次或报错
                    results[subject_id] = final_acc

        return results

    def calculate_mean_accuracy(self, target_model, target_dataset, target_lr, target_wd):
        subject_results = self.get_summary(target_model, target_dataset, target_lr, target_wd)

        if not subject_results:
            print(f"未找到模型 [{target_model}] 在数据集 [{target_dataset}] 上的实验结果。")
            return 0.0

        accuracies = list(subject_results.values())
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"--- 统计报告 ---")
        print(f"模型: {target_model} | 数据集: {target_dataset}")
        print(f"超参: lr={target_lr}, wd={target_wd}")
        print(f"受试者数量: {len(accuracies)}")
        print(f"详细结果 (Sub: Acc): {subject_results}")
        print(f"平均精度 (Mean Acc): {mean_acc:.4f}")
        print(f"标准差 (Std Dev): {std_acc:.4f}")

        return mean_acc


# 使用示例
if __name__ == "__main__":
    analyzer = ResultAnalyzer(root_dir='outputs_four_lr/')

    # 设定你想要分析的固定参数
    models = ['SSAtt_MAtt']
    datasets = ['MI', 'SSVEP', 'ERN']
    # lrs = [0.001, 0.0001, 0.00001]
    lrs = [0.01, 0.005]
    wds = [0]

    for d in datasets:
        for m in models:
            for lr in lrs:
                for wd in wds:
                    analyzer.calculate_mean_accuracy(m, d, lr, wd)
                    print("-" * 30)
