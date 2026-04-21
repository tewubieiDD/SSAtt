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

    def get_summary(self, target_model, target_dataset):
        """
        获取特定模型和数据集下，所有受试者的最后一轮 test_acc
        返回格式: {subject_id: final_test_acc}
        """
        results = {}

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            config = self._load_json(os.path.join(folder_path, 'config.json'))
            metrics = self._load_json(os.path.join(folder_path, 'metrics.json'))

            if not config or not metrics:
                continue

            model_name = config.get('model_name')
            dataset_name = config.get('dataset')
            subject_id = config.get('sub')

            if model_name == target_model and dataset_name == target_dataset:
                test_acc_list = metrics.get('test_acc', [])

                valid_accs = [acc for acc in test_acc_list if acc is not None and acc > 0]
                if valid_accs:
                    final_acc = valid_accs[-1]
                    # 如果同一个受试者跑了多次，这里可以根据需要保留最新的一次或报错
                    results[subject_id] = final_acc

        return results

    def calculate_mean_accuracy(self, target_model, target_dataset):
        subject_results = self.get_summary(target_model, target_dataset)

        if not subject_results:
            print(f"未找到模型 [{target_model}] 在数据集 [{target_dataset}] 上的实验结果。")
            return 0.0

        accuracies = list(subject_results.values())
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        print(f"统计报告: {target_model} @ {target_dataset}")
        print(f"受试者数量: {len(accuracies)}")
        print(f"详细结果: {subject_results}")
        print(f"平均精度 (Mean Acc): {mean_acc:.4f}")
        print(f"标准差 (Std Dev): {std_acc:.4f}")

        return mean_acc


# 使用示例
if __name__ == "__main__":
    analyzer = ResultAnalyzer(root_dir='outputs/')

    # 你可以轻松扩展，通过循环分析所有模型和数据集
    models = ['SSAtt_bci']
    datasets = ['MI']

    for m in models:
        for d in datasets:
            analyzer.calculate_mean_accuracy(m, d)
            print("-" * 30)