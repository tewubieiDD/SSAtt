import json
import os

import numpy as np
import torch

from utils import testNetwork_auc
from utils.GetBCIcha import getAllDataloader


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
            metrics = None

            if not config:
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

                # --- 核心逻辑修改：针对 ERN 数据集计算 AUC ---
                if dataset_name == 'ERN':
                    model_path = os.path.join(folder_path, 'best_model.pt')
                    if os.path.exists(model_path):
                        try:
                            # 加载最佳模型
                            net = torch.load(model_path, weights_only=False, map_location='cpu')
                            net.eval()

                            # 注意：你需要在这里获取对应 subject 的 testloader
                            # 假设你有一个函数可以根据 sub_id 获取 loader
                            testloader = self._get_loader_for_subject(subject_id)

                            if testloader is not None:
                                # 调用你提到的测试函数
                                # 注意：确保 testNetwork_auc 在当前作用域可用
                                auc_val = testNetwork_auc(net, testloader)
                                results[subject_id] = auc_val
                            else:
                                print(f"无法为 Sub {subject_id} 加载数据")
                        except Exception as e:
                            print(f"处理 Sub {subject_id} 的模型时出错: {e}")


                # --- 常规数据集逻辑：读取 metrics.json 中的 Accuracy ---
                else:
                    metrics = self._load_json(os.path.join(folder_path, 'metrics.json'))
                if metrics:
                    test_acc_list = metrics.get('test_acc', [])
                    valid_accs = [acc for acc in test_acc_list if acc is not None and acc > 0]
                    if valid_accs:
                        results[subject_id] = valid_accs[-1]

        return results

    def _get_loader_for_subject(self, sub_id):
        trainloader, validloader, testloader = getAllDataloader(subject=sub_id,
                                                                data_path='./data/BCIcha/',
                                                                bs=64)
        return testloader

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
    # datasets = ['MI', 'SSVEP', 'ERN']
    datasets = ['ERN']
    lrs = [0.1, 0.05, 0.01, 0.005]
    wds = [0]

    for d in datasets:
        for m in models:
            for lr in lrs:
                for wd in wds:
                    analyzer.calculate_mean_accuracy(m, d, lr, wd)
                    print("-" * 30)
