import torch
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SPDDataset(Dataset):
    def __init__(self, data_list, num_classes=13, is_train=True):
        """
        data_list: 直接传入一个列表或 NumPy 数组，其中每个元素都是 (400, D) 的数据矩阵。
        num_classes: 共有多少个类别，默认13。
        is_train: 是否为训练集，决定标签的分配规则。
        """
        self.data = [np.array(sample, dtype=np.float64) for sample in data_list]  # 统一转换为 NumPy 数组
        self.labels = self._generate_labels(len(data_list), num_classes, is_train)

    def _generate_labels(self, num_samples, num_classes, is_train):
        """
        根据数据数量生成对应的类别标签。
        """
        samples_per_class = num_samples // num_classes  # 每个类别的样本数
        labels = np.concatenate([
            np.full(samples_per_class, cls, dtype=np.int64) for cls in range(num_classes)
        ])

        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        EPSILON = 1e-5
        # 获取原始矩阵 (400, D)
        X = self.data[idx]  # 形状: (400, D)

        # 计算 SPD 矩阵 X @ X.T + EPSILON * I
        X_spd = X @ X.T
        X_spd += np.eye(X_spd.shape[0]) * EPSILON  # 添加对角偏置项

        # 转换为 PyTorch Tensor
        X_spd = torch.tensor(X_spd, dtype=torch.float64)
        label = self.labels[idx]  # 获取对应的标签

        # 扩展 label 的维度，变成 (batch_size, 1)
        # label = label.unsqueeze(-1)  # 变为 (batch_size, 1)

        # **返回字典**
        return X_spd, label


class DataLoaderMDSD:
    def __init__(self, data_path, batch_size=20):
        mat_data = scipy.io.loadmat(data_path)
        mdsd_train = mat_data["mdsd_train"][0]  # 变成 (91,)
        mdsd_test = mat_data["mdsd_test"][0]  # 变成 (39,)

        train_dataset = SPDDataset(mdsd_train, is_train=True)
        test_dataset = SPDDataset(mdsd_test, is_train=False)

        self._train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self._val_generator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    data_path = "../data/MDSD_py.mat"
    batch_size = 20
    data_loader = DataLoaderMDSD(data_path, batch_size)

    print(f"训练集样本数: {len(data_loader._train_generator.dataset)}")
    print(f"验证集样本数: {len(data_loader._val_generator.dataset)}")
    for x, y in data_loader._train_generator:
        print(x)
        print(y)
        print(f"输入形状: {x.shape}")
        print(f"标签形状: {y.shape}")
        print(f"标签示例（应为1、2、3、4中的值）: {y[:5]}")
        break
