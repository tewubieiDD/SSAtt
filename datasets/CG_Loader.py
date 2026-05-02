import sys
import os
import h5py

import numpy as np
import torch as th
import random

from torch.utils import data
from scipy.io import loadmat


class DatasetCG(data.Dataset):
    def __init__(self, path, names):
        self._path = path
        self._names = names  # 存储的是相对路径，如"1/2.mat"、"2/5.mat"等

    def __len__(self):
        return len(self._names)

    def __getitem__(self, item):
        with h5py.File(self._path + self._names[item], 'r') as f:
            x = f['temp_2'][()][None, :, :].real
            # x = loadmat(self._path + self._names[item])['temp_2'][None, :, :].real
            x = th.from_numpy(x).double()
            # 提取标签：第三级目录名称（如"1/2.mat"中的"1"）
            # 分割路径，获取第三级目录名（即类别标签）
            # 例如"1/2.mat"会分割为["1", "2.mat"]，取第一个元素作为标签
            rel_path = self._names[item]
            label_dir = os.path.dirname(rel_path)  # 获取文件所在的目录（第三级目录）
            y = int(label_dir) - 1  # 转换为整数作为标签
            # y = int(self._names[item].split('.')[0].split('_')[-1])
            y = th.from_numpy(np.array(y)).long()
            # return x.to(device),y.to(device)
        return x, y


class DataLoaderCG:
    def __init__(self, data_path, batch_size):
        path_train, path_test = data_path + '/train/', data_path + '/val/'

        # 递归获取所有.mat文件的相对路径
        def get_mat_files(root_dir):
            mat_files = []
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    if filename.endswith('.mat'):
                        # 计算相对路径（相对于root_dir）
                        # 例如在train目录下，"1/2.mat"会被正确记录
                        rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                        mat_files.append(rel_path)
            return sorted(mat_files)

        names_train = get_mat_files(path_train)
        names_test = get_mat_files(path_test)
        # for filenames in os.walk(path_train):
        #     names_train = sorted(filenames[2])
        # for filenames in os.walk(path_test):
        #     names_test = sorted(filenames[2])
        self._train_generator = data.DataLoader(DatasetCG(path_train, names_train), batch_size=batch_size,
                                                shuffle='True')
        self._val_generator = data.DataLoader(DatasetCG(path_test, names_test), batch_size=batch_size,
                                              shuffle='False')


# if __name__ == "__main__":
#     data_path = "../data/CG"
#     batch_size = 30
#     data_loader = DataLoaderCG(data_path, batch_size)
#
#     print(f"训练集样本数: {len(data_loader._train_generator.dataset)}")
#     print(f"验证集样本数: {len(data_loader._val_generator.dataset)}")
#     for x, y in data_loader._train_generator:
#         print(x)
#         print(y)
#         print(f"输入形状: {x.shape}")
#         print(f"标签形状: {y.shape}")
#         print(f"标签示例（应为1、2、3、4中的值）: {y[:5]}")
#         break
