import os
import random
import time

import numpy as np
import torch


def set_seed(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed_thread(seed, threadnum=10):
    torch.set_num_threads(threadnum)
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_exp_dir(base_path, *args):
    timestamp = time.strftime('%y%m%d_%H%M%S', time.localtime())
    arg_strings = [str(arg) for arg in args]
    arg_strings.append(timestamp)
    folder_name = "-".join(arg_strings)
    path = os.path.join(base_path, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
