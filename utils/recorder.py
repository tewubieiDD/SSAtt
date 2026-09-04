import json
import logging
import os
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


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


class Logger:
    def __init__(self, exp_dir, mode='a'):
        self.logger = logging.getLogger(exp_dir)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")

        fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)


class Visualizer:
    @staticmethod
    def plot_learning_curves(metrics, save_path):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(metrics['train_loss'], label='Train')
        plt.plot(metrics['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_acc'], label='Train')
        plt.plot(metrics['val_acc'], label='Val')
        plt.title('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "curves.png"))
        plt.close()


class Recorder:
    def __init__(self, exp_dir, args):
        self.exp_dir = exp_dir
        self.logger = Logger(exp_dir)
        # self.writer = SummaryWriter(exp_dir)
        # self.visualizer = Visualizer()

        self.history = {}

        self.summary = {
            "is_finished": False,
            "best_test_acc": 0.0
        }

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(args, f, indent=4)

    def log_iteration(self, epoch_data):
        # ================= Dynamically record History =================
        for key, value in epoch_data.items():
            if "_" in key:
                prefix, metric = key.split("_", 1)
                if prefix not in self.history:
                    self.history[prefix] = {}
                if metric not in self.history[prefix]:
                    self.history[prefix][metric] = []
                self.history[prefix][metric].append(value)

        # ================= write file =================
        with open(os.path.join(self.exp_dir, "info.json"), "w") as f:
            json.dump({"summary": self.summary, "history": self.history}, f, indent=4)

    def save_summary(self, best_acc):
        self.summary["is_finished"] = True
        self.summary["best_test_acc"] = best_acc

        with open(os.path.join(self.exp_dir, "info.json"), "w") as f:
            json.dump({"summary": self.summary, "history": self.history}, f, indent=4)
