import json
import logging
import os

from matplotlib import pyplot as plt


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
        # self.visualizer = Visualizer()
        self.metrics = {
            "train_loss": [], "val_loss": [], "test_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": []
        }

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(args, f, indent=4)

    def log_iteration(self, epoch_data):
        for k in self.metrics.keys():
            if k in epoch_data:
                self.metrics[k].append(epoch_data[k])
            else:
                self.metrics[k].append(None)

        with open(os.path.join(self.exp_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

        # self.logger.info(f"Epoch Metrics: {epoch_data}")

    def save_final_results(self):
        self.visualizer.plot_learning_curves(self.metrics, self.exp_dir)
