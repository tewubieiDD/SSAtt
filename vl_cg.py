import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from SSAtt.models import build_model
from datasets.CG_Loader import DataLoaderCG
from utils import set_seed_thread, create_exp_dir, Recorder
from utils.training import training

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--wd', type=float, default=0, help='weight decay')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--bs', type=int, default=30, help='batch size')
    ap.add_argument('--model_path', type=str, default='')
    ap.add_argument('--data_path', type=str, default='data/CG/', help='data path')
    ap.add_argument('--output_dir', type=str, default='outputs_test', help='output dir')
    ap.add_argument('--model_name', type=str, default='MSNet')
    ap.add_argument('--dataset', type=str, default='CG', help='dataset')
    ap.add_argument('--device', type=str, default='cpu', help='device')
    ap.add_argument('--description', type=str, default='100,80,50,25,no scheduler', help='description')
    args = vars(ap.parse_args())

    set_seed_thread(args["seed"])
    exp_path = create_exp_dir(args["output_dir"], args["seed"], args["model_name"], args["dataset"], args["lr"],
                              args["wd"])
    recorder = Recorder(exp_path, args)

    writer_path = os.path.join('tensorboard_logs', exp_path)
    writer = SummaryWriter(writer_path)

    DataLoader = DataLoaderCG(args["data_path"], args["bs"])

    net = build_model(model_name=args['model_name'],
                      dataset_name=args['dataset'],
                      epochs=0).double().cpu()

    args.pop('bs')
    args.pop('data_path')
    args.pop('output_dir')
    args.pop('model_name')
    args.pop('dataset')
    args.pop('seed')
    args.pop('model_path')
    args.pop('description')
    acc = training(net, DataLoader, recorder, writer, **args)
    recorder.logger.info(f'{acc * 100:.2f}')
