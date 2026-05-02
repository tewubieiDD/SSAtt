import torch

from SSAtt.models import build_model
from utils import set_seed_thread, create_exp_dir, Recorder, trainNetwork, testNetwork_auc, trainNetwork_auc
from utils.GetBCIcha import getAllDataloader
import os
import argparse


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=1, help='No.xxx repeat for training model')
    ap.add_argument('--sub', type=int, default=2, help='subjectxx you want to train')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--wd', type=float, default=1e-1, help='weight decay')
    ap.add_argument('--iterations', type=int, default=130, help='number of training iterations')
    ap.add_argument('--epochs', type=int, default=3, help='number of epochs that you want to use for split EEG signals')
    ap.add_argument('--bs', type=int, default=64, help='batch size')
    ap.add_argument('--model_path', type=str, default='./checkpoint/BCIcha/', help='the folder path for saving the model')
    ap.add_argument('--data_path', type=str, default='./data/BCIcha/', help='data path')
    ap.add_argument('--output_dir', type=str, default='outputs/', help='output dir')
    ap.add_argument('--model_name', type=str, default='SSAtt_Res_MAtt')
    ap.add_argument('--dataset', type=str, default='ERN', help='dataset name')
    args = vars(ap.parse_args())

    set_seed_thread(args["repeat"])
    exp_path = create_exp_dir(args["output_dir"], args["sub"], args["model_name"], args["dataset"], args["repeat"], args["lr"], args["wd"])
    recorder = Recorder(exp_path, args)

    recorder.logger.info(f'subject{args["sub"]}')
    trainloader, validloader, testloader = getAllDataloader(subject=args['sub'],
                                                            data_path=args['data_path'], 
                                                            bs=args['bs'])

    net = build_model(model_name=args['model_name'],
                      dataset_name=args['dataset'],
                      epochs=args['epochs']).double().cpu()

    args.pop('bs')
    args.pop('data_path')
    args.pop('output_dir')
    args.pop('model_name')
    args.pop('dataset')
    trainNetwork_auc(net,
                trainloader, 
                validloader, 
                testloader,
                recorder,
                **args
                )


    net = torch.load(os.path.join(recorder.exp_dir, 'best_model.pt'), weights_only=False)
    auc = testNetwork_auc(net, testloader)
    recorder.logger.info(f'{auc*100:.2f}')




