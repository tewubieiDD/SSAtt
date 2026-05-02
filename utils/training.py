import os
import time

import numpy as np
import torch as th
from torch import nn

from SSAtt.optimizer import MixOptimizer


def training(net, DataLoader, recorder, writer, lr=1e-3, wd=None, epochs=200, device='cpu'):
    loss_fn = nn.CrossEntropyLoss().to(device)
    opti = th.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=wd)
    opti = MixOptimizer(opti)
    scheduler = th.optim.lr_scheduler.StepLR(opti.optimizer, step_size=50, gamma=0.8)
    bestAcc = 0
    final_path = os.path.join(recorder.exp_dir, 'best_model.pt')

    for epoch in range(epochs):
        net.train()
        loss_tr, acc_tr, tr_len = 0, 0, 0
        start = time.time()
        for xb, yb in DataLoader._train_generator:
            bs = xb.shape[0]
            tr_len += bs
            xb = xb.to(th.double).to(device)
            yb = yb.to(device)
            opti.zero_grad()
            out = net(xb)
            l = loss_fn(out, yb)
            l.backward()
            opti.step()
            acc_tr += (out.argmax(1) == yb).sum().item()
            loss_tr += l.item() * bs
        scheduler.step()
        end = time.time()

        net.eval()
        loss_val, acc_val, val_len = 0, 0, 0
        # y_true, y_pred = [], []
        for xb, yb in DataLoader._val_generator:
            with th.no_grad():
                bs = xb.shape[0]
                val_len += bs
                xb = xb.to(th.double).to(device)
                yb = yb.to(device)
                out = net(xb)
                l = loss_fn(out, yb)
                predicted_labels = out.argmax(1)
                # y_true.extend(list(yb.cpu().numpy()))
                # y_pred.extend(list(predicted_labels.cpu().numpy()))
                # y_true.extend(yb.cpu().tolist())
                # y_pred.extend(predicted_labels.cpu().tolist())
                acc_val += (predicted_labels == yb).sum().item()
                loss_val += l.item() * bs

        ep_loss_tr = loss_tr / tr_len
        ep_acc_tr = acc_tr / tr_len
        ep_loss_val = loss_val / val_len
        ep_acc_val = acc_val / val_len
        elapse = end - start

        writer.add_scalar('Loss/val', ep_loss_val, epoch + 1)
        writer.add_scalar('Accuracy/val', ep_acc_val, epoch + 1)
        writer.add_scalar('Loss/train', ep_loss_tr, epoch + 1)
        writer.add_scalar('Accuracy/train', ep_acc_tr, epoch + 1)

        recorder.logger.info('')
        recorder.logger.info(f'Iteration{epoch}=====    time:{elapse:.4f}')
        recorder.logger.info(f'train_loss:{ep_loss_tr:.4f}    val_loss:{ep_loss_val:.4f}')
        recorder.logger.info(f'train_acc:{ep_acc_tr:.4f}    val_acc:{ep_acc_val:.4f}')

        if ep_acc_val > bestAcc:
            bestAcc = ep_acc_val
            recorder.logger.info(f'saving to {final_path}')
            th.save(net, final_path)
            # testnet = th.load(final_path, weights_only=False)
            # test_acc = testNetwork(testnet, testloader)
            recorder.logger.info(f'val_acc:{bestAcc}')

        epoch_metrics = {
            "train_loss": ep_loss_tr,
            "val_loss": ep_loss_val,
            "train_acc": ep_acc_tr,
            "val_acc": ep_acc_val,
            "test_acc": bestAcc
        }
        recorder.log_iteration(epoch_metrics)

    writer.close()
    return bestAcc
