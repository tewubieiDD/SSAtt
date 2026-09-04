from __future__ import annotations

import os
import time
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score


def get_optimizer_param_groups(model, weight_decay, no_decay_classes=None):
    manifold_params = []
    standard_params = []

    no_decay_param_ids = set()

    if no_decay_classes is not None:
        for module in model.modules():
            if isinstance(module, no_decay_classes):
                for param in module.parameters():
                    if param.requires_grad:
                        no_decay_param_ids.add(id(param))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_parametrized = "parametrizations" in name

        if id(param) in no_decay_param_ids or is_parametrized:
            manifold_params.append(param)
        else:
            standard_params.append(param)

    # print(f"Standard params (wd={weight_decay}): {len(standard_params)} tensors")
    # print(f"Manifold params (wd=0.0): {len(manifold_params)} tensors")

    return [
        {'params': standard_params, 'weight_decay': weight_decay},
        {'params': manifold_params, 'weight_decay': 0.0}
    ]


def _x_y_d(batch):
    if isinstance(batch, dict):
        return batch["x"], batch.get("y"), batch.get("d")
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2 and isinstance(batch[0], dict):
            return batch[0]["x"], batch[1], batch[0].get("d")
        return batch[0], batch[1] if len(batch) > 1 else None, batch[2] if len(batch) > 2 else None
    return batch, None, None


def _logits(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _bacc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred) if y_true else 0.0


def _evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_loss, total_n = 0.0, 0
    truth, pred = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, d = _x_y_d(batch)
            if y is None:
                raise ValueError("The source validation loader must provide labels.")
            x, y = x.to(device), y.to(device)
            out = _logits(model(x, d)) if d is not None else _logits(model(x))
            loss = loss_fn(out, y)
            n = int(y.numel())
            total_loss += loss.item() * n
            total_n += n
            truth.extend(y.cpu().tolist())
            pred.extend(out.argmax(1).cpu().tolist())
    return total_loss / total_n if total_n else 0.0, _bacc(truth, pred)


def adapt_target_domain(model, target_loader, device="cpu"):
    device = torch.device(device)
    if not callable(getattr(model, "domainadapt_finetune", None)):
        raise AttributeError(
            "Model must implement domainadapt_finetune(x, y, d, target_domains)."
        )

    xs, domains = [], []
    for batch in target_loader:
        x, _y, d = _x_y_d(batch)
        if d is None:
            raise ValueError("Target batches must contain domain ids d.")
        xs.append(x.to(device))
        domains.append(torch.as_tensor(d, device=device).reshape(-1))
    if not xs:
        raise ValueError("The target loader is empty.")

    target_x = torch.cat(xs, dim=0)
    target_d = torch.cat(domains, dim=0)
    domain_ids = torch.unique(target_d)
    with torch.no_grad():
        model.domainadapt_finetune(
            x=target_x,
            y=None,
            d=target_d,
            target_domains=domain_ids,
        )
    return {
        "mode": "REFIT->BUFFER",
        "n_target": int(target_x.shape[0]),
        "domains": [int(domain) for domain in domain_ids.cpu().tolist()],
    }


def predict_target(model, target_loader, device="cpu"):
    model.eval()
    device = torch.device(device)
    predictions, labels = [], []
    with torch.no_grad():
        for batch in target_loader:
            x, y, d = _x_y_d(batch)
            x = x.to(device)
            d = torch.as_tensor(d, device=device).reshape(-1) if d is not None else None
            out = model(x, d) if d is not None else model(x)
            out = _logits(out)
            predictions.extend(out.argmax(1).cpu().tolist())
            if y is not None:
                labels.extend(y.cpu().tolist())
    return predictions, labels


def train_network_loss(net, train_loader, val_loader, test_loader, recorder, device="cpu", **kwargs):
    device = torch.device(device)
    epochs = int(kwargs.get("epochs", 200))
    lr = float(kwargs.get("lr", 1e-3))
    wd = float(kwargs.get("wd", 0.0))
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer_cls = kwargs.get("optimizer_cls", torch.optim.Adam)
    optimizer = optimizer_cls(get_optimizer_param_groups(net, wd), lr=lr)

    best_val_loss = float("inf")
    best_path = os.path.join(recorder.exp_dir, "best_model.pt") if recorder is not None else None
    logger = getattr(recorder, "logger", None)
    writer = getattr(recorder, "writer", None)

    for epoch in range(epochs):
        net.train()
        total_loss, n = 0.0, 0
        truth, pred = [], []
        start = time.time()
        for batch in train_loader:
            x, y, d = _x_y_d(batch)
            if y is None:
                raise ValueError("The source training loader must provide labels.")
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = _logits(net(x, d)) if d is not None else _logits(net(x))
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            bs = int(y.numel())
            total_loss += loss.item() * bs
            n += bs
            truth.extend(y.detach().cpu().tolist())
            pred.extend(out.detach().argmax(1).cpu().tolist())

        train_loss = total_loss / n if n else 0.0
        train_bacc = _bacc(truth, pred)
        val_loss, val_bacc = _evaluate(net, val_loader, device)

        # print
        if logger:
            logger.info(
                f"Iteration{epoch + 1}===== train_loss:{train_loss:.4f} "
                f"val_loss:{val_loss:.4f} train_bacc:{train_bacc:.4f} "
                f"val_bacc:{val_bacc:.4f} train_time:{time.time() - start:.4f}"
            )

        # update
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_path:
                torch.save(net.state_dict(), best_path)

        # record
        if recorder is not None and hasattr(recorder, "log_iteration"):
            recorder.log_iteration({"epoch": epoch + 1, "train_loss": train_loss,
                                    "train_bacc": train_bacc, "val_loss": val_loss,
                                    "val_bacc": val_bacc})
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("BalancedAccuracy/train", train_bacc, epoch + 1)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
            writer.add_scalar("BalancedAccuracy/val", val_bacc, epoch + 1)

    # reload
    if best_path and os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    adaptation = adapt_target_domain(net, test_loader, device)
    # adaptation = None
    predictions, labels = predict_target(net, test_loader, device)
    target_bacc = _bacc(labels, predictions) if labels else None

    if logger:
        mode = adaptation["mode"] if adaptation is not None else "BUFFER"
        logger.info(f"mode: {mode}")

    if writer is not None:
        writer.close()

    return target_bacc
