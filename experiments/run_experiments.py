"""Run one explicitly selected BNCI UDA experiment."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from dataloader import make_dataloader, merge_records
from spd.models import build_model
from utils import Recorder, create_exp_dir, set_seed
from utils.train_bacc_uda import train_network_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dataset", choices=("BNCI2014001", "BNCI2015001"), default="BNCI2015001")
    p.add_argument("--evaluation", choices=("inter-session", "inter-subject"), default="inter-subject")
    p.add_argument("--model", default="BNCI2015NetOneLow11_re")
    # p.add_argument("--model", default="matt")
    p.add_argument("--data_path", default=None)
    p.add_argument("--output_dir", default="outputs/bnci15_time/inter_subject")
    p.add_argument("--subject", "--sub", dest="subject", type=int, required=True)
    # p.add_argument("--subject", "--sub", dest="subject", type=int, default=1)
    p.add_argument("--test_session", type=str, default="1", help="Target session, e.g. T/E or 1/2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", "--bs", dest="batch_size", type=int, default=128)
    p.add_argument("--validation_size", type=float, default=.2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--slice", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--loader_workers", type=int, default=0)
    p.add_argument("--add_channel_dim", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--description", type=str, default="")
    return p.parse_args()


def dataset_module(dataset):
    return __import__(f"dataloader.{dataset.lower()}_dataloader", fromlist=["*"])


def build_inter_session_config(args, module, data_path):
    if args.test_session is None or args.test_session.lower() == "all":
        raise ValueError("inter-session requires one explicit --test_session")
    target = module.session_id(args.test_session)
    sessions = module.available_sessions(data_path, args.subject)
    if target not in sessions:
        raise ValueError(f"Session {args.test_session!r} is unavailable for subject {args.subject}")
    source = [session for session in sessions if session != target]
    if not source:
        raise ValueError(f"No source session remains for subject {args.subject}")
    return {
        "experiment_id": f"sub{args.subject:02d}_target{module.session_name(target)}",
        "source_pairs": [(args.subject, session) for session in source],
        "target_pairs": [(args.subject, target)],
        "domain_key": "domain_inter_session",
    }


def build_inter_subject_config(args, module, data_path):
    all_subjects = module.available_subjects(data_path)
    if args.subject not in all_subjects:
        raise ValueError(f"Subject {args.subject} is unavailable in {data_path}")
    source_subjects = [subject for subject in all_subjects if subject != args.subject]
    return {
        "experiment_id": f"target_sub{args.subject:02d}",
        "source_pairs": [(subject, session) for subject in source_subjects for session in
                         module.available_sessions(data_path, subject)],
        "target_pairs": [(args.subject, session) for session in
                         module.available_sessions(data_path, args.subject)],
        "domain_key": "domain_inter_subject",
    }


def build_experiment_config(args):
    module = dataset_module(args.dataset)
    data_path = args.data_path or str(module.DEFAULT_DATA_PATH)
    if args.evaluation == "inter-session":
        experiment = build_inter_session_config(args, module, data_path)
    else:
        experiment = build_inter_subject_config(args, module, data_path)

    config = vars(args).copy()
    config["data_path"] = data_path
    config["split_seed"] = 42
    config.update(experiment)
    return config


def split_source(indices, y, d, validation_size, seed=42):
    indices = np.asarray(indices, dtype=np.int64)
    if validation_size <= 0: return indices, np.empty(0, dtype=np.int64)
    classes = max(1, len(np.unique(y)))
    strat = y[indices] + d[indices] * classes
    try:
        a, b = next(StratifiedShuffleSplit(1, test_size=validation_size, random_state=seed).split(indices, strat))
    except ValueError:
        a, b = train_test_split(np.arange(len(indices)), test_size=validation_size, random_state=seed,
                                stratify=y[indices] if len(np.unique(y[indices])) > 1 else None)
    return indices[a], indices[b]


def load_records(module, data_path, pairs):
    return [module.load_subject_session(data_path, subject, session) for subject, session in pairs]


def run_experiment(config):
    torch.set_num_threads(1)
    set_seed(config["seed"])

    module = dataset_module(config["dataset"])
    source_records = load_records(module, config["data_path"], config["source_pairs"])
    target_records = load_records(module, config["data_path"], config["target_pairs"])
    source = merge_records(source_records)
    target = merge_records(target_records)

    source_indices = np.arange(len(source["y"]))
    train_idx, val_idx = split_source(source_indices, source["y"], source[config["domain_key"]],
                                      config["validation_size"])
    target_indices = np.arange(len(target["y"]))

    train_loader = make_dataloader(source, train_idx, config["domain_key"], config["batch_size"], True,
                                   config["add_channel_dim"], config["loader_workers"])
    val_loader = make_dataloader(source, val_idx, config["domain_key"], config["batch_size"], False,
                                 config["add_channel_dim"], config["loader_workers"])
    test_loader = make_dataloader(target, target_indices, config["domain_key"], config["batch_size"], False,
                                  config["add_channel_dim"], config["loader_workers"])

    config.update({
        "source_total_size": len(source_indices),
        "source_train_size": len(train_idx),
        "source_val_size": len(val_idx),
        "target_size": len(target_indices),
        "train_shuffle": True,
        "val_shuffle": False,
        "test_shuffle": False,
    })

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    exp_dir = create_exp_dir(
        output_dir,
        f"seed{config['seed']}",
        config["dataset"],
        config["evaluation"],
        config["model"],
        config["experiment_id"],
        f"lr{config['lr']}",
        f"wd{config['wd']}",
    )
    config["experiment_dir"] = str(exp_dir)
    recorder = Recorder(exp_dir, config)

    net = build_model(config["model"], config["dataset"], config).cpu()

    score = train_network_loss(net, train_loader, val_loader, test_loader, recorder, device=config["device"],
                               epochs=config["epochs"], lr=config["lr"], wd=config["wd"])
    recorder.save_summary(score)
    recorder.logger.info(f"finished {config['experiment_id']} target_bacc:{score:.6f}")

    return score


def main():
    args = parse_args()
    config = build_experiment_config(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
