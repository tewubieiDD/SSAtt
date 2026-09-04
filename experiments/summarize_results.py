import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETAIL_COLUMNS = ("dataset", "evaluation", "model", "experiment", "lr", "wd", "seed", "target_bacc")
SUMMARY_COLUMNS = ("dataset", "evaluation", "model", "lr", "wd", "seed", "mean±std")
TARGET_DOMAIN_SUMMARY_COLUMNS = (
    "dataset", "evaluation", "model", "target_d", "lr", "wd", "seed", "mean±std",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/bnci14_ablation")
    parser.add_argument("--results_dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dataset", default="BNCI2014001")
    parser.add_argument("--evaluation", choices=("inter-session", "inter-subject"), default="inter-session")
    parser.add_argument("--model", default=None, help="Only summarize this model; omit for all models.")
    parser.add_argument("--seed", type=int, nargs="+", default=None, help="One or more seeds; omit for all.")
    parser.add_argument("--lr", type=float, nargs="+", default=None, help="One or more learning rates; omit for all.")
    parser.add_argument("--wd", type=float, nargs="+", default=None, help="One or more weight decays; omit for all.")
    return parser.parse_args()


def matches(config, args):
    if config.get("dataset") != args.dataset or config.get("evaluation") != args.evaluation:
        return False
    if args.model is not None and config.get("model") != args.model:
        return False
    if args.seed is not None and int(config.get("seed")) not in args.seed:
        return False
    if args.lr is not None and float(config.get("lr")) not in args.lr:
        return False
    if args.wd is not None and float(config.get("wd")) not in args.wd:
        return False
    return True


def collect_experiments(args):
    experiments = []
    for info_path in args.output_dir.rglob("info.json"):
        config_path = info_path.with_name("config.json")
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not matches(config, args):
            continue

        info = json.loads(info_path.read_text(encoding="utf-8"))
        summary = info.get("summary", {})
        target_bacc = summary.get("best_test_acc")
        experiments.append({
            "dataset": config["dataset"],
            "evaluation": config["evaluation"],
            "model": config["model"],
            "experiment": config["experiment_id"],
            "lr": float(config["lr"]),
            "wd": float(config["wd"]),
            "seed": int(config["seed"]),
            "is_finished": bool(summary.get("is_finished", False)),
            "target_bacc": float(target_bacc) if target_bacc is not None else None,
        })
    return experiments


def _detail_group_key(experiment):
    return experiment["dataset"], experiment["evaluation"], experiment["lr"], experiment["wd"], experiment["seed"]


def _summary_group_key(experiment):
    return (
        experiment["dataset"], experiment["evaluation"], experiment["model"],
        experiment["lr"], experiment["wd"], experiment["seed"],
    )


def _format_bacc(value):
    return f"{value:.4f}" if value is not None else "N/A"


def _format_summary(values):
    if not values:
        return "N/A(0)"
    mean = sum(values) / len(values)
    if len(values) == 1:
        return f"{mean:.4f}±0.0000(1)"
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{mean:.4f}±{math.sqrt(variance):.4f}({len(values)})"


def build_detail_tables(experiments):
    grouped = defaultdict(list)
    for experiment in experiments:
        grouped[_detail_group_key(experiment)].append(experiment)

    tables = []
    for key in sorted(grouped):
        rows = []
        by_model = defaultdict(list)
        for experiment in grouped[key]:
            by_model[experiment["model"]].append(experiment)
        for model_index, model in enumerate(sorted(by_model)):
            if model_index:
                rows.append(None)
            for experiment in sorted(by_model[model], key=lambda item: item["experiment"]):
                rows.append([
                    experiment["dataset"], experiment["evaluation"], experiment["model"],
                    experiment["experiment"], experiment["lr"], experiment["wd"], experiment["seed"],
                    _format_bacc(experiment["target_bacc"]) if experiment["is_finished"] else "N/A",
                ])
        tables.append(rows)
    return tables


def build_summary_rows(experiments):
    grouped = defaultdict(list)
    for experiment in experiments:
        grouped[_summary_group_key(experiment)].append(experiment)

    rows = []
    for key in sorted(grouped):
        dataset, evaluation, model, lr, wd, seed = key
        values = [
            experiment for experiment in grouped[key]
            if experiment["is_finished"] and experiment["target_bacc"] is not None
        ]
        if evaluation == "inter-session":
            summary = _format_inter_session_summary(values)
        else:
            summary = _format_summary([experiment["target_bacc"] for experiment in values])
        rows.append([dataset, evaluation, model, lr, wd, seed, summary])
    return rows


def _target_domain(experiment_id):
    prefix = "_target"
    if prefix not in experiment_id:
        raise ValueError(f"Cannot determine target domain from experiment_id: {experiment_id!r}")
    return experiment_id.rsplit(prefix, maxsplit=1)[1]


def _subject_id(experiment_id):
    prefix = "sub"
    separator = "_target"
    if not experiment_id.startswith(prefix) or separator not in experiment_id:
        raise ValueError(f"Cannot determine subject from experiment_id: {experiment_id!r}")
    return experiment_id[len(prefix):experiment_id.index(separator)]


def _format_inter_session_summary(experiments):
    """Match the paper: average subject-level session means and standard deviations."""
    by_subject = defaultdict(list)
    for experiment in experiments:
        by_subject[_subject_id(experiment["experiment"])].append(experiment["target_bacc"])

    subject_means = []
    subject_stds = []
    for values in by_subject.values():
        subject_means.append(sum(values) / len(values))
        if len(values) > 1:
            mean = subject_means[-1]
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            subject_stds.append(math.sqrt(variance))

    if not subject_means:
        return "N/A(0)"
    mean = sum(subject_means) / len(subject_means)
    std = sum(subject_stds) / len(subject_stds) if subject_stds else 0.0
    return f"{mean:.4f}±{std:.4f}({len(subject_means)})"


def build_target_domain_summary_rows(experiments):
    grouped = defaultdict(list)
    for experiment in experiments:
        if experiment["evaluation"] != "inter-session":
            continue
        target_d = _target_domain(experiment["experiment"])
        key = (
            experiment["dataset"], experiment["evaluation"], experiment["model"], target_d,
            experiment["lr"], experiment["wd"], experiment["seed"],
        )
        grouped[key].append(experiment)

    rows = []
    for key in sorted(grouped):
        values = [
            experiment["target_bacc"] for experiment in grouped[key]
            if experiment["is_finished"] and experiment["target_bacc"] is not None
        ]
        dataset, evaluation, model, target_d, lr, wd, seed = key
        rows.append([dataset, evaluation, model, target_d, lr, wd, seed, _format_summary(values)])
    return rows


def _ascii_table(columns, rows):
    text_rows = [[str(value) for value in row] for row in rows if row is not None]
    widths = [len(column) for column in columns]
    for row in text_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header = "|" + "|".join(f" {column:<{width}} " for column, width in zip(columns, widths)) + "|"
    lines = [border, header, border]
    for row in rows:
        if row is None:
            lines.append(border)
        else:
            lines.append("|" + "|".join(f" {str(value):<{width}} " for value, width in zip(row, widths)) + "|")
    lines.append(border)
    return "\n".join(lines)


def write_csv(path, detail_tables, summary_rows, target_domain_summary_rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        for table_index, rows in enumerate(detail_tables, start=1):
            writer.writerow([f"Table {table_index}"])
            writer.writerow(DETAIL_COLUMNS)
            for row in rows:
                writer.writerow([] if row is None else row)
            writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(summary_rows)
        if target_domain_summary_rows:
            writer.writerow([])
            writer.writerow(["Target Domain Summary"])
            writer.writerow(TARGET_DOMAIN_SUMMARY_COLUMNS)
            writer.writerows(target_domain_summary_rows)


def main():
    args = parse_args()
    experiments = collect_experiments(args)
    detail_tables = build_detail_tables(experiments)
    summary_rows = build_summary_rows(experiments)
    target_domain_summary_rows = build_target_domain_summary_rows(experiments)
    successful_count = sum(
        experiment["is_finished"] and experiment["target_bacc"] is not None
        for experiment in experiments
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    result_path = args.results_dir / f"summary_{args.dataset}_{args.evaluation}_{timestamp}.csv"
    write_csv(result_path, detail_tables, summary_rows, target_domain_summary_rows)

    for table_index, rows in enumerate(detail_tables, start=1):
        print(f"Table {table_index}")
        print(_ascii_table(DETAIL_COLUMNS, rows))
    print("Summary")
    print(_ascii_table(SUMMARY_COLUMNS, summary_rows))
    if target_domain_summary_rows:
        print("Target Domain Summary")
        print(_ascii_table(TARGET_DOMAIN_SUMMARY_COLUMNS, target_domain_summary_rows))
    print(f"total_experiments: {len(experiments)}")
    print(f"successful_experiments: {successful_count}")
    print(f"result_file: {result_path}")


if __name__ == "__main__":
    main()
