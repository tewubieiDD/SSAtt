import os
import re
from pathlib import Path


def check_experiments(root_dir):
    base_path = Path(root_dir)
    required_files = ['best_model.pt', 'config.json', 'metrics.json', 'train.log']
    log_pattern = re.compile(r"\[.*\]\[INFO\] - \d+\.\d{2}$")

    success_list = []
    failure_list = []

    # 遍历文件夹
    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue

        errors = []

        # 1. 详细检查每一个必要文件
        for file_name in required_files:
            file_path = folder / file_name
            if not file_path.exists():
                errors.append(f"缺失 {file_name}")
            elif file_path.stat().st_size == 0:
                errors.append(f"{file_name} 为空")

        # 2. 只有在 train.log 存在且不为空的情况下，检查最后一行格式
        train_log_path = folder / 'train.log'
        if train_log_path.exists() and train_log_path.stat().st_size > 0:
            try:
                with open(train_log_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if not lines or not log_pattern.search(lines[-1]):
                        errors.append("日志末尾格式不匹配(未完成)")
            except Exception as e:
                errors.append(f"读取日志出错: {str(e)}")

        # 分类存储结果
        result_data = {
            "name": folder.name,
            "errors": errors
        }

        if not errors:
            success_list.append(result_data)
        else:
            failure_list.append(result_data)

    # --- 可视化输出部分 ---

    # 打印成功列表
    print("\n" + "=" * 30 + " 训练成功的实验 " + "=" * 30)
    if not success_list:
        print("暂无成功的实验。")
    else:
        print(f"{'序号':<4} | {'实验文件夹名称'}")
        print("-" * 75)
        for i, exp in enumerate(success_list, 1):
            print(f"{i:<4} | {exp['name']}")

    # 打印失败列表
    print("\n" + "=" * 30 + "  训练失败/未完成的实验 " + "=" * 30)
    if not failure_list:
        print("没有失败的实验，太棒了！")
    else:
        print(f"{'序号':<4} | {'实验文件夹名称':<50} | {'详细错误原因'}")
        print("-" * 110)
        for i, exp in enumerate(failure_list, 1):
            # 将多个错误合并成一个字符串显示
            error_msg = " , ".join(exp['errors'])
            print(f"{i:<4} | {exp['name']:<50} | {error_msg}")

    # 总结
    print("\n" + "=" * 75)
    print(
        f"统计报告: 总计 {len(success_list) + len(failure_list)} | 成功 {len(success_list)} | 失败 {len(failure_list)}")
    print("=" * 75)


if __name__ == "__main__":
    # 指定你的实验结果根目录
    target_dir = "outputs_mdsd"
    if os.path.exists(target_dir):
        check_experiments(target_dir)
    else:
        print(f"找不到目录: {target_dir}")