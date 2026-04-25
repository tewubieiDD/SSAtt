#!/usr/bin/env bash

#!/usr/bin/env bash

# ================= 配置区 =================
# 限制最大并发进程数
MAX_THREADS=14

# 创建任务队列
TASK_FILE="tasks.txt"
> "$TASK_FILE"

echo "正在生成超参数网格搜索任务..."

# ================= 参数定义 =================
# 超参数搜索空间
LRS=(0.01 0.001 0.005)
WDS=(0)

# 各数据集受试者编号
subs_bci=(1 2 3 4 5 6 7 8 9)
subs_mamem=(1 2 3 4 5 6 7 8 9 10 11)
subs_bcicha=(2 6 7 11 12 13 14 16 17 18 20 21 22 23 24 26)

# ================= 任务生成 =================
#for lr in "${LRS[@]}"; do
#    for wd in "${WDS[@]}"; do
for wd in "${WDS[@]}"; do
    for lr in "${LRS[@]}"; do

        # 1. 遍历 BCI 2a 数据集
        for sub in "${subs_bci[@]}"; do
            CMD="python SSAtt_bci.py --sub=$sub --lr=$lr --wd=$wd"
            echo "$CMD" >> "$TASK_FILE"
        done

        # 2. 遍历 MAMEM 数据集
        for sub in "${subs_mamem[@]}"; do
            CMD="python SSAtt_mamem.py --sub=$sub --lr=$lr --wd=$wd"
            echo "$CMD" >> "$TASK_FILE"
        done

        # 3. 遍历 BCI Challenge 数据集
        for sub in "${subs_bcicha[@]}"; do
            CMD="python SSAtt_bcicha.py --sub=$sub --lr=$lr --wd=$wd"
            echo "$CMD" >> "$TASK_FILE"
        done

    done
done

TOTAL_TASKS=$(wc -l < "$TASK_FILE")
echo "任务生成完毕！共计 $TOTAL_TASKS 个实验。"
echo "开始以最大 $MAX_THREADS 并发数执行进程池..."
echo "---------------------------------------------------"

# ================= 并发执行 =================
# 读取 tasks.txt 中的命令，交由 xargs 进行多核并发调度
cat "$TASK_FILE" | xargs -P "$MAX_THREADS" -I {} sh -c "{}"

echo "---------------------------------------------------"
echo "所有实验执行完毕！请检查代码内部生成的日志或输出文件夹。"