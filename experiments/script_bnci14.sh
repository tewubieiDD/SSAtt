#!/usr/bin/env bash

set -u
export PATH="/usr/bin:/bin:$PATH"

MAX_PROCESSES=14
PYTHON_BIN="/d/anaconda3/envs/py39/python.exe"
TASK_FILE="tasks_bnci14.txt"

dataset="BNCI2014001"
evaluation="inter-session"
models=(
#  "BNCI2014NetOneLow11_re"
#  "BNCI2014NetOneMid22_re"
#  "BNCI2014NetOneHigh34_re"
#  "BNCI2014NetTwoLow9_11_re"
#  "BNCI2014NetTwoMid18_22_re"
#  "BNCI2014NetTwoHigh30_34_re"
#  "BNCI2014NetThreeLow9_11_13_re"
#  "BNCI2014NetThreeMid18_22_26_re"
#  "BNCI2014NetThreeHigh30_34_38_re"
#  "BNCI2014NetLowHigh11_34_re"
#  "BNCI2014NetLowMid11_22_re"
#  "BNCI2014NetLowMidHigh11_22_34_re"
#  "BNCI2014NetMidHigh22_34_re"
#  "tsmnet"
# ablation
# inter-subject
#  "BNCI2014NetOneLow11_re_Base"
#  "BNCI2014NetOneLow11_re_BiMap"
#  "BNCI2014NetOneLow11_re_Dynamic"
#  "BNCI2014NetOneLow11_re_Lie"
#  "BNCI2014NetTwoLow9_11_re_Base"
#  "BNCI2014NetTwoLow9_11_re_Dynamic"
#  "BNCI2014NetTwoLow9_11_re_Lie"
#  "BNCI2014NetTwoLow9_11_re_BiMap"
# inter-session
#  "BNCI2014NetLowHigh11_34_re_Base"
#  "BNCI2014NetLowHigh11_34_re_Dynamic"
#  "BNCI2014NetLowHigh11_34_re_Lie"
#  "BNCI2014NetLowHigh11_34_re_BiMap"
  "BNCI2014NetOneMid22_re_Base"
  "BNCI2014NetOneMid22_re_BiMap"
  "BNCI2014NetOneMid22_re_Dynamic"
  "BNCI2014NetOneMid22_re_Lie"
)
output_dir_session="outputs/bnci14_ablation/inter_session"
output_dir_subject="outputs/bnci14_ablation/inter_subject"
epochs=80
batch_size=128
lrs=(0.005)
wds=(0.01)
slice=3

seeds=(1)
subjects=(1 2 3 4 5 6 7 8 9)
sessions=(T E)

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR"
: > "$TASK_FILE"

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for wd in "${wds[@]}"; do
      for model in "${models[@]}"; do
        for subject in "${subjects[@]}"; do
          if [[ "$evaluation" == "inter-session" ]]; then
            for session in "${sessions[@]}"; do
              echo "$PYTHON_BIN -u -m run_experiments --dataset=$dataset --evaluation=$evaluation --model=$model --subject=$subject --test_session=$session --seed=$seed --epochs=$epochs --batch_size=$batch_size --lr=$lr --wd=$wd --slice=$slice --loader_workers=0 --output_dir=$output_dir_session" >> "$TASK_FILE"
            done
          else
            echo "$PYTHON_BIN -u -m run_experiments --dataset=$dataset --evaluation=$evaluation --model=$model --subject=$subject --seed=$seed --epochs=$epochs --batch_size=$batch_size --lr=$lr --wd=$wd --slice=$slice --loader_workers=0 --output_dir=$output_dir_subject" >> "$TASK_FILE"
          fi
        done
      done
    done
  done
done

if [[ ! -s "$TASK_FILE" ]]; then
  echo "No tasks in $TASK_FILE"
  exit 0
fi

echo "Running $(wc -l < "$TASK_FILE") experiments with at most $MAX_PROCESSES processes"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: tasks generated in $TASK_FILE"
  exit 0
fi


#for seed in "${seeds[@]}"; do
#  for lr in "${lrs[@]}"; do
#    for wd in "${wds[@]}"; do
#      echo "Running seed $seed with lr=$lr, wd=$wd"
#      awk -v seed="$seed" -v lr="$lr" -v wd="$wd" \
#        '$0 ~ ("--seed=" seed "([[:space:]]|$)") &&
#         $0 ~ ("--lr=" lr "([[:space:]]|$)") &&
#         $0 ~ ("--wd=" wd "([[:space:]]|$)")' "$TASK_FILE" |
#        xargs -I {} -P "$MAX_PROCESSES" bash -c '{}'
#    done
#  done
#done


xargs -I {} -P "$MAX_PROCESSES" bash -c '{}' < "$TASK_FILE"


#for seed in "${seeds[@]}"; do
#  "$PYTHON_BIN" -u -m summarize_results \
#    --output_dir="$output_dir" \
#    --dataset="$dataset" \
#    --evaluation="$evaluation" \
#    --model="$model" \
#    --seed="$seed" \
#    --lr="$lr" \
#    --wd="$wd"
#done
