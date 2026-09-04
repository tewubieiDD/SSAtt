#!/usr/bin/env bash

set -u
export PATH="/usr/bin:/bin:$PATH"

MAX_PROCESSES=14
PYTHON_BIN="/d/anaconda3/envs/py39/python.exe"
TASK_FILE="tasks_bnci15.txt"

dataset="BNCI2015001"
evaluation="inter-session"
models=(
#  "BNCI2015NetOneLow11_re"
#  "BNCI2015NetOneMid22_re"
#  "BNCI2015NetOneHigh34_re"
#  "BNCI2015NetTwoLow9_11_re"
#  "BNCI2015NetTwoMid18_22_re"
#  "BNCI2015NetTwoHigh30_34_re"
#  "BNCI2015NetThreeLow9_11_13_re"
#  "BNCI2015NetThreeMid18_22_26_re"
#  "BNCI2015NetThreeHigh30_34_38_re"
#  "BNCI2015NetLowHigh11_34_re"
#  "BNCI2015NetLowMid11_22_re"
#  "BNCI2015NetLowMidHigh11_22_34_re"
#  "BNCI2015NetMidHigh22_34_re"
# ablation
# inter-session
  "BNCI2015NetOneLow11_re_Base"
  "BNCI2015NetOneLow11_re_BiMap"
  "BNCI2015NetOneLow11_re_Dynamic"
  "BNCI2015NetOneLow11_re_Lie"
# inter-subject
#  "BNCI2015NetLowMidHigh11_22_34_re_Base"
#  "BNCI2015NetLowMidHigh11_22_34_re_BiMap"
#  "BNCI2015NetLowMidHigh11_22_34_re_Dynamic"
#  "BNCI2015NetLowMidHigh11_22_34_re_Lie"
)
output_dir_session="outputs/bnci15_ablation/inter_session"
output_dir_subject="outputs/bnci15_ablation/inter_subject"
epochs=50
batch_size=128
lrs=(0.005)
wds=(0.01)
slice=3

seeds=(1)
subjects=(1 2 3 4 5 6 7 8 9 10 11 12)
sessions=(A B)
subjects_with_session_c=(8 9 10 11)

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
if [[ "$SCRIPT_DIR" == "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="."
fi
cd "$SCRIPT_DIR"
: > "$TASK_FILE"

has_session_c() {
  local subject=$1
  local subject_with_c
  for subject_with_c in "${subjects_with_session_c[@]}"; do
    [[ "$subject" == "$subject_with_c" ]] && return 0
  done
  return 1
}

for seed in "${seeds[@]}"; do
  for lr in "${lrs[@]}"; do
    for wd in "${wds[@]}"; do
      for model in "${models[@]}"; do
        for subject in "${subjects[@]}"; do
          if [[ "$evaluation" == "inter-session" ]]; then
            subject_sessions=("${sessions[@]}")
            if has_session_c "$subject"; then
              subject_sessions+=(C)
            fi
            for session in "${subject_sessions[@]}"; do
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
