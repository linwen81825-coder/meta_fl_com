#!/bin/bash

# 检查是否提供了足够的参数  
if [ "$#" -ne 5 ]; then  
    echo "Usage: $0 <script_name> <dataset_name> <gpu_id> <use_dirichlet> <num_selected>"  
    exit 1  
fi  

SCRIPT_NAME="$1"  
DATASET_NAME="$2"  
GPU_ID="$3"
USE_DIRICHLET="$4"
NUM_SELECTED="$5"
LOG_FILE="../meta_log/log_${SCRIPT_NAME}_${DATASET_NAME}_${USE_DIRICHLET}_${NUM_SELECTED}_$(date +'%m%d_%H%M').out"  

cat "$SCRIPT_NAME" >> "$LOG_FILE" 

# 运行Python脚本并将输出追加到日志文件  
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u "$SCRIPT_NAME" --dataset "$DATASET_NAME" --use_dirichlet "$USE_DIRICHLET" --num_selected "$NUM_SELECTED" >> "$LOG_FILE" 2>&1 &  

# 可选：打印日志文件的路径，以便稍后检查  
echo "Log file: $LOG_FILE"
