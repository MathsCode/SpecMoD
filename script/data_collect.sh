#!/bin/bash

# ================= 1. 在这里定义你的 List =================
# 注意：Shell 里的数组用圆括号 ()，中间用空格分开
# 如果你的路径里有空格，必须用引号包起来
DATASETS=(
    "mt-bench"
    "gsm8k"
    "alpaca"
    "sum"
    "vicuna-bench"
    "math_infini"
)
# ========================================================

SCRIPT_NAME="inference_traindata_gen.py"
NUM_GPUS=8

# 获取数组的总长度
TOTAL_TASKS=${#DATASETS[@]}

echo "检测到 $TOTAL_TASKS 个数据集任务，准备分发给 $NUM_GPUS 张卡..."

# 循环启动 8 个后台进程 (Worker)
# 这里的 i 就是 GPU 的 ID (0, 1, ... 7)
for i in 0 1 2 3 4 5 6 7; do
    (
        echo "[GPU $i] 准备就绪..."
        
        # ⚠️ 核心逻辑：步长循环
        # 从 i 开始，每次跳 NUM_GPUS (8) 个位置
        # 例如 GPU 0 会处理索引: 0, 8, 16...
        # 例如 GPU 1 会处理索引: 1, 9, 17...
        for (( j=i; j<TOTAL_TASKS; j+=NUM_GPUS )); do
        
            # 取出对应索引的数据集路径
            CURRENT_DATA="${DATASETS[j]}"
            
            echo "  -> [GPU $i] 开始跑: $CURRENT_DATA"
            
            # 执行 Python 命令
            CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME \
                --dataset "$CURRENT_DATA" --max_gen 1000 \
                > ./log/log_gpu_thinking_0.97_${CURRENT_DATA}.txt 2>&1
                
            echo "  -> [GPU $i] 完成: $CURRENT_DATA"
            
        done
        
        echo "[GPU $i] 队列全部完成，下班！"
    ) &
done

# 等待所有后台任务结束
wait
echo "-----------------------------------"
echo "所有数据集处理完毕。"