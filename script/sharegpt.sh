#!/bin/bash

# 1. 检查参数输入
if [ $# -ne 2 ]; then
    echo "使用方法: $0 <起始编号> <结束编号>"
    echo "示例: $0 0 1000"
    exit 1
fi

START_ID=$1
END_ID=$2

# 2. 设置 GPU 数量和 Python 脚本路径
NUM_GPUS=8
PYTHON_SCRIPT="inference_traindata_gen.py"  # 请修改为你的 Python 文件名

# 3. 计算总任务量
TOTAL_TASKS=$((END_ID - START_ID))

if [ $TOTAL_TASKS -le 0 ]; then
    echo "错误：结束编号必须大于起始编号"
    exit 1
fi

# 4. 计算分配逻辑
# 每个 GPU 的基础任务数
BASE_PER_GPU=$((TOTAL_TASKS / NUM_GPUS))
# 余数（如果有余数，分给前几个 GPU 各多跑 1 个）
REMAINDER=$((TOTAL_TASKS % NUM_GPUS))

echo "=========================================="
echo "总任务数: $TOTAL_TASKS"
echo "使用 GPU 数量: $NUM_GPUS"
echo "=========================================="

CURRENT_START=$START_ID

# 5. 循环启动任务
for (( i=0; i<NUM_GPUS; i++ )); do
    # 计算当前 GPU 分配到的任务数量
    # 如果当前 GPU 索引小于余数，则多分配 1 个任务
    if [ $i -lt $REMAINDER ]; then
        COUNT=$((BASE_PER_GPU + 1))
    else
        COUNT=$BASE_PER_GPU
    fi

    # 计算当前 GPU 的结束编号
    CURRENT_END=$((CURRENT_START + COUNT))

    # 只有当任务数大于0时才执行（防止总任务数小于8的情况）
    if [ $COUNT -gt 0 ]; then
        echo "[GPU $i] 正在启动... 范围: $CURRENT_START -> $CURRENT_END (任务量: $COUNT)"
        
        # 核心执行命令：
        # CUDA_VISIBLE_DEVICES=$i 指定当前进程可见的 GPU
        # & 符号让程序在后台运行，实现并行
        CUDA_VISIBLE_DEVICES=$i python $PYTHON_SCRIPT \
        -d sharegpt_common_en --max_gen 1000 \
        -b $CURRENT_START -e $CURRENT_END \
        > ./log/sharegpt/log_${CURRENT_START}_${CURRENT_END}.txt 2>&1 &
    fi

    # 更新下一个 GPU 的起始编号
    CURRENT_START=$CURRENT_END
done

# 6. 等待所有后台任务完成
echo "所有任务已分发，正在等待执行完成..."
wait
echo "所有 GPU 任务执行完毕！"