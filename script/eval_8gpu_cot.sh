#!/bin/bash

START=$1
END=$2
GPUS=8  # 固定8张卡

# 3. 计算步长 (Shell 默认向下取整)
RANGE=$((END - START))
STEP=$((RANGE / GPUS))

echo "任务总范围: $RANGE, 每张卡基准步长: $STEP"
echo "----------------------------------------"

# 4. 循环 0 到 7
for i in {0..7}; do
    # 计算当前 GPU 的起始点
    # 公式: Start + (id * step)
    CURRENT_START=$((START + i * STEP))
    
    # 计算当前 GPU 的终止点
    if [ $i -eq 7 ]; then
        # ⚠️ 关键逻辑：如果是最后一张卡，直接用总 END
        # 这样可以吃掉“除不尽”剩下的所有余数
        CURRENT_END=$END
    else
        # 否则就是：起点 + 步长
        CURRENT_END=$((CURRENT_START + STEP))
    fi

    echo "启动 GPU $i: 范围 $CURRENT_START -> $CURRENT_END"
    
    # 5. 后台执行任务
    # 请把 'python train.py' 换成你实际的运行命令
    CUDA_VISIBLE_DEVICES=$i python -m eval.eval_gsm8k -o \
        --begin $CURRENT_START \
        --end $CURRENT_END \
        > ./log/log_gpu_${i}_cot.txt 2>&1 &
done

# 等待所有后台任务结束（可选）
wait
echo "所有任务已执行完毕。"