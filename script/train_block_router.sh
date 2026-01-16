#!/bin/bash
# 比如您想每次并发的数量等于显卡数量
BATCH_SIZE=16 

ulimit -c 0  # 禁止生成 core 文件

echo ">>> 开始执行，总任务 32 个，每批并发 $BATCH_SIZE 个..."

for i in {0..7}
do
    # 1. 计算 GPU ID
    gpu_id=$((i % 8))

    # 2. 启动任务
    echo "启动任务 $i (GPU: $gpu_id)"
    CUDA_VISIBLE_DEVICES="$gpu_id" python train_block_router.py  -d 1024 -b "$i" -bz 4 > "./log/router/log_${i}_router.txt" 2>&1 &

    count=$((i + 1))
    
    # 如果 (当前任务数 对 8 取余 == 0)，说明已经凑满 8 个了
    if [ $((count % BATCH_SIZE)) -eq 0 ]; then
        echo ">>> 已启动 $BATCH_SIZE 个任务，等待它们完成..."
        wait  # 暂停脚本，直到后台这 8 个任务全部跑完，才继续下一轮
        echo ">>> 本批次完成，继续下一批..."
    fi
done

wait
echo "所有任务完成。"