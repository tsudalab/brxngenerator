#!/bin/bash

# ===============================================
# 脚本配置区 (请确保这部分存在！)
# ===============================================

# 1. 定义要测试的种子范围
START_SEED=1
END_SEED=20

# 2. 定义最大并发进程数 (请根据您的CPU核心数来设置)
#    例如，如果您的电脑是8核，可以设置为8
MAX_JOBS=10

# 3. 定义存放日志的目录名
LOG_DIR="logs"

# ===============================================
# 脚本执行区
# ===============================================

# 创建日志目录 (如果不存在的话)
mkdir -p $LOG_DIR

# 循环启动进程
# for循环会使用上面定义的 START_SEED 和 END_SEED
for seed in $(seq $START_SEED $END_SEED); do
    echo "Starting process for seed ${seed}, logging to ${LOG_DIR}/seed_${seed}.log"
    
    # 将标准输出和错误输出都重定向到单独的日志文件中
    python mainstream.py --seed ${seed} > "${LOG_DIR}/seed_${seed}.log" 2>&1 &

    # 当后台任务数量达到MAX_JOBS时，等待一个任务完成后再继续
    # 这样可以防止一次性启动过多进程导致系统崩溃
    if (($(jobs -p | wc -l) >= MAX_JOBS)); then
        wait -n
    fi
done

# 等待所有剩余的后台任务执行完毕
wait

echo "All jobs finished. Check the '${LOG_DIR}' directory for logs."