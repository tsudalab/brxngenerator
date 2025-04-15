#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # 每个任务分配8个CPU核心 [2,3](@ref)
#SBATCH --gres=gpu:1           # 每个任务申请1个GPU卡 [2](@ref)
#SBATCH --array=0-7            # 创建6个任务（索引0到5）[3,4](@ref)
#SBATCH --log=trainvae_%a.log
#SBATCH --output=trainvae_%a.out     # 输出文件格式：作业ID_任务索引.out [3](@ref)
#SBATCH --error=trainvae_%a.err      # 错误文件格式同上 [3](@ref)

# Log the start time
echo "Job started at: $(date)"

# 传递参数-n ${SLURM_ARRAY_TASK_ID}到Python脚本
/home/gzou/miniforge3/envs/newbrx/bin/python ./trainvae.py -n $SLURM_ARRAY_TASK_ID

# Log the finish time
echo "Job finished at: $(date)"

# time with minutes
echo "Time taken: $((SECONDS/60)) minutes"