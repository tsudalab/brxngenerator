import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import savgol_filter  # 曲线平滑处理

# 配置参数
root_dir = "./weights"  # 包含多个实验文件夹的根目录
output_dir = "./validation_plots"  # 图表输出路径
os.makedirs(output_dir, exist_ok=True)

# 数据结构：{folder_name: [epoch0_val_loss, epoch1_val_loss...]}
val_loss_data = defaultdict(list)

# 遍历所有实验文件夹
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # 查找日志文件（假设每个文件夹只有一个含所有epoch的txt文件）
    log_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not log_files:
        print(f"警告：{folder} 中未找到日志文件")
        continue
    
    with open(os.path.join(folder_path, log_files[0]), 'r') as f:
        content = f.read()
        
        # 使用正则表达式提取所有epoch块
        epoch_blocks = re.findall(
            r"Epoch (\d+).*?Validation Loss: (\d+\.\d+).*?Train Losses:.*?pred loss: (\d+\.\d+).*?kl loss: (\d+\.\d+)", 
            content, 
            re.DOTALL
        )
        
        # 按epoch顺序存储验证集loss, kl loss 和 pred loss
        for epoch_num, val_loss, pred_loss, kl_loss in sorted(epoch_blocks, key=lambda x: int(x[0])):
            val_loss_data[folder].append({
                'val_loss': float(val_loss),
                'pred_loss': float(pred_loss),
                'kl_loss': float(kl_loss)
            })

# 创建三个子图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
colors = plt.cm.tab10.colors

# 设置每个子图的样式
for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 绘制各实验曲线
for i, (exp_name, losses) in enumerate(val_loss_data.items()):
    if len(losses) < 5:  # 跳过数据量过少的实验
        continue
    
    # 提取各类loss数据
    epochs = range(len(losses))
    val_losses = [d['val_loss'] for d in losses]
    pred_losses = [d['pred_loss'] for d in losses]
    kl_losses = [d['kl_loss'] for d in losses]
    
    color = colors[i % 10]
    
    # 绘制Validation Loss
    ax1.plot(epochs, val_losses, color=color, linewidth=2, alpha=0.8, label=exp_name)
    ax1.scatter(np.arange(len(losses))[::5], np.array(val_losses)[::5], color=color, s=20, alpha=0.5)
    
    # 绘制Pred Loss
    ax2.plot(epochs, pred_losses, color=color, linewidth=2, alpha=0.8, label=exp_name)
    ax2.scatter(np.arange(len(losses))[::5], np.array(pred_losses)[::5], color=color, s=20, alpha=0.5)
    
    # 绘制KL Loss
    ax3.plot(epochs, kl_losses, color=color, linewidth=2, alpha=0.8, label=exp_name)
    ax3.scatter(np.arange(len(losses))[::5], np.array(kl_losses)[::5], color=color, s=20, alpha=0.5)

# 设置标题和标签
ax1.set_title("Validation Loss", fontsize=14, pad=20)
ax2.set_title("Pred Loss", fontsize=14, pad=20)
ax3.set_title("KL Loss", fontsize=14, pad=20)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", frameon=False)

plt.tight_layout()

# 保存图表
plt.savefig(
    os.path.join(output_dir, "loss_comparison.png"), 
    dpi=300, 
    bbox_inches='tight'
)
plt.close()

print(f"Loss对比图已保存至 {output_dir}")