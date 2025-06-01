import torch
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def langevin_sampling_batch(x_d_fixed, model, n_samples=20, steps=60, 
                           lr=1e-2, noise_scale=0.01, 
                           annealing=True, annealing_rate=0.95, 
                           min_lr=1e-4, min_noise=1e-4,
                           x_range=(-2, 2), y_range=(-2, 2),
                           visualize=False, return_fig=False):
    """
    使用Langevin动力学进行批量采样，支持退火采样
    
    参数:
        x_d_fixed: 离散变量
        model: 能量模型
        n_samples: 采样点数量
        steps: 采样步数
        lr: 初始学习率
        noise_scale: 初始噪声尺度
        annealing: 是否启用退火
        annealing_rate: 退火率 (0-1之间，越接近1退火越慢)
        min_lr: 最小学习率
        min_noise: 最小噪声尺度
        x_range: x坐标范围
        y_range: y坐标范围
        visualize: 是否可视化采样轨迹
        return_fig: 是否返回图像对象（用于wandb记录）
    
    返回:
        x_c: 采样得到的连续变量
        img: 如果return_fig为True，返回包含采样轨迹的图像
    """
    # 获取设备
    device = next(model.parameters()).device
    
    # 初始化采样点: 从指定范围内均匀采样
    x_c = torch.zeros((n_samples, 2), requires_grad=True, device=device)
    x_c.data[:, 0] = torch.rand(n_samples, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    x_c.data[:, 1] = torch.rand(n_samples, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    
    traj_all = []
    
    # 记录每一步的噪声尺度和学习率，用于可视化
    lr_history = []
    noise_history = []
    
    # 当前学习率和噪声尺度
    current_lr = lr
    current_noise = noise_scale
    
    # 存储初始位置
    if visualize:
        traj_all.append(x_c.detach().cpu().numpy())
        lr_history.append(current_lr)
        noise_history.append(current_noise)
    
    for t in range(steps):
        # 计算能量和梯度
        x_full = torch.cat([x_d_fixed.repeat(n_samples, 1), x_c], dim=1)
        energy = model(x_full).sum()
        grad = torch.autograd.grad(energy, x_c, create_graph=False)[0]

        with torch.no_grad():
            # 生成噪声
            noise = torch.randn_like(x_c) * current_noise
            
            # 更新位置（梯度下降寻找低能量区域）
            x_c.data -= current_lr * grad + noise
            
            # 约束位置在指定范围内
            x_c.data[:, 0] = torch.clamp(x_c.data[:, 0], x_range[0], x_range[1])
            x_c.data[:, 1] = torch.clamp(x_c.data[:, 1], y_range[0], y_range[1])
            
            # 每一步都存储轨迹
            if visualize:
                traj_all.append(x_c.detach().cpu().numpy())
                lr_history.append(current_lr)
                noise_history.append(current_noise)
            
            # 退火: 逐步减小学习率和噪声
            if annealing:
                current_lr = max(current_lr * annealing_rate, min_lr)
                current_noise = max(current_noise * annealing_rate, min_noise)
        
        # 重新设置requires_grad标志（关键步骤）- 移动到no_grad块外
        x_c = x_c.detach().requires_grad_(True)

    # 可视化轨迹和退火过程
    if visualize and len(traj_all) > 0:
        traj_all = np.stack(traj_all)  # shape: [steps+1, n_samples, 2]
        
        # 创建一个2x1的子图布局
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 上面的图: 采样轨迹
        for i in range(n_samples):
            axes[0].plot(traj_all[:, i, 0], traj_all[:, i, 1], 'o-', alpha=0.6, 
                      markersize=2, label=f'Sample {i+1}' if i == 0 else None)
        
        # 特别标记起点和终点
        axes[0].scatter(traj_all[0, :, 0], traj_all[0, :, 1], c='blue', 
                      label='Start points', zorder=5)
        axes[0].scatter(traj_all[-1, :, 0], traj_all[-1, :, 1], c='red', 
                      label='End points', zorder=5)
        
        # 设置坐标轴范围
        axes[0].set_xlim(x_range)
        axes[0].set_ylim(y_range)
        
        axes[0].set_title("Langevin Sampling Trajectories" + (" with Annealing" if annealing else ""))
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].grid(True)
        axes[0].legend()
        
        # 下面的图: 学习率和噪声尺度
        x_steps = np.arange(len(lr_history))
        axes[1].plot(x_steps, lr_history, 'b-', label='Learning rate')
        axes[1].plot(x_steps, noise_history, 'r-', label='Noise scale')
        axes[1].set_title("Annealing Schedule")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Value")
        axes[1].set_yscale('log')  # 使用对数刻度更清晰地显示变化
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        
        if return_fig:
            # 将图像转换为PIL Image对象
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img = Image.open(img_buf)
            plt.close(fig)
            return x_c.detach(), img
        else:
            plt.show()
            plt.close(fig)

    return x_c.detach() 