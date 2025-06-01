import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


def plot_energy_contour(model, x_d_class, xlim=(-2, 2), ylim=(-2, 2), res=100, return_fig=False):
    """
    可视化特定离散类别下的能量场
    
    参数:
        model: 能量模型
        x_d_class: 离散类别索引
        xlim: x轴范围
        ylim: y轴范围
        res: 网格分辨率
        return_fig: 是否返回图像对象
        
    返回:
        img: 如果return_fig为True，返回包含能量场的图像
    """
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    x = np.linspace(*xlim, res)
    y = np.linspace(*ylim, res)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    num_discrete = model.num_discrete if hasattr(model, 'num_discrete') else 4
    x_d = F.one_hot(torch.tensor([x_d_class]), num_classes=num_discrete).float().to(device).repeat(coords.shape[0], 1)
    x_c = torch.tensor(coords, dtype=torch.float32).to(device)
    x_full = torch.cat([x_d, x_c], dim=1)
    
    with torch.no_grad():
        zz = model(x_full).cpu().numpy().reshape(res, res)

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title(f"Energy Landscape (Discrete class {x_d_class})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    
    if return_fig:
        # 如果需要返回图像对象而非显示
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img = Image.open(img_buf)
        plt.close(fig)
        return img
    else:
        plt.show()
        plt.close(fig)
        return None


def visualize_data_distribution(x, x_d_idx, title="Data Distribution", return_fig=False):
    """
    可视化数据分布，带离散类别标记
    
    参数:
        x: 数据点
        x_d_idx: 离散变量索引
        title: 图表标题
        return_fig: 是否返回图像对象
        
    返回:
        img: 如果return_fig为True，返回包含数据分布的图像
    """
    # 确保数据在CPU上
    x = x.cpu()
    x_d_idx = x_d_idx.cpu()
    
    # 分离连续变量
    x_c = x[:, -2:]  # 假设最后两列是连续变量
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 为每个类别绘制散点图
    for i in range(x_d_idx.max() + 1):
        mask = x_d_idx == i
        ax.scatter(x_c[mask, 0], x_c[mask, 1], label=f'Class {i}', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    
    if return_fig:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img = Image.open(img_buf)
        plt.close(fig)
        return img
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_training_curve(losses, return_fig=False):
    """
    绘制训练损失曲线
    
    参数:
        losses: 损失值列表
        return_fig: 是否返回图像对象
        
    返回:
        img: 如果return_fig为True，返回包含损失曲线的图像
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if return_fig:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img = Image.open(img_buf)
        plt.close(fig)
        return img
    else:
        plt.show()
        plt.close(fig)
        return None


def plot_samples_comparison(x_pos, x_neg, num_discrete, return_fig=False):
    """
    绘制每个类别的正负样本对比图
    
    参数:
        x_pos: 正样本
        x_neg: 负样本
        num_discrete: 离散变量维度
        return_fig: 是否返回图像对象列表
        
    返回:
        imgs: 如果return_fig为True，返回包含所有类别对比图的图像列表
    """
    # 将数据移到CPU并转为numpy
    x_pos_cpu = x_pos.cpu()
    x_neg_cpu = x_neg.cpu()
    
    # 提取离散变量
    x_d_pos = x_pos_cpu[:, :num_discrete]
    x_d_neg = x_neg_cpu[:, :num_discrete]
    
    # 提取连续变量
    x_c_pos = x_pos_cpu[:, num_discrete:].numpy()
    x_c_neg = x_neg_cpu[:, num_discrete:].numpy()
    
    imgs = []
    # 为每个类别创建一个图
    for class_idx in range(num_discrete):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 找到属于当前类别的正样本
        pos_mask = x_d_pos[:, class_idx] > 0.5
        if pos_mask.sum() > 0:
            ax.scatter(x_c_pos[pos_mask, 0], x_c_pos[pos_mask, 1], 
                      c='blue', marker='o', label='Positive Samples', alpha=0.7)
        
        # 找到属于当前类别的负样本
        neg_mask = x_d_neg[:, class_idx] > 0.5
        if neg_mask.sum() > 0:
            ax.scatter(x_c_neg[neg_mask, 0], x_c_neg[neg_mask, 1], 
                      c='red', marker='x', label='Negative Samples', alpha=0.7)
        
        ax.set_title(f'Class {class_idx} Samples Comparison')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
        
        if return_fig:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img = Image.open(img_buf)
            plt.close(fig)
            imgs.append(img)
        else:
            plt.show()
            plt.close(fig)
    
    return imgs if return_fig else None


def plot_energy_contour_zoomed(model, x_d_class, x_pos, zoom_factor=.5, res=100, return_fig=False):
    """
    可视化特定离散类别下的能量场，并且放大显示正样本分布区域
    
    参数:
        model: 能量模型
        x_d_class: 离散类别索引
        x_pos: 正样本数据，用于确定放大区域
        zoom_factor: 放大因子，决定放大区域的大小（0-1之间）
        res: 网格分辨率
        return_fig: 是否返回图像对象
        
    返回:
        img: 如果return_fig为True，返回包含能量场的图像
    """
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    # 提取当前类别的正样本
    num_discrete = model.num_discrete
    x_d_pos = x_pos[:, :num_discrete].cpu()
    mask = x_d_pos[:, x_d_class] > 0.5
    
    if mask.sum() > 0:
        # 提取连续变量
        x_c_pos = x_pos[mask, num_discrete:].cpu().numpy()
        
        # 计算该类别正样本的分布范围
        min_x, max_x = x_c_pos[:, 0].min(), x_c_pos[:, 0].max()
        min_y, max_y = x_c_pos[:, 1].min(), x_c_pos[:, 1].max()
        
        # 扩大范围一点，确保所有样本都在视图内
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        # 使用zoom_factor来决定放大区域的大小
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 确保有最小的范围
        range_x = max(range_x, 0.5)
        range_y = max(range_y, 0.5)
        
        # 设置放大视图的范围
        xlim = (center_x - range_x / zoom_factor, center_x + range_x / zoom_factor)
        ylim = (center_y - range_y / zoom_factor, center_y + range_y / zoom_factor)
    else:
        # 如果没有该类别的正样本，使用默认范围
        xlim = (-2, 2)
        ylim = (-2, 2)
    
    # 创建网格
    x = np.linspace(*xlim, res)
    y = np.linspace(*ylim, res)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # 准备离散变量
    x_d = F.one_hot(torch.tensor([x_d_class]), num_classes=num_discrete).float().to(device).repeat(coords.shape[0], 1)
    x_c = torch.tensor(coords, dtype=torch.float32).to(device)
    x_full = torch.cat([x_d, x_c], dim=1)
    
    # 计算能量
    with torch.no_grad():
        zz = model(x_full).cpu().numpy().reshape(res, res)

    # 绘制能量等高线
    fig = plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, zz, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Energy')
    
    # 在放大图上绘制正样本点
    if mask.sum() > 0:
        plt.scatter(x_c_pos[:, 0], x_c_pos[:, 1], c='red', marker='o', label='Positive Samples', s=30, edgecolors='white')
    
    plt.title(f"Zoomed Energy Landscape (Class {x_d_class})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if return_fig:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img = Image.open(img_buf)
        plt.close(fig)
        return img
    else:
        plt.show()
        plt.close(fig)
        return None 