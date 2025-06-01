import torch
import torch.nn.functional as F
import numpy as np


def sample_clustered_data(batch_size=1000, num_discrete=4, dist_type='checkerboard'):
    """
    生成聚类数据，正样本分布在1和3象限
    
    参数:
        batch_size: 样本数量
        num_discrete: 离散变量维度
        dist_type: 数据分布类型
    
    返回:
        x: 样本数据
        x_d_idx: 离散变量索引
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成class=1的one-hot编码
    x_d = torch.zeros(batch_size, num_discrete, device=device)
    x_d[:, 1] = 1.0  # 设置class=1的one-hot编码
    
    if dist_type == 'checkerboard':
        # 生成棋盘格分布
        x_c = torch.rand(batch_size, 2, device=device) * 4 - 2  # 在[-2, 2]范围内均匀采样
        
        # 根据class=1的规则筛选点
        # 对于class=1，分布在1和3象限
        mask = ((x_c[:, 0] > 0) & (x_c[:, 1] > 0)) | ((x_c[:, 0] < 0) & (x_c[:, 1] < 0))
        x_c = x_c[mask]
        x_d = x_d[mask]
        
        # 如果筛选后的样本数量不足，继续生成直到满足要求
        while len(x_c) < batch_size:
            new_x_c = torch.rand(batch_size - len(x_c), 2, device=device) * 4 - 2
            new_mask = ((new_x_c[:, 0] > 0) & (new_x_c[:, 1] > 0)) | ((new_x_c[:, 0] < 0) & (new_x_c[:, 1] < 0))
            x_c = torch.cat([x_c, new_x_c[new_mask]])
            new_x_d = torch.zeros(len(new_x_c[new_mask]), num_discrete, device=device)
            new_x_d[:, 1] = 1.0
            x_d = torch.cat([x_d, new_x_d])
        
        # 只保留需要的数量
        x_c = x_c[:batch_size]
        x_d = x_d[:batch_size]
        
        # 组合离散和连续部分
        x = torch.cat([x_d, x_c], dim=1)
    
    return x, torch.ones(batch_size, dtype=torch.long, device=device)  # 返回class=1的索引

    
def sample_negatives(model, x_pos, num_negative_samples=None, device=None, 
                     step_size=0.01, noise_scale=0.005, steps=20, 
                     x_range=(-2.5, 2.5), y_range=(-2.5, 2.5)):
    """
    使用Langevin动力学采样从能量模型中生成负样本，并确保不与正样本分布重叠
    
    Args:
        model: 能量模型
        x_pos: 正样本数据，包含离散和连续部分
        num_negative_samples: 要生成的负样本数量，如果为None则使用x_pos的数量
        device: 计算设备，如果为None则使用x_pos的设备
        step_size: Langevin动力学的步长（对应于lr_init）
        noise_scale: 噪声系数（对应于noise_init）
        steps: Langevin动力学的步数
        x_range: x坐标范围，默认为(-2, 2)
        y_range: y坐标范围，默认为(-2, 2)
        
    Returns:
        生成的负样本，只包含连续部分
    """
    # 设置设备和样本数量
    if device is None:
        device = x_pos.device
    
    if num_negative_samples is None:
        batch_size = x_pos.shape[0]
    else:
        batch_size = num_negative_samples
    
    # 设置学习率和噪声参数
    lr_init = step_size
    lr_final = step_size * 0.01  # 结束时步长缩小到初始的1%
    noise_init = noise_scale
    noise_final = noise_scale * 0.001  # 结束时噪声缩小到初始的0.1%
    
    # 从x_pos提取离散部分
    num_discrete = x_pos.shape[1] - 2  # 假设连续部分是2维的
    x_d = x_pos[:, :num_discrete]
    
    # 为每个负样本随机选择离散变量类别
    indices = torch.randint(0, x_pos.shape[0], (batch_size,), device=device)
    x_d_sampled = x_d[indices]
    
    # 获取每个样本对应的离散类别索引
    discrete_class_indices = torch.argmax(x_d_sampled, dim=1)
    
    # 直接从范围内均匀采样初始连续变量
    x_c_init = torch.rand(batch_size, 2, device=device)
    x_c_init[:, 0] = x_c_init[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    x_c_init[:, 1] = x_c_init[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    
    # 检查是否在正样本分布范围内（1和3象限）
    is_in_positive_region = ((x_c_init[:, 0] > 0) & (x_c_init[:, 1] > 0)) | ((x_c_init[:, 0] < 0) & (x_c_init[:, 1] < 0))
    
    # 根据棋盘格规则和离散类别判断哪些点应该被排除
    cell_x = torch.floor(x_c_init[:, 0] + 2)
    cell_y = torch.floor(x_c_init[:, 1] + 2)
    is_on_checkerboard = ((cell_x + cell_y) % 2 == 0)
    
    # 计算这些点应该属于哪个类别
    pos_sum = torch.sum(torch.abs(x_c_init), dim=1)
    expected_class = (pos_sum * num_discrete / 4).long() % num_discrete
    
    # 判断点是否与其选择的类别冲突（在棋盘格上且类别匹配，或在正样本区域内）
    is_conflicting = (is_on_checkerboard & (expected_class == discrete_class_indices)) | is_in_positive_region
    
    # 重新生成有冲突的点
    while is_conflicting.any():
        # 只重新生成那些有冲突的点
        conflict_indices = torch.where(is_conflicting)[0]
        num_conflicts = len(conflict_indices)
        
        # 生成新的连续变量
        new_samples = torch.rand(num_conflicts, 2, device=device)
        new_samples[:, 0] = new_samples[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
        new_samples[:, 1] = new_samples[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
        
        # 检查新样本是否在正样本区域内
        new_is_in_positive_region = ((new_samples[:, 0] > 0) & (new_samples[:, 1] > 0)) | ((new_samples[:, 0] < 0) & (new_samples[:, 1] < 0))
        
        # 检查新样本是否在棋盘格上
        new_cell_x = torch.floor(new_samples[:, 0] + 2)
        new_cell_y = torch.floor(new_samples[:, 1] + 2)
        new_is_on_checkerboard = ((new_cell_x + new_cell_y) % 2 == 0)
        
        # 计算新样本应该属于哪个类别
        new_pos_sum = torch.sum(torch.abs(new_samples), dim=1)
        new_expected_class = (new_pos_sum * num_discrete / 4).long() % num_discrete
        
        # 获取这些样本对应的离散类别
        conflict_classes = discrete_class_indices[conflict_indices]
        
        # 判断新样本是否仍有冲突
        new_is_conflicting = (new_is_on_checkerboard & (new_expected_class.view(-1) == conflict_classes)) | new_is_in_positive_region
        
        # 替换那些不再冲突的样本
        valid_new_samples = ~new_is_conflicting
        if valid_new_samples.any():
            valid_indices = torch.where(valid_new_samples)[0]
            x_c_init[conflict_indices[valid_indices]] = new_samples[valid_indices]
            
            # 更新冲突状态
            cell_x = torch.floor(x_c_init[:, 0] + 2)
            cell_y = torch.floor(x_c_init[:, 1] + 2)
            is_on_checkerboard = ((cell_x + cell_y) % 2 == 0)
            is_in_positive_region = (x_c_init[:, 0] > 0) & (x_c_init[:, 1] > 0)
            
            pos_sum = torch.sum(torch.abs(x_c_init), dim=1)
            expected_class = (pos_sum * num_discrete / 4).long() % num_discrete
            
            is_conflicting = (is_on_checkerboard & (expected_class == discrete_class_indices)) | is_in_positive_region
    
    # 组合离散和连续部分，形成完整数据
    x_init = torch.cat([x_d_sampled, x_c_init], dim=1)
    
    # 现在x_init中的所有样本都不在对应类别的棋盘格上，且包含离散和连续部分
    x = x_init.clone().detach().requires_grad_(True)
    
    # Langevin动力学采样（只更新连续部分）
    for step in range(steps):
        # 计算冷却系数
        t = step / (steps - 1) if steps > 1 else 0
        lr = lr_init * (lr_final / lr_init) ** t
        noise_scale_current = noise_init * (noise_final / noise_init) ** t
        
        # 计算能量和梯度
        energy = model(x)
        grad_energy = torch.autograd.grad(energy.sum(), x)[0]
        
        # 只更新连续部分，保持离散部分不变
        grad_cont = grad_energy[:, num_discrete:]
        
        # 创建新样本，只更新连续部分
        x_new = x.clone()
        x_new[:, num_discrete:] = x[:, num_discrete:] + lr * grad_cont + noise_scale_current * torch.randn_like(grad_cont)
        
        # 确保连续部分在范围内
        x_new[:, num_discrete] = torch.clamp(x_new[:, num_discrete], x_range[0], x_range[1])
        x_new[:, num_discrete+1] = torch.clamp(x_new[:, num_discrete+1], y_range[0], y_range[1])
        
        # 检查新样本是否在正样本区域内（1和3象限）
        cont_part = x_new[:, num_discrete:]
        is_in_positive_region = ((cont_part[:, 0] > 0) & (cont_part[:, 1] > 0)) | ((cont_part[:, 0] < 0) & (cont_part[:, 1] < 0))
        
        # 检查新样本是否在棋盘格上
        cell_x_new = torch.floor(cont_part[:, 0] + 2)
        cell_y_new = torch.floor(cont_part[:, 1] + 2)
        is_new_on_checkerboard = ((cell_x_new + cell_y_new) % 2 == 0)
        
        # 计算新位置应该属于哪个类别
        new_pos_sum = torch.sum(torch.abs(cont_part), dim=1)
        new_expected_class = (new_pos_sum * num_discrete / 4).long() % num_discrete
        
        # 判断是否有冲突（在棋盘格上且类别匹配，或在正样本区域内）
        is_new_conflicting = (is_new_on_checkerboard & (new_expected_class == discrete_class_indices)) | is_in_positive_region
        
        # 只更新那些不冲突的样本
        update_mask = ~is_new_conflicting
        x = x.detach()
        if update_mask.any():
            x[update_mask] = x_new[update_mask].detach()
        x.requires_grad_(True)
    
    return x.detach()

