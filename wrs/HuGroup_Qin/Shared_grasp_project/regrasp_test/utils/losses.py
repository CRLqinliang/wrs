import torch
import torch.nn.functional as F


def improved_energy_loss(model, x_pos, x_neg, margin=2.0, reg_weight=0.1):
    """
    改进的能量损失函数，包含对比损失和正则化项
    
    参数:
        model: 能量模型
        x_pos: 正样本
        x_neg: 负样本
        margin: 能量差距阈值
        reg_weight: 正则化权重
        
    返回:
        loss: 总损失
        e_pos: 正样本能量
        e_neg: 负样本能量
    """
    # 计算正样本和负样本的能量
    e_pos = model(x_pos)
    e_neg = model(x_neg)
    
    # 计算能量差距
    energy_gap = e_neg - e_pos
    
    # 对比损失：确保负样本能量比正样本能量大margin
    contrastive_loss = F.relu(margin - energy_gap).mean()
    
    # 分离不同类别的能量
    num_discrete = model.num_discrete
    x_d_pos = x_pos[:, :num_discrete]
    
    # 计算类别分离损失
    class_loss = 0.0
    if x_pos.size(0) > num_discrete:
        # 对每个类别收集样本
        class_energies = []
        for i in range(num_discrete):
            # 找到属于当前类别的样本
            mask = x_d_pos[:, i] > 0.5
            if mask.sum() > 0:
                class_energies.append(e_pos[mask].mean())
        
        # 计算类别间能量差异
        if len(class_energies) > 1:
            class_energies = torch.stack(class_energies)
            # 计算类别间的能量方差，我们希望这个值越大越好
            class_var = torch.var(class_energies)
            # 使用负方差作为损失，鼓励不同类别有不同的能量值
            class_loss = -0.1 * class_var
    
    # 添加L2正则化
    l2_reg = torch.tensor(0., device=e_pos.device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    
    # 总损失
    loss = contrastive_loss + reg_weight * l2_reg + class_loss
    
    return loss, e_pos.mean(), e_neg.mean()