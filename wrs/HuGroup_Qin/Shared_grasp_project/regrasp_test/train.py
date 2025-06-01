import torch
import wandb
from tqdm import tqdm

from .models import HybridEnergyModel
from .data import sample_clustered_data, sample_negatives
from .utils import improved_energy_loss, plot_energy_contour, visualize_data_distribution, plot_training_curve, plot_samples_comparison
from .utils import plot_energy_contour_zoomed
from .config import get_config


def train(config=None, use_wandb=True):
    """
    训练混合能量模型
    
    参数:
        config: 配置字典，如果为None则使用默认配置
        use_wandb: 是否使用wandb记录实验
    
    返回:
        model: 训练好的模型
        losses: 训练损失列表
    """
    # 获取配置
    if config is None:
        config = get_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化wandb
    if use_wandb:
        wandb.init(
            project=config['wandb']['project'],
            config={
                'num_discrete': config['model']['num_discrete'],
                'dist_type': config['data']['dist_type'],
                'batch_size': config['data']['batch_size'],
                'learning_rate': config['training']['learning_rate'],
                'weight_decay': config['training']['weight_decay'],
                'margin': config['training']['margin'],
                'epochs': config['training']['epochs'],
                'device': str(device),
            }
        )
    
    # 模型参数
    num_discrete = config['model']['num_discrete']
    dim_cont = config['model']['dim_cont']
    
    # 数据参数
    dist_type = config['data']['dist_type']
    batch_size = config['data']['batch_size']
    
    # 训练参数
    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    margin = config['training']['margin']
    reg_weight = config['training']['reg_weight']
    negative_steps = config['training']['negative_steps']
    negative_step_size = config['training']['negative_step_size']
    negative_noise_scale = config['training']['negative_noise_scale']
    
    # 可视化参数
    vis_interval = config['visualization']['vis_interval']
    
    # 初始化模型并移到GPU
    model = HybridEnergyModel(num_discrete=num_discrete, dim_cont=dim_cont).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 初始数据可视化
    if use_wandb:
        x_init, x_d_idx_init = sample_clustered_data(
            batch_size=batch_size, 
            num_discrete=num_discrete, 
            dist_type=dist_type
        )
        # 修改后：
        data_vis = visualize_data_distribution(
            x_init, 
            x_d_idx_init, 
            title=f"Training Data ({dist_type})", 
            return_fig=True
)
        wandb.log({"data_distribution": wandb.Image(data_vis)})
    
    # 训练循环
    losses = []
    for epoch in tqdm(range(epochs), desc="Training"):
        # 获取正样本并移到GPU
        x_pos, x_d_idx = sample_clustered_data(
            batch_size=batch_size, 
            num_discrete=num_discrete, 
            dist_type=dist_type
        )
        x_pos = x_pos.to(device)
        
        # 生成负样本
        x_neg = sample_negatives(
            model, 
            x_pos, 
            steps=negative_steps, 
            step_size=negative_step_size, 
            noise_scale=negative_noise_scale,
            x_range=config['visualization']['xlim'],
            y_range=config['visualization']['ylim']
        )
        
        # 计算损失
        loss, e_pos, e_neg = improved_energy_loss(
            model, 
            x_pos, 
            x_neg, 
            margin=margin, 
            reg_weight=reg_weight
        )
        
        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 梯度裁剪很重要
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 更新学习率调度器
        scheduler.step(loss)
        
        # 记录到wandb
        if use_wandb and (epoch % config['wandb']['log_interval'] == 0):
            wandb.log({
                "loss": loss.item(),
                "e_pos": e_pos.item(),
                "e_neg": e_neg.item(),
                "energy_gap": e_neg.item() - e_pos.item(),
                "epoch": epoch
            })
        
        # 定期可视化
        if (epoch + 1) % vis_interval == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, E_pos: {e_pos.item():.4f}, E_neg: {e_neg.item():.4f}")
            
            # 可视化能量分布并记录到wandb
            if use_wandb:
                # 只可视化class=1的能量场
                x_d_class = 1
                # 全局能量场
                energy_fig = plot_energy_contour(
                    model, 
                    x_d_class,
                    xlim=config['visualization']['xlim'],
                    ylim=config['visualization']['ylim'],
                    res=config['visualization']['res'],
                    return_fig=True
                )
                wandb.log({"energy_landscape_class_1": wandb.Image(energy_fig)})
                
                # 放大的能量场
                zoomed_fig = plot_energy_contour_zoomed(
                    model,
                    x_d_class,
                    x_pos,
                    zoom_factor=1,
                    res=150,
                    return_fig=True
                )
                wandb.log({"zoomed_energy_landscape_class_1": wandb.Image(zoomed_fig)})
                
                # 绘制正负样本对比图
                comparison_figs = plot_samples_comparison(
                    x_pos,
                    x_neg,
                    num_discrete,
                    return_fig=True
                )
                
                # 只记录class=1的正负样本对比图
                wandb.log({"samples_comparison_class_1": wandb.Image(comparison_figs[1])})
        
        # 释放未使用的内存
        torch.cuda.empty_cache()
    
    # 可视化训练损失
    if use_wandb:
        loss_fig = plot_training_curve(losses, return_fig=True)
        wandb.log({"training_loss_curve": wandb.Image(loss_fig)})
    
    # 在训练结束时生成最终的可视化
    x_pos_final, x_d_idx_final = sample_clustered_data(
        batch_size=1000, 
        num_discrete=num_discrete, 
        dist_type=dist_type
    )
    x_pos_final = x_pos_final.to(device)
    
    x_neg_final = sample_negatives(
        model, 
        x_pos_final, 
        steps=negative_steps*2,
        step_size=negative_step_size,
        noise_scale=negative_noise_scale,
        x_range=config['visualization']['xlim'],
        y_range=config['visualization']['ylim']
    )
    
    # 绘制最终的可视化
    if use_wandb:
        # 只记录class=1的最终可视化
        final_zoomed_fig = plot_energy_contour_zoomed(
            model,
            1,  # 只关注class=1
            x_pos_final,
            zoom_factor=0.5,
            res=200,
            return_fig=True
        )
        wandb.log({"final_zoomed_energy_landscape_class_1": wandb.Image(final_zoomed_fig)})
        
        final_comparison_figs = plot_samples_comparison(
            x_pos_final,
            x_neg_final,
            num_discrete,
            return_fig=True
        )
        wandb.log({"final_samples_comparison_class_1": wandb.Image(final_comparison_figs[1])})
    
    return model, losses 