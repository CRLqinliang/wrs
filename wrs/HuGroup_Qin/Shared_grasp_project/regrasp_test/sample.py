import torch
import torch.nn.functional as F
import wandb

from .utils import langevin_sampling_batch, plot_energy_contour, plot_samples_comparison, plot_energy_contour_zoomed
from .data import sample_clustered_data
from .config import get_config


def sample_from_model(model, config=None, use_wandb=True):
    """
    从训练好的模型中采样
    
    参数:
        model: 训练好的能量模型
        config: 配置字典，如果为None则使用默认配置
        use_wandb: 是否使用wandb记录结果
    
    返回:
        samples_dict: 采样结果字典，键为离散类别，值为对应的采样点
    """
    # 获取配置
    if config is None:
        config = get_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 采样参数
    num_discrete = model.num_discrete
    n_samples = config['sampling']['n_samples']
    steps = config['sampling']['steps']
    lr = config['sampling']['lr']
    noise_scale = config['sampling']['noise_scale']
    annealing = config['sampling']['annealing']
    annealing_rate = config['sampling']['annealing_rate']
    min_lr = config['sampling']['min_lr']
    min_noise = config['sampling']['min_noise']
    
    # 可视化参数
    xlim = config['visualization']['xlim']
    ylim = config['visualization']['ylim']
    res = config['visualization']['res']
    
    # 采样结果存储
    samples_dict = {}
    all_samples = []
    
    # 获取一些真实数据用于确定缩放区域
    x_real, x_d_idx_real = sample_clustered_data(
        batch_size=1000,  # 足够大以便捕捉分布
        num_discrete=num_discrete,
        dist_type=config['data']['dist_type']
    )
    x_real = x_real.to(device)
    
    # 对每个离散类别进行采样
    for i in range(num_discrete):
        x_d_class = i
        x_d_fixed = F.one_hot(torch.tensor([x_d_class]), num_classes=num_discrete).float().to(device)
        
        # 可视化能量场（全局）
        if use_wandb:
            energy_fig = plot_energy_contour(
                model, 
                x_d_class,
                xlim=xlim,
                ylim=ylim,
                res=res,
                return_fig=True
            )
            wandb.log({f"final_energy_landscape_class_{i}": wandb.Image(energy_fig)})
            
            # 可视化能量场（放大）
            zoomed_fig = plot_energy_contour_zoomed(
                model,
                x_d_class,
                x_real,
                zoom_factor=0.5,
                res=150,
                return_fig=True
            )
            wandb.log({f"zoomed_energy_landscape_class_{i}": wandb.Image(zoomed_fig)})
        
        # 采样
        if use_wandb:
            sampled_points, sampling_fig = langevin_sampling_batch(
                x_d_fixed, 
                model, 
                n_samples=n_samples, 
                steps=steps,
                lr=lr,
                noise_scale=noise_scale,
                annealing=annealing,
                annealing_rate=annealing_rate,
                min_lr=min_lr,
                min_noise=min_noise,
                x_range=xlim,
                y_range=ylim,
                visualize=True,
                return_fig=True
            )
            
            wandb.log({f"sampling_trajectory_class_{i}": wandb.Image(sampling_fig)})
            
            # 记录采样点到wandb
            points_data = sampled_points.cpu().numpy()
            wandb.log({f"sampled_points_class_{i}": wandb.Table(
                columns=["x", "y"],
                data=[[x, y] for x, y in points_data]
            )})
        else:
            sampled_points = langevin_sampling_batch(
                x_d_fixed, 
                model, 
                n_samples=n_samples, 
                steps=steps,
                lr=lr,
                noise_scale=noise_scale,
                annealing=annealing,
                annealing_rate=annealing_rate,
                min_lr=min_lr,
                min_noise=min_noise,
                x_range=xlim,
                y_range=ylim,
                visualize=True
            )
        
        # 存储采样结果
        samples_dict[i] = sampled_points.cpu()  # 将结果移回CPU
        
        # 构建完整的采样点（包含离散和连续部分）
        x_d_repeated = x_d_fixed.repeat(n_samples, 1)
        full_sample = torch.cat([x_d_repeated.cpu(), sampled_points.cpu()], dim=1)
        all_samples.append(full_sample)
        
        print(f"Sampled points for class {i}:\n", sampled_points.cpu().numpy())
    
    # 将所有采样点连接起来
    all_samples = torch.cat(all_samples, dim=0)
    
    # 生成最终的可视化，对比真实样本和生成样本
    if use_wandb:
        # 比较可视化
        comparison_figs = plot_samples_comparison(
            x_real,
            all_samples.to(device),
            num_discrete,
            return_fig=True
        )
        
        # 记录每个类别的真实/采样对比图
        for i, fig in enumerate(comparison_figs):
            wandb.log({f"real_vs_sampled_class_{i}": wandb.Image(fig)})
        
        # 将采样结果叠加到放大的能量场上
        for i in range(num_discrete):
            # 提取当前类别的采样点
            sampled_points_i = samples_dict[i]
            
            # 创建带有采样结果的能量场可视化
            x_d_one_hot = F.one_hot(torch.tensor([i]), num_classes=num_discrete).float().repeat(len(sampled_points_i), 1)
            sampled_full = torch.cat([x_d_one_hot, sampled_points_i], dim=1).to(device)
            
            # 使用放大视图函数
            zoomed_with_samples = plot_energy_contour_zoomed(
                model,
                i,
                torch.cat([x_real, sampled_full], dim=0),  # 同时包含真实样本和采样结果
                zoom_factor=0.5,
                res=200,
                return_fig=True
            )
            wandb.log({f"zoomed_with_samples_class_{i}": wandb.Image(zoomed_with_samples)})
    
    return samples_dict 