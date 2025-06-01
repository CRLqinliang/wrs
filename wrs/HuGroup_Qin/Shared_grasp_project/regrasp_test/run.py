import torch
import wandb
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regrasp_test.models import HybridEnergyModel
from regrasp_test.train import train
from regrasp_test.sample import sample_from_model
from regrasp_test.config import get_config


def run_demo(dist_type='multi_gaussian', num_discrete=4, epochs=500, use_wandb=True):
    """
    运行演示：训练模型并进行采样
    
    参数:
        dist_type: 数据分布类型
        num_discrete: 离散变量维度
        epochs: 训练轮数
        use_wandb: 是否使用wandb记录实验
    """
    # 获取配置并修改 
    config = get_config(dist_type=dist_type)
    config['model']['num_discrete'] = num_discrete
    config['training']['epochs'] = epochs
    
    print(f"======== 开始训练：分布类型 = {dist_type}，离散维度 = {num_discrete} ========")
    
    # 训练模型
    model, losses = train(config=config, use_wandb=use_wandb)
    
    print("======== 训练完成，开始采样 ========")
    
    # 采样
    samples = sample_from_model(model, config=config, use_wandb=use_wandb)
    
    # 如果使用了wandb，则完成记录
    if use_wandb:
        wandb.finish()
    
    return model, samples


if __name__ == "__main__":
    # 直接运行演示，使用不同的分布类型
    # 可选：'multi_gaussian', 'circle', 'spiral', 'checkerboard'
    model, samples = run_demo(dist_type='checkerboard', epochs=10000)
    
    # 保存模型
    torch.save(model.state_dict(), 'E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/regrasp_test/checkpoint/hybrid_energy_model.pth')
    print("模型已保存到 hybrid_energy_model.pth") 