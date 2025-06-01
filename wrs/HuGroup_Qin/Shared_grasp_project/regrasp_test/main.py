import argparse
import os
import torch
import wandb

from regrasp_test.train import train
from regrasp_test.sample import sample_from_model
from regrasp_test.config import get_config


def main():
    """主函数，解析命令行参数并运行训练或采样"""
    parser = argparse.ArgumentParser(description='混合能量模型训练与采样')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'train_and_sample'],
                        help='运行模式: train, sample 或 train_and_sample')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='是否使用wandb记录实验')
    parser.add_argument('--model_path', type=str, default='models/hybrid_energy_model.pth',
                        help='模型保存/加载路径')
    
    # 数据相关参数
    parser.add_argument('--dist_type', type=str, default='multi_gaussian',
                      choices=['multi_gaussian', 'circle', 'spiral', 'checkerboard'],
                      help='数据分布类型')
    parser.add_argument('--num_discrete', type=int, default=4,
                        help='离散变量维度')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=500,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    
    # 采样相关参数
    parser.add_argument('--n_samples', type=int, default=20,
                        help='每个类别的采样点数')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Langevin采样步数')
    
    args = parser.parse_args()
    
    # 根据命令行参数修改配置
    config = get_config(dist_type=args.dist_type)
    config['model']['num_discrete'] = args.num_discrete
    config['data']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['sampling']['n_samples'] = args.n_samples
    config['sampling']['steps'] = args.steps
    
    # 创建模型保存目录
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # 运行训练和/或采样
    if args.mode in ['train', 'train_and_sample']:
        print("======== 开始训练 ========")
        model, losses = train(config=config, use_wandb=args.use_wandb)
        
        # 保存模型
        torch.save(model.state_dict(), args.model_path)
        print(f"模型已保存到 {args.model_path}")
    
    if args.mode in ['sample', 'train_and_sample']:
        print("======== 开始采样 ========")
        
        # 如果只进行采样，则需要加载模型
        if args.mode == 'sample':
            from regrasp_test.models import HybridEnergyModel
            model = HybridEnergyModel(num_discrete=args.num_discrete)
            model.load_state_dict(torch.load(args.model_path))
            print(f"模型已从 {args.model_path} 加载")
        
        # 进行采样
        samples = sample_from_model(model, config=config, use_wandb=args.use_wandb)
    
    # 如果使用了wandb，则完成记录
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 