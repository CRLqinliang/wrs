import os
import sys
import subprocess
import argparse
import time
import gc, pickle
from ast import parse

import torch

def run_obstacle_network_train(data_path, grasp_info_path, save_path, train_split=0.7,data_ratio=.5, val_split=0.15,
                              model_type='conv3d', latent_dim=32, beta=1.0, hidden_dims=None, vae_dropout_rate=0.2, ebm_dropout_rate=0.1,
                              input_size=[25, 25, 30], batch_size=32, num_epochs=100, lr=1e-4, lambda_energy=0.5,
                              early_stop_patience=10, seed=42, wandb_project='voxel-grasp-energy',
                              wandb_name=None, train=True, eval=True, load_model=None):
    """运行障碍物网络训练脚本"""
    
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'VAE-EBM_Obstacle_network_train.py')
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 如果hidden_dims为None，使用默认值
    if hidden_dims is None:
        hidden_dims = [512, 512, 512]
    
    # 构建命令行参数
    cmd = [
        'python', script_path,
        '--data_path', data_path,
        '--grasp_info_path', grasp_info_path,
        '--save_path', save_path,
        '--train_split', str(train_split),
        '--data_ratio', str(data_ratio),
        '--val_split', str(val_split),
        '--model_type', model_type,
        '--latent_dim', str(latent_dim),
        '--beta', str(beta),
        '--vae_dropout_rate', str(vae_dropout_rate),
        '--ebm_dropout_rate', str(ebm_dropout_rate),
        '--input_size', str(input_size[0]), str(input_size[1]), str(input_size[2]),
        '--batch_size', str(batch_size),
        '--num_epochs', str(num_epochs),
        '--lr', str(lr),
        '--lambda_energy', str(lambda_energy),
        '--early_stop_patience', str(early_stop_patience),
        '--seed', str(seed),
        '--wandb_project', wandb_project
    ]
    
    # 添加hidden_dims参数
    for dim in hidden_dims:
        cmd.extend(['--hidden_dims', str(dim)])
    
    # 添加input_size参数
    for dim in input_size:
        cmd.extend(['--input_size', str(dim)])
    
    # 添加可选参数
    if wandb_name:
        cmd.extend(['--wandb_name', wandb_name])

    # 打印命令行命令    
    print("执行命令:", " ".join(cmd))
    
    # 执行命令
    subprocess.run(cmd)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行障碍物网络训练")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\obstacle_grasp_data.npz",
                      help='体素数据的npz文件路径')
    parser.add_argument('--grasp_info_path', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle",
                      help='抓取信息的pickle文件路径')
    parser.add_argument('--save_path', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\vae_ebm_model",
                      help='模型保存路径')
    parser.add_argument('--train_split', type=float, default=0.7,
                      help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15,
                      help='验证集比例')
    parser.add_argument('--data_ratio', type=float, default=0.1, help="training dataset ratio.")
    parser.add_argument('--num_workers', type=int, default=24, help='数据加载的worker数量')
    parser.add_argument('--prefetch_factor', type=int, default=16, help='数据预取因子')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='mlp', choices=['conv3d', 'mlp'],
                      help='VAE模型类型: conv3d或mlp')
    parser.add_argument('--latent_dim', type=int, default=32, # 64
                      help='潜在空间维度')
    parser.add_argument('--beta', type=float, default=0.05,
                      help='beta-VAE的beta参数')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512],
                      help='能量模型的隐藏层维度')
    parser.add_argument('--ebm_dropout_rate', type=float, default=0.1,
                      help='dropout率')
    parser.add_argument('--vae_dropout_rate', type=float, default=0.3,
                      help='dropout率')
    parser.add_argument('--input_size', type=int, nargs='+', default=[25, 25, 30],
                      help='输入体素的尺寸 [高度, 宽度, 深度]')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=150,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-3,
                      help='学习率')
    parser.add_argument('--lambda_energy', type=float, default=1e3,
                      help='能量损失权重')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                      help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')

    # 保存和日志
    parser.add_argument('--wandb_project', type=str, default='Voxel_energy_experiments',
                      help='Wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                      help='Wandb运行名称')
    
    # 运行模式
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--eval', action='store_true', help='评估模型')
    parser.add_argument('--load_model', type=str, default=None, help='加载预训练模型的路径')
    parser.add_argument('--disable_multiprocessing', action='store_true', help='禁用多进程数据加载，解决序列化问题')
    
    # 实验名称定制
    parser.add_argument('--exp_name', type=str, default="Bottle",
                      help='实验名称（用于构建wandb_name和save_path）')
    
    # GPU选择
    parser.add_argument('--gpu', type=int, default=0,
                      help='使用的GPU ID')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 如果提供了实验名称，构建wandb_name和save_path
    if args.exp_name:
        if not args.wandb_name:
            args.wandb_name = f"vae_ebm_{args.model_type}_{args.exp_name}_ld{args.latent_dim}_b{args.beta}_lr{args.lr}"
        
        # 扩展保存路径
        args.save_path = os.path.join(args.save_path, f"{args.model_type}_{args.exp_name}")
    
    print(f"===== VAE-EBM障碍物网络训练 =====")
    print(f"数据路径: {args.data_path}")
    print(f"抓取信息路径: {args.grasp_info_path}")
    print(f"保存路径: {args.save_path}")
    print(f"模型类型: {args.model_type}")
    print(f"潜在维度: {args.latent_dim}, Beta: {args.beta}")
    print(f"隐藏层: {args.hidden_dims}")
    print(f"批次大小: {args.batch_size}, 学习率: {args.lr}")
    print(f"Wandb项目: {args.wandb_project}, 运行名称: {args.wandb_name}")
    print(f"运行模式: {'训练' if args.train else ''} {'评估' if args.eval else ''}")
    print(f"GPU ID: {args.gpu}")

    # 运行训练
    run_obstacle_network_train(
            data_path=args.data_path,
            grasp_info_path=args.grasp_info_path,
            save_path=args.save_path,
            train_split=args.train_split,
            val_split=args.val_split,
            data_ratio = args.data_ratio,
            model_type=args.model_type,
            latent_dim=args.latent_dim,
            beta=args.beta,
            hidden_dims=args.hidden_dims,
            ebm_dropout_rate = args.ebm_dropout_rate,
            vae_dropout_rate = args.vae_dropout_rate,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_energy=args.lambda_energy,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            train=args.train,
            eval=args.eval,
            load_model=args.load_model,
            disable_multiprocessing=args.disable_multiprocessing
    )


if __name__ == "__main__":
    # 确保进程开始时内存干净
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("警告: CUDA不可用，将使用CPU")
        
    # 启动主函数
    main()
