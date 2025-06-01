def get_config(dist_type='checkerboard'):
    """
    获取默认配置
    
    返回:
        config: 配置字典
    """
    config = {
        'model': {
            'num_discrete': 4,  # 离散变量维度
            'dim_cont': 2,  # 连续变量维度
            'hidden_dim': 128,  # 减小隐藏层维度
        },
        'data': {
            'dist_type': dist_type,  # 数据分布类型
            'batch_size': 1024,  # 减小批量大小
        },
        'training': {
            'epochs': 5000,  # 减少训练轮数
            'learning_rate': 5e-5,  # 降低学习率
            'weight_decay': 1e-2,  # 增加权重衰减
            'margin': 5.0,  # 减小margin
            'reg_weight': 0.1,  # 增加正则化权重
            'negative_steps': 20,  # 减少负样本采样步数
            'negative_step_size': 0.001,  # 减小负样本步长
            'negative_noise_scale': 0.0005,  # 减小噪声
            'grad_clip': 0.1,  # 梯度裁剪阈值
        },
        'sampling': {
            'n_samples': 100,         # 减少采样点数量
            'steps': 20,             # 采样步数
            'lr': 0.01,               # 采样学习率
            'noise_scale': 0.03,      # 采样噪声
            'annealing': True,        # 启用退火
            'annealing_rate': 0.98,   # 退火率
            'min_lr': 1e-3,           # 最小学习率
            'min_noise': 1e-3         # 最小噪声
        },
        'visualization': {
            'vis_interval': 10,  # 可视化间隔
            'xlim': (-2.5, 2.5),  # x轴范围
            'ylim': (-2.5, 2.5),  # y轴范围
            'res': 100,  # 降低可视化分辨率
        },
        'wandb': {
            'project': 'hybrid-energy-model',  # wandb项目名
            'name': 'class1-3',  # wandb实验名
            'log_interval': 1,  # 日志记录间隔
        }
    }
    return config 