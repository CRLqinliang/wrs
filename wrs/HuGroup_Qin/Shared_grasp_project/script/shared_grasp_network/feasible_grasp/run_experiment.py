import subprocess
import os
import sys
import time
import gc
import torch

def run_grasp_experiment(hidden_dims, num_layers, dropout_rate, batch_size, 
                        learning_rate, dataset_type, data_id, 
                        train_split, data_ratio, data_range, seed, use_quaternion, use_stable_label,
                        grasp_type, state_type):
    """运行单个抓取网络实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, 'grasp_network_train.py')
    
    # 数据路径
    data_path = os.path.join(r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/{dataset_type}_{data_id}.pickle')

    grasp_data_path = os.path.join(r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/bottle_grasp_{data_id}.pickle')
    
    # 构建模型保存路径
    exp_name = f'BC_feasible_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr{learning_rate}_r{data_range}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_seed{seed}'
    model_save_path = os.path.join(r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                   f'model/feasible_best_model/best_model_grasp_{exp_name}.pth')
    
    # 将hidden_dims列表转换为命令行参数字符串
    hidden_dims_str = ' '.join(map(str, hidden_dims))
    cmd = [
        'python', script_path,
        '--data_path', data_path,
        '--grasp_data_path', grasp_data_path,
        '--model_save_path', model_save_path,
        '--hidden_dims', *hidden_dims_str.split(),
        '--num_layers', str(num_layers),
        '--dropout_rate', str(dropout_rate),
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--weight_decay', '0.01',
        '--num_epochs', '80',
        '--early_stop_patience', '10',
        '--train_split', str(train_split),  
        '--val_split', '0.15',
        '--data_ratio', str(data_ratio),
        '--data_range', str(data_range),
        '--grasp_type', grasp_type,
        '--state_type', state_type,
        '--seed', str(seed),
        '--use_quaternion', '1' if use_quaternion else '0',
        '--use_stable_label', '1' if use_stable_label else '0',
        '--num_workers', '4',
        '--pin_memory', 'True',
        '--use_balanced_sampler', 'True',
        '--lr_factor', '0.5',
        '--lr_patience', '30',
        '--min_lr', '1e-5',
        '--wandb_project', 'feasible_grasp_experiments',
        '--wandb_name', exp_name
    ]
    
    print(f"\n开始训练抓取网络...")
    print(f"配置: hidden_dims={hidden_dims}, num_layers={num_layers}, batch_size={batch_size}, lr={learning_rate}")
    print(f"表示方式: use_quaternion={use_quaternion}, use_stable_label={use_stable_label}")
    print(f"数据: dataset_type={dataset_type}, data_id={data_id}, train_split={train_split}, data_ratio={data_ratio}, seed={seed}")
    subprocess.run(cmd)
    print(f"抓取网络训练完成\n")

def main():
    # 确保模型保存目录存在
    os.makedirs(r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model', exist_ok=True)
    
    # 实验配置
    experiment_configs = [
        # 1. 使用四元数 + stable label
        {
            'hidden_dims': [512, 512, 512],
            'num_layers': 3,
            'dropout_rate': 0.1,
            'batch_size': 2048,
            'learning_rate': 1e-3,
            'use_quaternion': True,
            'use_stable_label': True
        },
        
        # # 2. 使用四元数，不使用stable label
        # {
        #     'hidden_dims': [512, 512, 512],
        #     'num_layers': 3,
        #     'dropout_rate': 0.1,
        #     'batch_size': 2048,
        #     'learning_rate': 1e-3,
        #     'use_quaternion': True,
        #     'use_stable_label': False
        # },
        #
        # # 3. 使用简化表示(xy+rz) + stable label
        # {
        #     'hidden_dims': [512, 512, 512],
        #     'num_layers': 3,
        #     'dropout_rate': 0.1,
        #     'batch_size': 2048,
        #     'learning_rate': 1e-3,
        #     'use_quaternion': False,
        #     'use_stable_label': True
        # }
    ]
    
    dataset_types = ["SharedGraspNetwork_bottle_table_experiment_data"]
    dataset_ids = [57]
    seeds = [22]
    train_splits = [0.7] 
    data_ranges = [70000, 28000, 14000, 2800]
    data_ratios = [0.3, 0.6, 0.9, 0.95, 0.99]
    grasp_types = ['table']
    state_types = ['init']

    # 运行所有实验组合
    for config in experiment_configs:
        for dataset_type in dataset_types:
            for data_id in dataset_ids:
                # 每次实验前强制清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for train_split in train_splits:
                    for seed in seeds:
                        for grasp_type in grasp_types:
                            for state_type in state_types:
                                for data_ratio in data_ratios:
                                    for data_range in data_ranges:
                                        run_grasp_experiment(
                                            hidden_dims=config['hidden_dims'],
                                            num_layers=config['num_layers'],
                                            dropout_rate=config['dropout_rate'],
                                            batch_size=config['batch_size'],
                                            learning_rate=config['learning_rate'],
                                            dataset_type=dataset_type,
                                            data_id=data_id,
                                            train_split=train_split,
                                            data_ratio=data_ratio,
                                            data_range=data_range,
                                            seed=seed,
                                            grasp_type=grasp_type,
                                            state_type=state_type,
                                            use_quaternion=config['use_quaternion'],
                                            use_stable_label=config['use_stable_label']
                                        )
                                        time.sleep(2)  # 等待内存释放

if __name__ == '__main__':
    main()