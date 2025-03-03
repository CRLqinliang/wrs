import subprocess
import os
import sys
import time
import gc
import torch

def run_ebm_experiment(hidden_dims, num_layers, dropout_rate, batch_size, 
                      learning_rate, temperature, dataset_type, data_id, 
                      train_split, data_range, seed, use_quaternion,
                      use_stable_label, grasp_type, state_type):
    """运行单个EBM实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, f'EBM_feasible_grasp_network_train.py')
    
    # 数据路径
    data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/{dataset_type}_{data_id}.pickle')

    grasp_data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/bottle_grasp_{data_id}.pickle')
    
    # 构建模型保存路径
    exp_name = f'ebm_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr{learning_rate}_t{temperature}_r{data_range}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_g{grasp_type}_st{state_type}'
    model_save_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
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
        '--weight_decay', '1e-4',
        '--temperature', str(temperature),           
        '--num_epochs', '80',
        '--early_stop_patience', '10',
        '--train_split', str(train_split),
        '--data_range', str(data_range),
        '--seed', str(seed),
        '--use_quaternion', '1' if use_quaternion else '0',
        '--use_stable_label', '1' if use_stable_label else '0',
        '--grasp_type', grasp_type,
        '--state_type', state_type,
        '--wandb_project', 'EBM_grasp_experiments_paper',
        '--wandb_name', exp_name
    ]
    
    print(f"\n开始训练 EBM 网络...")
    print(f"配置: hidden_dims={hidden_dims}, num_layers={num_layers}, batch_size={batch_size}, lr={learning_rate}, temp={temperature}")
    print(f"表示方式: use_quaternion={use_quaternion}, use_stable_label={use_stable_label}")
    print(f"抓取类型: grasp_type={grasp_type}, state_type={state_type}")
    subprocess.run(cmd)
    print(f"EBM 网络训练完成\n")

def main():
    # 确保模型保存目录存在
    os.makedirs('model/Binary_ebm_model', exist_ok=True)
    
    # 实验配置
    experiment_configs = [
        # 1. 使用四元数 + stable label
        {
            'hidden_dims': [512, 512, 512],
            'num_layers': 3,
            'dropout_rate': 0.1,
            'batch_size': 2048,
            'learning_rate': 1e-3,
            'temperature': 0.5,
            'use_quaternion': True,
            'use_stable_label': True,
            'grasp_type': 'robot_table',
            'state_type': 'init' # 使用init保证数据量一致
        }
    ]
    
    dataset_types = ["SharedGraspNetwork_bottle_experiment_data"]
    dataset_ids = [57, 83, 109, 352]
    seeds = [22]
    train_splits = [0.7] 
    # data_ratio = [0.3, 0.6, 0.9, 0.95, 0.99]
    data_range = [75000]
    
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
                        for num in data_range:
                            run_ebm_experiment(
                                hidden_dims=config['hidden_dims'],
                                num_layers=config['num_layers'],
                                dropout_rate=config['dropout_rate'],
                                batch_size=config['batch_size'],
                                learning_rate=config['learning_rate'],
                                temperature=config['temperature'],
                                dataset_type=dataset_type,
                                data_id=data_id,
                                train_split=train_split,
                                data_range = num,
                                seed=seed,
                                use_quaternion=config['use_quaternion'],
                                use_stable_label=config['use_stable_label'],
                                grasp_type=config['grasp_type'],
                                state_type=config['state_type']
                            )
                            time.sleep(2)  # 等待内存释放

if __name__ == '__main__':
    main()