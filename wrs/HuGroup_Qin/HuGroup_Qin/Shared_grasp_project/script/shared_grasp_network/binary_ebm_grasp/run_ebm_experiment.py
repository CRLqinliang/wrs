import subprocess
import os
import sys
import time
import gc
import torch

def run_ebm_experiment(hidden_dims, num_layers, dropout_rate, batch_size, 
                      learning_rate, temperature, dataset_type, data_id, train_split, data_ratio, seed):
    """运行单个EBM实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, f'EBM_grasp_network_train.py')
    
    # 数据路径
    data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/{dataset_type}_{data_id}.pickle')

    grasp_data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/bottle_grasp_{data_id}.pickle')
    
    # 构建模型保存路径
    exp_name = f'ebm_selu_{dataset_type}_{data_id}_h{len(hidden_dims)}_{batch_size}_lr{learning_rate}_t{temperature}_dataratio_{data_ratio}_trainsplit_{train_split}'
    model_save_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                   f'model/Binary_ebm_model/best_model_grasp_{exp_name}.pth')
    
    # 将hidden_dims列表转换为命令行参数字符串
    hidden_dims_str = ' '.join(map(str, hidden_dims))
    cmd = [
        'python', script_path,
        '--data_path', data_path,
        '--grasp_data_path', grasp_data_path,
        '--model_save_path', model_save_path,
        '--multiscale_network', 'False',
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
        '--data_ratio', str(data_ratio),
        '--seed', str(seed),
        '--wandb_project', 'EBM_grasp_experiments_paper',
        '--wandb_name', exp_name
    ]
    
    print(f"\n开始训练 EBM 网络...")
    print(f"配置: hidden_dims={hidden_dims}, num_layers={num_layers}, batch_size={batch_size}, lr={learning_rate}, temp={temperature}")
    subprocess.run(cmd)
    print(f"EBM 网络训练完成\n")

def main():
    # 确保模型保存目录存在
    os.makedirs('model/Binary_ebm_model', exist_ok=True)
    
    # 实验配置
    experiment_configs = [
        #1. 基础SeLU结构
        {
            'hidden_dims': [512, 512, 512],  # 三层相同宽度的隐藏层
            'num_layers': 3,                 # 层数
            'dropout_rate': 0.1,             # SeLU推荐的较低dropout
            'batch_size': 2048,              # 较大的batch size
            'learning_rate': 1e-3,           # SeLU适合较大的学习率
            'temperature': 0.5               # 适中的temperature
        },
        
        # 2. 更宽的SeLU结构
        # {
        #     'hidden_dims': [1024, 1024, 1024],  # 更宽的隐藏层
        #     'num_layers': 3,                    # 保持层数不变
        #     'dropout_rate': 0.1,                # 保持较低dropout
        #     'batch_size': 1024,                 # 减小batch size以适应更大的网络
        #     'learning_rate': 5e-4,              # 略微降低学习率
        #     'temperature': 0.1                  # 保持相同的temperature
        # },
        
        # # 3. 更深的SeLU结构
        # {
        #     'hidden_dims': [512, 512, 512, 512, 512],  # 五层隐藏层
        #     'num_layers': 5,                           # 增加层数
        #     'dropout_rate': 0.15,                      # 略微增加dropout
        #     'batch_size': 1024,                        # 减小batch size
        #     'learning_rate': 5e-4,                     # 降低学习率
        #     'temperature': 0.08                        # 略微降低temperature
        # },
        
        # 4. 金字塔SeLU结构
        # {
        #     'hidden_dims': [256, 512, 256],  # 金字塔结构
        #     'num_layers': 3,                 # 三层结构
        #     'dropout_rate': 0.1,             # 保持较低dropout
        #     'batch_size': 2048,              # 较大的batch size
        #     'learning_rate': 1e-3,           # 较大的学习率
        #     'temperature': 0.12              # 略微增加temperature
        # },
        
        # # 5. 轻量级SeLU结构
        # {
        #     'hidden_dims': [256, 256, 256],  # 较小的隐藏层
        #     'num_layers': 3,                 # 保持层数不变
        #     'dropout_rate': 0.05,            # 非常低的dropout
        #     'batch_size': 4096,              # 更大的batch size
        #     'learning_rate': 2e-3,           # 更大的学习率
        #     'temperature': 0.15              # 较大的temperature
        # }
    ]
    
    dataset_types = ["SharedGraspNetwork_bottle_experiment_data"]
    dataset_ids = [57, 83, 109]
    seeds = [42]
    train_splits = [0.7] 
    data_ratio = [0.3, 0.6, 0.9, 0.99] # 以这个比例往后的数据大小； 90% 70% 50% 30% 10% 1%
    
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
                        for ratio in data_ratio:
                            run_ebm_experiment(
                                hidden_dims=config['hidden_dims'],
                                num_layers=config['num_layers'],
                                dropout_rate=config['dropout_rate'],
                                batch_size=config['batch_size'],
                                learning_rate=config['learning_rate'],
                                temperature=config['temperature'],
                                dataset_type=dataset_type,
                                data_id=data_id,
                                train_split = train_split,
                                data_ratio = ratio,
                                seed=seed
                            )
                            time.sleep(5) # wait for RAM to be released

if __name__ == '__main__':
    main()