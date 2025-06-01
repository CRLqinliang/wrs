import subprocess
import os
import sys
import time
import gc
import torch

def run_ebm_experiment(hidden_dims, num_layers, dropout_rate, batch_size, 
                      learning_rate, temperature, objects_with_ids, 
                      train_split, data_range, seed, use_quaternion, use_stable_label):
    """运行单个EBM实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, f'EBM_grasp_network_train.py')
    
    # 数据路径
    data_paths = []
    grasp_data_paths = []
    base_path = 'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps'
    
    for obj, info in objects_with_ids.items():
        data_paths.append(os.path.join(base_path, obj, f"{info['dataset_type']}_{info['id']}.pickle"))
        grasp_data_paths.append(os.path.join(base_path, obj, f"{obj.lower()}_grasp_{info['id']}.pickle"))
    
    # 构建模型保存路径
    objects_str = '_'.join([f"{obj.lower()}{info['id']}" for obj, info in objects_with_ids.items()])
    exp_name = f'ebm_multi_{objects_str}_h{len(hidden_dims)}_b{batch_size}_lr{learning_rate}_t{temperature}_r{data_range}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}'
    model_save_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                   f'model/Binary_ebm_model/best_model_grasp_{exp_name}.pth')
    
    # 将hidden_dims列表转换为命令行参数字符串
    hidden_dims_str = ' '.join(map(str, hidden_dims))

    # 将数据路径列表转换为命令行参数
    data_paths_str = ','.join(data_paths)
    grasp_data_paths_str = ','.join(grasp_data_paths)

    cmd = [
        'python', script_path,
        '--data_path', data_paths_str,
        '--grasp_data_path', grasp_data_paths_str,
        '--model_save_path', model_save_path,
        '--hidden_dims', *hidden_dims_str.split(),
        '--num_layers', str(num_layers),
        '--dropout_rate', str(dropout_rate),
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--weight_decay', '1e-4',
        '--temperature', str(temperature),
        '--num_epochs', '80',
        '--early_stop_patience', '8',
        '--train_split', str(train_split),
        '--data_range', str(data_range),
        '--seed', str(seed),
        '--use_quaternion', '1' if use_quaternion else '0',
        '--use_stable_label', '0' if use_stable_label else '0',
        '--wandb_project', 'EBM_grasp_experiments_paper',
        '--wandb_name', exp_name
    ]
    
    print(f"\n开始训练 EBM 网络...")
    print(f"配置: hidden_dims={hidden_dims}, num_layers={num_layers}, batch_size={batch_size}, lr={learning_rate}, temp={temperature}")
    print(f"表示方式: use_quaternion={use_quaternion}, use_stable_label={use_stable_label}")
    subprocess.run(cmd)
    print(f"EBM 网络训练完成\n")

def main():
    # 确保模型保存目录存在
    os.makedirs('model/Binary_ebm_model', exist_ok=True)
    # 实验配置
    experiment_configs = [
        {
            'hidden_dims': [512, 512, 512],
            'num_layers': 3,
            'dropout_rate': 0.1,
            'batch_size': 2048,
            'learning_rate': 1e-3,
            'temperature': 0.5,
            'use_quaternion': True,
            'use_stable_label': False
        }
    ]
        # 定义物体信息映射
    object_info = {
        'Bottle': {
            'id': 352,
            'dataset_type': 'SharedGraspNetwork_bottle_experiment_data_final'
        },
        'Bunny': {
            'id': 252,
            'dataset_type': 'SharedGraspNetwork_bunny_experiment_data_final'
        },
        'Mug': {
            'id': 195,
            'dataset_type': 'SharedGraspNetwork_mug_experiment_data_final'
        },
        'Power_drill': {
            'id': 199,
            'dataset_type': 'SharedGraspNetwork_power_drill_experiment_data_final'
        }
    }

    object_combinations = [
        {'Bottle': object_info['Bottle']},
        {'Bottle': object_info['Bottle'], 'Bunny': object_info['Bunny']},
        {'Bottle': object_info['Bottle'], 'Mug': object_info['Mug']},
        {'Bottle': object_info['Bottle'], 'Power_drill': object_info['Power_drill']},
        {'Bottle': object_info['Bottle'], 'Bunny': object_info['Bunny'], 'Mug': object_info['Mug'] },
        {'Bottle': object_info['Bottle'], 'Mug': object_info['Mug'], 'Power_drill': object_info['Power_drill']},
        {'Bottle': object_info['Bottle'], 'Power_drill': object_info['Power_drill'],  'Bunny': object_info['Bunny']}
    ]
    
    # dataset_ids = [57, 83, 109, 352]
    seeds = [22]
    train_splits = [0.7] 
    # data_ratio = [0]
    data_range = [150000]

    # 运行所有实验组合
    for config in experiment_configs:
        for objects_with_ids in object_combinations:
            print(f"\n开始训练物体组合: {list(objects_with_ids.keys())}")
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
                            objects_with_ids=objects_with_ids,
                            train_split=train_split,
                            data_range=num,
                            seed=seed,
                            use_quaternion=config['use_quaternion'],
                            use_stable_label=config['use_stable_label']
                        )
                        time.sleep(2)  # 等待内存释放

if __name__ == '__main__':
    main()