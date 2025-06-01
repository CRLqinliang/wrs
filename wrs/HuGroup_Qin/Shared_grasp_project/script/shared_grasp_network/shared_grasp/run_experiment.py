import subprocess
import os
import sys

def run_experiment(dateset_type, data_id, train_split, mixtuers_num, batch_size, use_quaternion, use_stable_label, data_ratio, seed):
    """运行单个实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\shared_grasp_network\shared_grasp\shared_grasp_network_train.py"
    data_path = os.path.join(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle",
                             f'{dateset_type}_{data_id}.pickle'.format(dateset_type=dateset_type,
                              data_id=data_id))

    save_path = os.path.join(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\shared_best_model",
                             f'MDN_shared_grasp_{dateset_type}_{data_id}.pth'.format(dateset_type=dateset_type,
                                                                                          data_id=data_id))
    
    cmd = [
        'python', script_path,  # 使用完整路径
        '--data_path', data_path,
        '--seed', str(seed),
        '--model_save_path', save_path,
        '--data_ratio', str(data_ratio),
        '--train_split', str(train_split),
        '--use_quaternion', '1' if use_quaternion else '0',
        '--use_stable_label', '0' if use_stable_label else '0',
        '--n_mixtures', str(mixtuers_num),
        '--batch_size', str(batch_size),
        '--wandb_project', 'grasp_mdn',
        '--wandb_name', f'MDN_shared_grasp_{dateset_type}_bottle_{data_id}_ratio_{data_ratio}_n{mixtuers_num}_q{use_quaternion}_s{use_stable_label}_seed_{seed}'
    ]
    
    print(f"\n开始训练 网络 (dataset={data_id}, seed={seed})...")
    subprocess.run(cmd)
    print(f" 网络训练完成 (dataset={data_id}, seed={seed})\n")

def main():
        # 实验配置
    experiment_configs = [
        # # 1. 使用四元数 + stable label
        # {
        #     'use_quaternion': True,
        #     'use_stable_label': True,
        #     'batch_size': 256,
        # },
        
        # 2. 使用四元数，不使用stable label
        {
            'use_quaternion': True,
            'use_stable_label': False,
            'batch_size': 256
        },

        # 3. 使用简化表示(xy+rz) + stable label
        {
            'use_quaternion': False,
            'use_stable_label': True,
            'batch_size': 256
        }
    ]

    # 设置不同的网络架构、数据集ID和随机种子
    dateset_types = ['SharedGraspNetwork_bottle_experiment_data']
    dataset_ids = [83, 109, 352]
    seeds = [42]
    train_split_set = [0.7]
    data_ratio_set = [0]
    n_mixtures_set = [50]
    
    # 依次运行所有组合的实验
    for config in experiment_configs:
        for dateset_type in dateset_types:
            for data_id in dataset_ids:
                for train_split in train_split_set:
                    for data_ratio in data_ratio_set:
                        for seed in seeds:
                            for mixtuers_num in n_mixtures_set:
                                run_experiment(dateset_type, data_id, train_split, mixtuers_num, config['batch_size'], config['use_quaternion'],
                                           config['use_stable_label'], data_ratio, seed)

if __name__ == '__main__':
    main()