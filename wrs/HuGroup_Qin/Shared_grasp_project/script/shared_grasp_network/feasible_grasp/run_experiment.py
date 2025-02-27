import subprocess
import os
import sys

def run_experiment(network_type, dateset_type, data_id, train_split, seed):
    """运行单个实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = r"H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\feasible_grasp\grasp_network_train.py"
    data_path = os.path.join(r"H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle",
                             f'feasible_grasp_{dateset_type}_{data_id}.pickle'.format(dateset_type=dateset_type,
                                                                                             data_id=data_id))

    save_path = os.path.join(r"H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model",
                             f'feasible_grasp_{dateset_type}_{data_id}.pth'.format(dateset_type=dateset_type,
                                                                                          data_id=data_id))
    
    cmd = [
        'python', script_path,  # 使用完整路径
        '--network_type', network_type,
        '--data_path', data_path,
        '--random_seed', str(seed),
        '--model_save_path', save_path,
        '--num_epochs', '500',
        '--batch_size', '512',
        '--learning_rate', '4e-3',
        '--input_dim', '10',
        '--output_dim', str(data_id),
        '--early_stop_patience', '60',
        '--min_precision', '0.8',
        '--min_recall', '0.4',
        '--train_split', str(train_split),
        '--use_stable_label',
        '--wandb_project', 'feasible_grasp',
        '--wandb_name', f'feasible_grasp_{dateset_type}_{network_type}_bottle_{data_id}_seed_{seed}'
    ]
    
    print(f"\n开始训练 {network_type} 网络 (dataset={data_id}, seed={seed})...")
    subprocess.run(cmd)
    print(f"{network_type} 网络训练完成 (dataset={data_id}, seed={seed})\n")

def main():

    # 设置不同的网络架构、数据集ID和随机种子
    network_types = ['mlp']
    dateset_types = ['robot']
    dataset_ids = [109]
    seeds = [22]
    train_split_set = [0.4, 0.6, 0.8]
    
    # 依次运行所有组合的实验
    for network_type in network_types:
        for dateset_type in dateset_types:
            for data_id in dataset_ids:
                for train_split in train_split_set:
                    for seed in seeds:
                        run_experiment(network_type, dateset_type, data_id, train_split, seed)

if __name__ == '__main__':
    main()