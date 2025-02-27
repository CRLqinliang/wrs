import subprocess
import os
import sys

def run_experiment(dateset_type, data_id, train_split, data_ratio, seed):
    """运行单个实验"""
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\shared_grasp_network\shared_grasp\shared_grasp_network_train.py"
    data_path = os.path.join(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle",
                             f'{dateset_type}_{data_id}.pickle'.format(dateset_type=dateset_type,
                                                                                             data_id=data_id))

    save_path = os.path.join(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\shared_best_model",
                             f'{dateset_type}_{data_id}.pth'.format(dateset_type=dateset_type,
                                                                                          data_id=data_id))
    
    cmd = [
        'python', script_path,  # 使用完整路径
        '--data_path', data_path,
        '--random_seed', str(seed),
        '--model_save_path', save_path,
        '--data_ratio', str(data_ratio),
        '--num_epochs', '200',
        '--batch_size', '512',
        '--learning_rate', '4e-3',
        '--input_dim', '24',
        '--output_dim', str(data_id),
        '--early_stop_patience', '60',
        '--train_split', str(train_split),
        '--stable_label', 'True',
        '--wandb_project', 'shared_grasp_experiments_paper',
        '--wandb_name', f'multi_label_grasp_{dateset_type}_bottle_{data_id}_seed_{seed}'
    ]
    
    print(f"\n开始训练 网络 (dataset={data_id}, seed={seed})...")
    subprocess.run(cmd)
    print(f" 网络训练完成 (dataset={data_id}, seed={seed})\n")

def main():

    # 设置不同的网络架构、数据集ID和随机种子
    dateset_types = ['SharedGraspNetwork_bottle_experiment_data']
    dataset_ids = [57, 83, 109]
    seeds = [42]
    train_split_set = [0.7]
    data_ratio_set = [0.3, 0.6, 0.9, 0.99]
    
    # 依次运行所有组合的实验
    for dateset_type in dateset_types:
        for data_id in dataset_ids:
            for train_split in train_split_set:
                for data_ratio in data_ratio_set:
                    for seed in seeds:
                        run_experiment(dateset_type, data_id, train_split, data_ratio, seed)

if __name__ == '__main__':
    main()