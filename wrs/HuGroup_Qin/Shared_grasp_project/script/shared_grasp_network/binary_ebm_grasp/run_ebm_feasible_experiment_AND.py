import subprocess
import os
import sys
import time
import gc
import torch

def run_ebm_and_evaluation(hidden_dims, num_layers, dropout_rate, batch_size, 
                         dataset_type, data_id, train_split, use_quaternion, 
                         use_stable_label, model_mode, model_ratios, data_ratio):
    """运行EBM模型的AND评估
    
    Args:
        model_mode: 'standard'(标准，使用两个模型) 或 'extended'(扩展，使用四个模型)
        model_ratios: 模型比例，格式取决于mode:
            - standard: (init_ratio, goal_ratio)
            - extended: (robot_init_ratio, table_init_ratio, robot_goal_ratio, table_goal_ratio)
            
    注意:
        在评估过程中，各个模型的能量输出会先进行标准化处理，然后才会相加，
        这确保了不同模型的贡献是平衡的，防止某个模型因数值范围大而主导结果。
    """
    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, f'EBM_grasp_network_AND_evaluation.py')
    
    # 数据路径
    data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/{dataset_type}_{data_id}.pickle') # temp exp

    grasp_data_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
        f'grasps/Bottle/bottle_grasp_{data_id}.pickle')
    
    # 根据模式构建模型路径和命令行参数
    cmd = [
        'python', script_path,
        '--test_dataset', data_path,
        '--grasp_pickle_file', grasp_data_path,
        '--model_mode', model_mode,
        '--hidden_dims', *' '.join(map(str, hidden_dims)).split(),
        '--num_layers', str(num_layers),
        '--dropout_rate', str(dropout_rate),
        '--batch_size', str(batch_size),
        '--train_split', str(train_split),
        '--use_quaternion', '1' if use_quaternion else '0',
        '--use_stable_label', '1' if use_stable_label else '0',
        '--data_ratio', str(data_ratio)
    ]
    
    # 根据模式设置wandb项目和实验名称
    if model_mode == 'standard':
        # 标准模式 - 两个模型
        init_ratio, goal_ratio = model_ratios
        
        # 构建模型路径
        grasp_type = 'robot_table'  # 标准模式下使用robot_table类型
        init_exp_name = f'ebm_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{init_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_g{grasp_type}_stinit'
        goal_exp_name = f'ebm_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{goal_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_g{grasp_type}_stinit'
        
        init_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                      f'model/feasible_best_model/best_model_grasp_{init_exp_name}.pth')
        goal_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                      f'model/feasible_best_model/best_model_grasp_{goal_exp_name}.pth')
        
        # 添加模型路径到命令行参数
        cmd.extend(['--model_init_path', init_model_path])
        cmd.extend(['--model_goal_path', goal_model_path])
        
        # wandb配置
        and_exp_name = f'ebm_and_{dataset_type}_{data_id}_i{init_ratio}_g{goal_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_g{grasp_type}'
        cmd.extend(['--wandb_project', 'EBM_grasp_AND_evaluation'])
        cmd.extend(['--wandb_name', and_exp_name])
        
        print(f"\n开始 EBM AND 标准评估...")
        print(f"使用模型 - 初始状态: {os.path.basename(init_model_path)}")
        print(f"使用模型 - 目标状态: {os.path.basename(goal_model_path)}")
        
    elif model_mode == 'extended':
        # 扩展模式 - 四个模型
        robot_init_ratio, table_init_ratio, robot_goal_ratio, table_goal_ratio = model_ratios
        
        # 构建模型路径 - Robot
        robot_init_exp_name = f'ebm_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{robot_init_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_grobot_stinit'
        robot_init_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                           f'model/feasible_best_model/best_model_grasp_{robot_init_exp_name}.pth')
        
        # 构建模型路径 - Table
        table_init_exp_name = f'ebm_SharedGraspNetwork_bottle_table_experiment_data_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{table_init_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_gtable_stinit'
        table_init_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                           f'model/feasible_best_model/best_model_grasp_{table_init_exp_name}.pth')
        
        # 构建模型路径 - Robot
        robot_goal_exp_name = f'ebm_{dataset_type}_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{robot_goal_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_grobot_stinit'
        robot_goal_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                          f'model/feasible_best_model/best_model_grasp_{robot_goal_exp_name}.pth')
        
        # 构建模型路径 - Table
        table_goal_exp_name = f'ebm_SharedGraspNetwork_bottle_table_experiment_data_{data_id}_h{len(hidden_dims)}_b{batch_size}_lr0.001_t0.5_r{table_goal_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_gtable_stinit'
        table_goal_model_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                          f'model/feasible_best_model/best_model_grasp_{table_goal_exp_name}.pth')
        
        # 添加模型路径到命令行参数
        cmd.extend(['--model_robot_init_path', robot_init_model_path])
        cmd.extend(['--model_table_init_path', table_init_model_path])
        cmd.extend(['--model_robot_goal_path', robot_goal_model_path])
        cmd.extend(['--model_table_goal_path', table_goal_model_path])
        
        # wandb配置
        and_exp_name = f'ebm_and_rt_{dataset_type}_{data_id}_ri{robot_init_ratio}_ti{table_init_ratio}_rg{robot_goal_ratio}_tg{table_goal_ratio}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}'
        cmd.extend(['--wandb_project', 'EBM_grasp_AND_RT_evaluation'])
        cmd.extend(['--wandb_name', and_exp_name])
        
        print(f"\n开始 EBM AND 扩展评估 (Robot+Table)...")
        print(f"使用模型 - Robot初始状态: {os.path.basename(robot_init_model_path)}")
        print(f"使用模型 - Table初始状态: {os.path.basename(table_init_model_path)}")
        print(f"使用模型 - Robot目标状态: {os.path.basename(robot_goal_model_path)}")
        print(f"使用模型 - Table目标状态: {os.path.basename(table_goal_model_path)}")
    
    # 设置输入维度
    n_categories = 5  # 稳定标签的one-hot编码维度
    if use_quaternion:
        input_dim = 7 + n_categories + 7  # 四元数表示：7(物体位姿) + 5(稳定标签) + 7(抓取位姿)
    else:
        input_dim = 3 + n_categories + 7  # 欧拉角表示：3(物体位姿) + 5(稳定标签) + 7(抓取位姿)
    
    cmd.extend(['--input_dim', str(input_dim)])
    
    print(f"配置: hidden_dims={hidden_dims}, num_layers={num_layers}, batch_size={batch_size}")
    print(f"表示方式: use_quaternion={use_quaternion}, use_stable_label={use_stable_label}")
    
    subprocess.run(cmd)
    print(f"EBM AND 评估完成\n")

def main():
    # 确保输出目录存在
    os.makedirs('output/EBM_AND_evaluation', exist_ok=True)
    
    # 实验配置
    experiment_configs = [
        # 使用四元数 + stable label
        {
            'hidden_dims': [512, 512, 512],
            'num_layers': 3,
            'dropout_rate': 0.1,
            'batch_size': 2048,  
            'use_quaternion': True,
            'use_stable_label': True
        }
    ]
    
    dataset_types = ["SharedGraspNetwork_bottle_experiment_data"]
    dataset_ids = [57]
    train_splits = [0.7]
    data_ratios = [0.9] # 10%的数据来测试
    # data_ratios = [75000]

    # 运行标准模式实验 (两个模型)
    standard_model_ratio_pairs = [
        (0.3, 0.3),   # 低数据量
        (0.6, 0.6),   # 中数据量
        (0.9, 0.9),   # 高数据量
        (0.95, 0.95),
        (0.99, 0.99)
        #(75000, 75000) # temp test.
    ]
    
    # 运行扩展模式实验 (四个模型)
    extended_model_ratio_tuples = [
        #(0.3, 0.3, 0.3, 0.3),  # 低数据量
        (0.6, 0.6, 0.6, 0.6),  # 中数据量
        (0.9, 0.9, 0.9, 0.9),  # 高数据量
        (0.95, 0.95, 0.95, 0.95),
        (0.99, 0.99, 0.99, 0.99)
    ]
    test_model = ['standard'] # 'standard' or 'extended'

    if 'standard' in test_model:
        # 运行所有标准模式实验
        for data_ratio in data_ratios:
            for config in experiment_configs:
                for dataset_type in dataset_types:
                    for data_id in dataset_ids:
                        for train_split in train_splits:
                            for model_ratios in standard_model_ratio_pairs:
                                # 每次实验前强制清理内存
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print(f"开始标准模式评估 {dataset_type} {data_id}")
                                run_ebm_and_evaluation(
                                    hidden_dims=config['hidden_dims'],
                                    num_layers=config['num_layers'],
                                    dropout_rate=config['dropout_rate'],
                                    batch_size=config['batch_size'],
                                    dataset_type=dataset_type,
                                    data_id=data_id,
                                    train_split=train_split,
                                    use_quaternion=config['use_quaternion'],
                                    use_stable_label=config['use_stable_label'],
                                    model_mode='standard',
                                    model_ratios=model_ratios,
                                    data_ratio=data_ratio
                                )
                                time.sleep(2)  # 等待内存释放

    if 'extended' in test_model:
        # 运行所有扩展模式实验
        for data_ratio in data_ratios:
            for config in experiment_configs:
                for dataset_type in dataset_types:
                    for data_id in dataset_ids:
                        for train_split in train_splits:
                            for model_ratios in extended_model_ratio_tuples:
                                # 每次实验前强制清理内存
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print(f"开始扩展模式评估 {dataset_type} {data_id}")
                                run_ebm_and_evaluation(
                                    hidden_dims=config['hidden_dims'],
                                    num_layers=config['num_layers'],
                                    dropout_rate=config['dropout_rate'],
                                    batch_size=config['batch_size'],
                                    dataset_type=dataset_type,
                                    data_id=data_id,
                                    train_split=train_split,
                                    use_quaternion=config['use_quaternion'],
                                    use_stable_label=config['use_stable_label'],
                                    model_mode='extended',
                                    model_ratios=model_ratios,
                                    data_ratio=data_ratio
                                )
                                time.sleep(2)  # 等待内存释放

if __name__ == '__main__':
    main()
