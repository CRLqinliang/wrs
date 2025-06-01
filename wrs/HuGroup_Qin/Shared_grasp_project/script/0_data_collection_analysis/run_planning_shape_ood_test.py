import subprocess
import os
import sys
import time
import gc
import torch
import psutil

def clear_memory():
    """清理内存的辅助函数"""
    gc.collect()  # 触发Python的垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
    
def run_ebm_experiment(hidden_dims, num_layers, dropout_rate, batch_size, 
                      learning_rate, temperature, objects_with_ids, 
                      train_split, data_range, seed, use_quaternion,
                      use_stable_label, grasp_type, state_type, target_env_type):
    """运行单个EBM实验"""

    # 获取当前文件的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, f'planning_shape_ood_test.py')
    
    # 构建数据路径列表
    env_data_paths = []
    grasp_data_paths = []
    validate_grasp_data_paths = []
    validate_env_data_paths = []
    base_path = 'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps'
    env_base_path = 'E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes'

    for obj, info in target_env_type.items():
        grasp_data_paths.append(os.path.join(base_path, obj, f"{obj.lower()}_grasp_{info['id']}.pickle"))
        env_data_paths.append(os.path.join(env_base_path, f"{obj.lower()}.stl"))
        grasp_id = info['id']

    for obj, info in objects_with_ids.items():
        validate_grasp_data_paths.append(os.path.join(base_path, obj, f"{obj.lower()}_grasp_{info['id']}.pickle"))
        validate_env_data_paths.append(os.path.join(env_base_path, f"{obj.lower()}.stl"))

    # 构建模型保存路径
    objects_str = '_'.join([f"{obj.lower()}{info['id']}" for obj, info in objects_with_ids.items()])
    exp_name = f'ebm_multi_{objects_str}_h{len(hidden_dims)}_b{batch_size}_lr{learning_rate}_t{temperature}_r{data_range}_s{train_split}_q{int(use_quaternion)}_sl{int(use_stable_label)}_g{grasp_type}_st{state_type}'
    model_save_path = os.path.join(f'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project',
                                f'model/feasible_best_model/best_model_grasp_{exp_name}.pth')
    
    grasp_data_paths_str = ','.join(grasp_data_paths)

    cmd = [
        'python', script_path,
        '--model_init_path', model_save_path,  # 初始状态模型路径
        '--model_goal_path', model_save_path,  # 目标状态模型路径
        '--target_env_type', ','.join(env_data_paths),
        '--grasp_data_path', grasp_data_paths_str,
        '--grasp_ids', str(grasp_id),
        '--objects_str', objects_str,
        '--target_objects_str', str(env_data_paths[0].split('\\')[-1]),
        '--validate_grasp_data_path', ','.join(validate_grasp_data_paths),
        '--validate_env_data_path', ','.join(validate_env_data_paths)
    ]

    # 在子进程运行前清理内存
    clear_memory()
    
    # 使用subprocess.Popen而不是run，这样我们可以更好地控制进程
    process = subprocess.Popen(cmd)
    
    # 等待进程完成
    process.wait()
    
    # 检查进程是否正常结束
    if process.returncode != 0:
        print(f"警告：exp_name {exp_name} 的实验未正常结束")
        
    # 确保进程完全终止
    try:
        process.kill()
    except:
        pass
        
    # 实验结束后清理内存
    clear_memory()
        
def main():

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
            'use_stable_label': False,
            'grasp_type': 'robot_table',
            'state_type': 'init'
        }
    ]
    # 实验配置
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

    target_env_types = [
        # {'Bottle': object_info['Bottle']},
        {'Bunny':object_info['Bunny']},
        {'Mug':object_info['Mug']},
        {'Power_drill':object_info['Power_drill']}
    ]
    
    # 设置进程优先级
    try:
        process = psutil.Process(os.getpid())
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows
    except:
        pass

    for config in experiment_configs:
        for objects_with_ids in object_combinations:
            for target_env_type in target_env_types:
                # if any(key in objects_with_ids.keys() for key in target_env_type.keys()):
                #     continue
                run_ebm_experiment(
                        hidden_dims=config['hidden_dims'],
                        num_layers=config['num_layers'],
                        dropout_rate=config['dropout_rate'],
                        batch_size=config['batch_size'],
                        learning_rate=config['learning_rate'],
                        temperature=config['temperature'],
                        objects_with_ids=objects_with_ids,
                        target_env_type=target_env_type,
                        train_split=0.7,
                        data_range=150000,
                        seed=22,
                        use_quaternion=config['use_quaternion'],
                        use_stable_label=config['use_stable_label'],
                        grasp_type=config['grasp_type'],
                        state_type=config['state_type']
                            )
        
        # 强制等待一段时间确保内存释放
        time.sleep(5)
        
        # 打印当前内存使用情况
        process = psutil.Process(os.getpid())
        print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    print("\n所有实验完成")

if __name__ == '__main__':
    main()