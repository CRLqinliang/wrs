import subprocess
import os
import sys
import time
import gc
import torch
import psutil
import signal

def clear_memory():
    """清理内存的辅助函数"""
    gc.collect()  # 触发Python的垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清理GPU缓存
    
def run_ebm_experiment(grasp_id):
    """运行单个EBM实验"""
    try:
        # 获取当前文件的目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, f'planning_effiency_test.py')
        
        # 构建模型路径
        model_init_path = os.path.join(r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\Binary_ebm_model',
            f'best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_{str(grasp_id)}_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl0.pth')
        model_goal_path = model_init_path

        cmd = [
            'python', script_path,
            '--model_init_path', model_init_path,  # 初始状态模型路径
            '--model_goal_path', model_goal_path,  # 目标状态模型路径
        ]

        # 在子进程运行前清理内存
        clear_memory()
        
        # 使用subprocess.Popen而不是run，这样我们可以更好地控制进程
        process = subprocess.Popen(cmd)
        
        # 等待进程完成
        process.wait()
        
        # 检查进程是否正常结束
        if process.returncode != 0:
            print(f"警告：grasp_id {grasp_id} 的实验未正常结束")
            
        # 确保进程完全终止
        try:
            process.kill()
        except:
            pass
            
        # 实验结束后清理内存
        clear_memory()
        
    except Exception as e:
        print(f"运行grasp_id {grasp_id} 时发生错误: {str(e)}")
        # 确保即使发生错误也清理内存
        clear_memory()

def main():
    # 实验配置
    grasp_ids = [352]
    
    # 设置进程优先级
    try:
        process = psutil.Process(os.getpid())
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)  # Windows
    except:
        pass
    
    for grasp_id in grasp_ids:
        print(f"\n开始处理 grasp_id: {grasp_id}")
        run_ebm_experiment(grasp_id)
        
        # 强制等待一段时间确保内存释放
        time.sleep(5)
        
        # 打印当前内存使用情况
        process = psutil.Process(os.getpid())
        print(f"当前内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    print("\n所有实验完成")

if __name__ == '__main__':
    main()