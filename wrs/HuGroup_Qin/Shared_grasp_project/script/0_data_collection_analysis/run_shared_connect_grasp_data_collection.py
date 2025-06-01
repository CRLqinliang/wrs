import subprocess
import os
import sys
import argparse
import time
import psutil  # 需要先 pip install psutil


def run_shared_grasp_data_collection(grasp_ids=[109], total_iterations=int(1e4), save_batch_size=1000):
    """运行共享抓取数据收集程序
    
    Args:
        grasp_ids (list): 要处理的抓取ID列表
        total_iterations (int): 总迭代次数
        save_batch_size (int): 每批保存的数据大小
    """
    script_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\0_data_collection_analysis\SharedGraspNetwork_connection_data_collection.py"
    script_name = "SharedGraspNetwork_connection_data_collection.py"  # 脚本名称
    
    for grasp_id in grasp_ids:
        print(f"处理 grasp_id: {grasp_id}")
        
        # 清理之前的进程
        os.system(f'taskkill /F /FI "WINDOWTITLE eq {script_name}*"')
        time.sleep(2)
        
        # 使用 Popen 创建进程，这样我们可以获取进程ID
        cmd = f'python "{script_path}" --grasp_ids {grasp_id} --total_iterations {total_iterations} --save_batch_size {save_batch_size}'
        process = subprocess.Popen(cmd)
        
        # 设置进程优先级为"高于标准"
        try:
            p = psutil.Process(process.pid)
            p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows下相当于"高于标准"优先级
        except Exception as e:
            print(f"warning : {e}")
        
        # 等待进程完成
        process.wait()
        time.sleep(2)

def parse_args():
    parser = argparse.ArgumentParser(description='运行共享抓取数据收集')
    parser.add_argument('--runs', type=int, default=200, help='运行次数')  # 0.3M  - Shared grasp 0.3M , feasible grasp 0.6M.
    parser.add_argument('--grasp_ids', type=int, nargs='+', default=[57], help='要处理的抓取ID列表')
    parser.add_argument('--total_iterations', type=int, default=int(1e3), help='每次运行的总迭代次数')
    parser.add_argument('--save_batch_size', type=int, default=500, help='每批保存的数据大小')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    for i in range(args.runs):
        print(f"\n{'='*50}")
        print(f"开始第 {i+1}/{args.runs} 次数据收集")
        print(f"当前参数配置:")
        print(f"- 抓取ID: {args.grasp_ids}")
        print(f"- 迭代次数: {args.total_iterations}")
        print(f"- 批量大小: {args.save_batch_size}")
        print(f"{'='*50}")
        
        start_time = time.time()
        run_shared_grasp_data_collection(
            grasp_ids=args.grasp_ids,
            total_iterations=args.total_iterations,
            save_batch_size=args.save_batch_size
        )
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"第 {i+1}/{args.runs} 次运行完成")
        print(f"耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
        print(f"已完成: {((i+1)/args.runs)*100:.1f}%")
        print(f"剩余运行次数: {args.runs-(i+1)}")
        print(f"{'='*50}\n")

