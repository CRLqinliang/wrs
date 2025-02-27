import subprocess
import os
import sys
import argparse
import time

import numpy as np
import psutil  # 需要先 pip install psutil


def run_obstacle_grasp_data_collection():

    script_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\0_data_collection_analysis\ObstacleGraspNetwork_data_collection.py"
    script_name = "ObstacleGraspNetwork_data_collection.py"  # 脚本名称

    # 清理之前的进程
    os.system(f'taskkill /F /FI "WINDOWTITLE eq {script_name}*"')
    time.sleep(2)

    # 使用 Popen 创建进程，这样我们可以获取进程ID
    cmd = f'python "{script_path}"'
    process = subprocess.Popen(cmd)

    # 设置进程优先级为"高于标准"
    try:
        p = psutil.Process(process.pid)
        p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows下相当于"高于标准"优先级
    except Exception as e:
        print(f"设置进程优先级时出错: {e}")

    # 等待进程完成
    process.wait()
    time.sleep(1)


if __name__ == '__main__':
    times = 3e4 / 1e3
    for i in np.arange(times):
        start_time = time.time()
        run_obstacle_grasp_data_collection()
        end_time = time.time()
        duration = end_time - start_time
        print(f"Collect time is {duration}.4f")

