import matplotlib.pyplot as plt
import ast
import numpy as np

# 定义读取数据的函数
def read_data(filename):
    fx, fy, fz, tx, ty, tz = [], [], [], [], [], []
    joint_6 = []
    with open(filename, 'r') as file:
        for line in file:
            data = ast.literal_eval(line.strip())
            # 提取力和力矩数据
            fx.append(data[0][0])
            fy.append(data[0][1])
            fz.append(data[0][2])
            tx.append(data[0][3])
            ty.append(data[0][4])
            tz.append(data[0][5])
            # 提取机器人第六个关节数据
            joint_6.append(data[1][5])
    return fx, fy, fz, tx, ty, tz, joint_6

# 定义 u-sigma 图的绘图函数
def plot_u_sigma_mean_std(filenames, label, color, adjust_joint_6=False):
    all_joint_6 = []
    all_tz = []

    # 读取每个文件的数据并收集
    for filename in filenames:
        _, _, _, _, _, tz, joint_6 = read_data(filename)
        # 对 Cap1-3 和 Cap1-4 的 joint_6 加上 180°（如果需要）
        if adjust_joint_6 and filename in ['Cap1-3.txt', 'Cap1-4.txt']:
            joint_6 = [x + 180 for x in joint_6]
        all_joint_6.extend(joint_6)
        all_tz.extend(tz)

    # 将所有数据转换为 numpy 数组
    all_joint_6 = np.array(all_joint_6)
    all_tz = np.array(all_tz)

    # 对 joint_6 进行排序，并根据排序索引对 tz 进行相应排序
    sorted_indices = np.argsort(all_joint_6)
    all_joint_6 = all_joint_6[sorted_indices]
    all_tz = all_tz[sorted_indices]

    # 计算均值和标准差（每个唯一 joint_6 值对应的 tz）
    unique_joint_6 = np.unique(all_joint_6)
    mean_tz = []
    std_tz = []
    for u in unique_joint_6:
        tz_values = all_tz[all_joint_6 == u]
        mean_tz.append(np.mean(tz_values))
        std_tz.append(np.std(tz_values))

    mean_tz = np.array(mean_tz)
    std_tz = np.array(std_tz)

    # 使用滑动窗口对均值进行平滑（窗口大小为 10）
    window_size = 10
    mean_tz_smooth = np.convolve(mean_tz, np.ones(window_size)/window_size, mode='valid')
    unique_joint_6_smooth = unique_joint_6[:len(mean_tz_smooth)]

    # 绘制均值和标准差图像
    plt.plot(unique_joint_6_smooth, mean_tz_smooth, label=f'Smoothed Mean tz - {label}', color=color)
    plt.fill_between(unique_joint_6_smooth, mean_tz_smooth - std_tz[:len(mean_tz_smooth)], mean_tz_smooth + std_tz[:len(mean_tz_smooth)], color=color, alpha=0.2)

# 主函数
def main():
    # Cap1 数据
    filenames_cap1 = ['Cap1-1.txt', 'Cap1-2.txt', 'Cap1-3.txt', 'Cap1-4.txt']
    # Cap2 数据
    filenames_cap2 = ['Cap2-1.txt', 'Cap2-2.txt', 'Cap2-3.txt', 'Cap2-4.txt']
    # Cap3 数据
    filenames_cap3 = ['Cap3-1.txt', 'Cap3-2.txt', 'Cap3-3.txt', 'Cap3-4.txt']
    # Cap4 数据
    filenames_cap4 = ['Cap4-1.txt', 'Cap4-2.txt', 'Cap4-3.txt', 'Cap4-4.txt']
    # Cap5 数据
    filenames_cap5 = ['Cap5-1.txt', 'Cap5-2.txt', 'Cap5-3.txt', 'Cap5-4.txt']
    # Cap6 数据
    filenames_cap6 = ['Cap6-1.txt', 'Cap6-2.txt', 'Cap6-3.txt']
    # Cap7 数据
    filenames_cap7 = ['Cap7-1.txt', 'Cap7-2.txt', 'Cap7-3.txt', 'Cap7-4.txt']

    # 绘制所有 Cap 的均值和标准差图像在同一张图上
    plt.figure(figsize=(12, 8))

    # 计算 Cap1 的均值和标准差图所对应的 joint_6 减去 180 后的曲线
    all_joint_6 = []
    all_tz = []
    for filename in filenames_cap1:
        _, _, _, _, _, tz, joint_6 = read_data(filename)
        if filename in ['Cap1-3.txt', 'Cap1-4.txt']:
            joint_6 = [x + 180 for x in joint_6]
        all_joint_6.extend(joint_6)
        all_tz.extend(tz)

    all_joint_6 = np.array(all_joint_6) - 180
    all_tz = np.array(all_tz)

    sorted_indices = np.argsort(all_joint_6)
    all_joint_6 = all_joint_6[sorted_indices]
    all_tz = all_tz[sorted_indices]

    unique_joint_6 = np.unique(all_joint_6)
    mean_tz = []
    std_tz = []
    for u in unique_joint_6:
        tz_values = all_tz[all_joint_6 == u]
        mean_tz.append(np.mean(tz_values))
        std_tz.append(np.std(tz_values))

    mean_tz = np.array(mean_tz)
    std_tz = np.array(std_tz)

    window_size = 10
    mean_tz_smooth = np.convolve(mean_tz, np.ones(window_size)/window_size, mode='valid')
    unique_joint_6_smooth = unique_joint_6[:len(mean_tz_smooth)]

    plt.plot(unique_joint_6_smooth, mean_tz_smooth, label='Smoothed Mean tz - Cap1', color='b', linestyle='-')
    plt.fill_between(unique_joint_6_smooth, mean_tz_smooth - std_tz[:len(mean_tz_smooth)], mean_tz_smooth + std_tz[:len(mean_tz_smooth)], color='b', alpha=0.2)

    plot_u_sigma_mean_std(filenames_cap2, label='Cap2', color='r', adjust_joint_6=False)
    plot_u_sigma_mean_std(filenames_cap3, label='Cap3', color='g', adjust_joint_6=False)
    plot_u_sigma_mean_std(filenames_cap4, label='Cap4', color='orange', adjust_joint_6=False)
    plot_u_sigma_mean_std(filenames_cap5, label='Cap5', color='purple', adjust_joint_6=False)

    plt.xlabel('u (Joint 6 Position)', fontsize=16)
    plt.ylabel('sigma (Torque tz)', fontsize=16)
    plt.title('u-sigma Plot with Smoothed Mean and Standard Deviation for Cap1 to Cap5', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 绘制 Cap6 的均值和标准差图像
    plt.figure(figsize=(12, 8))
    plot_u_sigma_mean_std(filenames_cap7, label='Cap7', color='b', adjust_joint_6=False)

    plt.xlabel('u (Joint 6 Position)', fontsize=16)
    plt.ylabel('sigma (Torque tz)', fontsize=16)
    plt.title('u-sigma Plot with Smoothed Mean and Standard Deviation for Cap7', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
