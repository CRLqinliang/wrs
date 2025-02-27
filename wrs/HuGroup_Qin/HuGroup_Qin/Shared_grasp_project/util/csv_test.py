""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241224 Osaka Univ.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_grasp_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Convert percentage string to float (removing '%' and converting to float)
    df['可行抓取比例'] = df['可行抓取比例'].str.rstrip('%').astype(float)
    
    # Group by object name and sample size
    grouped = df.groupby(['物体名称', '抓取样本数'])
    
    # Prepare plotting data
    objects = []
    sample_sizes = []
    means = []
    mins = []
    maxs = []
    
    # Analyze results
    for (name, sample_size), group in grouped:
        print(f"\n物体: {name}, 抓取样本数: {sample_size}")
        
        # Count stable poses
        stable_poses = group['稳定位姿编号'].value_counts()
        print(f"稳定位姿数量: {len(stable_poses)}")
        print("各稳定位姿的样本数量:")
        for pose_id, count in stable_poses.items():
            print(f"位姿 {pose_id}: {count}个样本")
        
        # Calculate feasible grasp ratio statistics
        feasible_ratio = group['可行抓取比例']
        mean_val = feasible_ratio.mean()
        max_val = feasible_ratio.max()
        min_val = feasible_ratio.min()
        
        print("\n可行抓取比例统计:")
        print(f"平均值: {mean_val:.2f}%")
        print(f"最大值: {max_val:.2f}%")
        print(f"最小值: {min_val:.2f}%")
        
        # Store plotting data
        objects.append(name)
        sample_sizes.append(sample_size)
        means.append(mean_val)
        mins.append(min_val)
        maxs.append(max_val)
    
    # Create bar plot
    plt.figure(figsize=(15, 6))
    
    # Get unique objects and sample sizes
    unique_objects = sorted(set(objects))
    unique_samples = sorted(set(sample_sizes))
    
    # Set bar plot parameters
    bar_width = 0.8 / len(unique_samples)
    x = np.arange(len(unique_objects))
    
    # Plot bars for each sample size
    for i, sample_size in enumerate(unique_samples):
        # Get data for this sample size
        indices = [j for j, s in enumerate(sample_sizes) if s == sample_size]
        sample_means = [means[j] for j in indices]
        sample_mins = [mins[j] for j in indices]
        sample_maxs = [maxs[j] for j in indices]
        
        # Calculate bar positions
        bar_positions = x + (i - len(unique_samples)/2 + 0.5) * bar_width
        
        # Plot bars
        plt.bar(bar_positions, sample_means, bar_width, 
                label=f'Samples={sample_size}',
                alpha=0.8)
        
        # Add error bars
        yerr_min = np.array(sample_means) - np.array(sample_mins)
        yerr_max = np.array(sample_maxs) - np.array(sample_means)
        plt.errorbar(bar_positions, sample_means, 
                    yerr=[yerr_min, yerr_max], 
                    fmt='none', 
                    color='black',
                    capsize=5)
    
    # Set plot properties
    plt.xlabel('物体名称')
    plt.ylabel('可行抓取比例 (%)')
    plt.title('不同物体和样本数的可行抓取比例统计')
    plt.xticks(x, unique_objects, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

def analyze_grasp_samples(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert percentage string to float
    df['可行抓取比例'] = df['可行抓取比例'].str.rstrip('%').astype(float)
    
    # Get unique object names and sample sizes
    objects = df['物体名称'].unique()
    sample_sizes = sorted(df['抓取样本数'].unique())
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Set bar plot parameters
    bar_width = 0.15
    x = np.arange(len(objects))
    
    # Plot bars for each sample size
    for i, sample_size in enumerate(sample_sizes):
        means = []
        mins = []
        maxs = []
        
        for obj in objects:
            # Get data for specific object and sample size
            mask = (df['物体名称'] == obj) & (df['抓取样本数'] == sample_size)
            obj_data = df[mask]['可行抓取比例']
            
            means.append(obj_data.mean())
            mins.append(obj_data.min())
            maxs.append(obj_data.max())
        
        # Calculate bar positions
        bar_positions = x + (i - 1.5) * bar_width
        
        # Plot bars
        plt.bar(bar_positions, means, bar_width, 
                label=f'Samples={sample_size}',
                alpha=0.8)
        
        # Add error bars
        yerr_min = np.array(means) - np.array(mins)
        yerr_max = np.array(maxs) - np.array(means)
        plt.errorbar(bar_positions, means, 
                    yerr=[yerr_min, yerr_max], 
                    fmt='none', 
                    color='black',
                    capsize=3)
    
    # Set plot properties
    plt.xlabel('Object name')
    plt.ylabel('Feasible grasp percentage (%)')
    plt.title('Comparison of feasible grasp percentage for different sample sizes')
    plt.xticks(x, objects, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

if __name__ == '__main__':
    file_path = r"H:\Qin\wrs\wrs\HuGroup_Qin\data\grasp_analysis_results.csv"
    analyze_grasp_data(file_path)  # Show first plot
    analyze_grasp_samples(file_path)  # Show second plot
