""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241209 Osaka Univ.

"""

import pickle
import sys
sys.path.append("E:/Qin/wrs")
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
import numpy as np
import matplotlib.pyplot as plt

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)


def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def show_grasp(gripper, grasp_collection):
    obj_cmodel = mcm.CollisionModel(initor=r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bracketR1.stl")
    obj_cmodel.attach_to(base)
    for grasp in grasp_collection:
        gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                                jaw_width=grasp.ee_values)
        gripper.gen_meshmodel(alpha=.1).attach_to(base)
    base.run()


def create_target_vector(label_dim, label_ids):
    target_vector = np.zeros(label_dim, dtype=np.float32)
    if label_ids == None:
        return target_vector
    target_vector[label_ids] = 1
    return target_vector


def analyze_ones_distribution(arrays):
    """
    Analyze the distribution of 1s in a 0-1 array.
    :param arrays: 2D numpy array, each row is a 0-1 array.
    :return: Dictionary of analysis results.
    """
    arrays = np.array(arrays)  # Ensure input is a numpy array
    total_arrays, array_length = arrays.shape

    # Overall statistics
    total_ones = np.sum(arrays)  # Total number of 1s in all arrays
    ones_proportion = total_ones / (total_arrays * array_length)  # Proportion of 1s in the overall array

    # Number of 1s per row
    ones_per_row = np.sum(arrays, axis=1)  # Number of 1s in each array
    average_ones_per_row = np.mean(ones_per_row)  # Average number of 1s per row
    median_ones_per_row = np.median(ones_per_row)  # Median number of 1s per row

    # Distribution of 1s per column
    ones_per_column = np.sum(arrays, axis=0)  # Number of 1s in each column
    column_ones_proportion = ones_per_column / total_arrays  # Proportion of 1s in each column

    # Distribution plot
    plt.figure(figsize=(10, 6))

    plt.bar(range(array_length), column_ones_proportion)
    plt.title('Proportion of 1s Per Column')
    plt.xlabel('Column Index')
    plt.ylabel('Proportion of 1s')

    plt.tight_layout()
    plt.show()

    # Return analysis results
    return {
        "total_ones": total_ones,
        "ones_proportion": ones_proportion,
        "average_ones_per_row": average_ones_per_row,
        "median_ones_per_row": median_ones_per_row,
        "ones_per_column": ones_per_column.tolist(),
        "column_ones_proportion": column_ones_proportion.tolist()
    }


def compare_multiple_distributions(file_numbers):
    """
    Compare distributions across multiple files and generate comparison charts
    :param file_numbers: list of file numbers [42, 57, 83, 109]
    """
    all_results = {}
    
    for num in file_numbers:
        # Load data
        common_grasp_data_path = f"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_{num}.pickle"
        common_grasp_data = grasp_load(common_grasp_data_path)
        # Create target vectors
        target_vector_set = []
        for item in common_grasp_data:
            target_vector_set.append(create_target_vector(num, item[-1]))
            
        # Analyze distribution
        result = analyze_ones_distribution(target_vector_set)
        all_results[f'Grasp_{num}'] = result
    
    # Create comparison charts
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Total count and proportion comparison with dual y-axes
    ax1 = plt.subplot(1, 2, 1)
    x = np.arange(len(file_numbers))
    
    total_ones = [all_results[f'Grasp_{num}']['total_ones'] for num in file_numbers]
    proportions = [all_results[f'Grasp_{num}']['ones_proportion'] for num in file_numbers]
    
    # First y-axis for total count
    color1 = 'tab:blue'
    ax1.set_xlabel('Grasp Candidate Number')
    ax1.set_ylabel('Total Count of positive samples', color=color1)
    bars1 = ax1.bar(x - 0.15, total_ones, 0.3, label='Total Count of positive samples', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Second y-axis for proportion
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Overall Proportion of positive samples', color=color2)
    bars2 = ax2.bar(x + 0.15, proportions, 0.3, label='Overall Proportion of positive samples', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set x-ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_numbers)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Comparison of Total Count and Proportion of Positive Samples Across All Datasets')
    
    # Subplot 2: Distribution comparison with separate bar groups
    ax3 = plt.subplot(1, 2, 2)
    
    # Create separate plots for each file
    for i, num in enumerate(file_numbers):
        proportions = all_results[f'Grasp_{num}']['column_ones_proportion']
        x_indices = np.arange(len(proportions))
        ax3.bar(x_indices, proportions, alpha=0.3, label=f'Grasp_{num}')
    
    ax3.set_xlabel('Grasp Candidate Index')
    ax3.set_ylabel('Proportion of positive samples')
    ax3.set_title('Distribution Comparison of positive samples Across All Datasets')
    ax3.legend()
    
    # Only show some x-ticks to avoid overcrowding
    max_ticks = 10
    step = max(len(x_indices) // max_ticks, 1)
    ax3.set_xticks(x_indices[::step])
    
    plt.tight_layout()
    plt.show()

# TODO modify the detail.
if __name__ == '__main__':
    # grasp data
    # grasp_data_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_352.pickle"
    # object_feasible_grasps = grasp_load(grasp_data_path)
    # id_dim = len(object_feasible_grasps)

    # common grasp data
    common_grasp_data_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Mug\SharedGraspNetwork_mug_experiment_data_final_195.pickle"
    common_grasp_data = grasp_load(common_grasp_data_path)


    # create target vectors
    target_vector_set = []
    for common_grasp_data_index in range(len(common_grasp_data)):
        common_grasp_data_item = common_grasp_data[common_grasp_data_index]
        target_vector_set.append(create_target_vector(id_dim, common_grasp_data_item[-1]))

    result = analyze_ones_distribution(target_vector_set)

    # # 查看分析结果
    print("总 1 的数量：", result["total_ones"])
    # print("1 的总体比例：", result["ones_proportion"])
    # print("每行平均 1 的数量：", result["average_ones_per_row"])
    # print("每行 1 的中位数：", result["median_ones_per_row"])
    # print("每列 1 的数量：", result["ones_per_column"])
    # print("每列 1 的比例：", result["column_ones_proportion"])
    
    # # 添加多文件对比分析
    # file_numbers = [57, 83, 109]
    # compare_multiple_distributions(file_numbers)