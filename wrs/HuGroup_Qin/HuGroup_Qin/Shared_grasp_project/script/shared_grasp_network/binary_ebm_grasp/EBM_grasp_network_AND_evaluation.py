""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241205 Osaka Univ.
"""

from typing import List, Tuple, Optional, Union
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
sys.path.append("H:/Qin/wrs")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation as R
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import argparse, tqdm, wandb
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
from wrs.HuGroup_Qin.Shared_grasp_project.script.binary_ebm_grasp.EBM_grasp_network_train import GraspEnergyNetwork
from wrs.HuGroup_Qin.Shared_grasp_project.script.binary_ebm_grasp.EBM_grasp_network_train import MultiScaleGraspEnergyNetwork
from wrs.HuGroup_Qin.Shared_grasp_project.script.binary_ebm_grasp.EBM_grasp_network_train import GraspEnergyDataset
from wrs.HuGroup_Qin.Shared_grasp_project.script.binary_ebm_grasp.EBM_grasp_network_train import split_data_indices, create_datasets, create_data_loaders, load_raw_data

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# 常量定义
OBJECT_MESH_PATH = Path(r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
# base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
# mgm.gen_frame().attach_to(base)


class EBM_grasp_AND_dataset(Dataset):
    def __init__(self, output_dim, pickle_file, grasp_pickle_file, normalize_data=False, 
                       stable_label=True, use_quaternion=True):
        self.output_dim = output_dim
        self.normalize_data = normalize_data
        self.stable_label = stable_label
        self.use_quaternion = use_quaternion

        # 加载抓取候选
        with open(grasp_pickle_file, 'rb') as f:
            self.grasp_candidates = pickle.load(f)

        if pickle_file is None:
            self.position_scaler = StandardScaler()
            return
        
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading file {pickle_file}: {e}")

        # 提取初始位姿和目标位姿
        init_poses = []
        goal_poses = []
    
        for item in data:
            init_pos = np.array(item[0][0]).flatten()
            init_rot = self._convert_rotation(item[0][1])
            init_poses.append(np.concatenate([init_pos, init_rot]))

            goal_pos = np.array(item[1][0]).flatten()
            goal_rot = self._convert_rotation(item[1][1])
            goal_poses.append(np.concatenate([goal_pos, goal_rot]))

        init_poses = np.array(init_poses, dtype=np.float32)
        goal_poses = np.array(goal_poses, dtype=np.float32)
        
        # 分离位置和角度数据
        init_positions = init_poses[:, :3]
        init_angles = init_poses[:, 3:]
        goal_positions = goal_poses[:, :3]
        goal_angles = goal_poses[:, 3:]
        
        # 根据normalize_data参数决定是否进行标准化
        if self.normalize_data:
            # 合并所有位置数据进行标准化
            all_positions = np.vstack([init_positions, goal_positions])
            self.position_scaler = StandardScaler()
            all_positions_scaled = self.position_scaler.fit_transform(all_positions)
            
            # 分别获取标准化后的初始位置和目标位置
            init_positions_scaled = all_positions_scaled[:len(init_positions)]
            goal_positions_scaled = all_positions_scaled[len(init_positions):]
        else:
            # 不进行标准化，直接使用原始数据
            init_positions_scaled = init_positions
            goal_positions_scaled = goal_positions
            self.position_scaler = None

        # 组合所有特征
        self.inputs = np.concatenate([
            init_positions_scaled,
            init_angles,
            goal_positions_scaled,
            goal_angles
        ], axis=1).astype(np.float32)

        # 是否增加稳定标签到输入里面
        if self.stable_label:
            # 获取原始标签并重塑为二维数组
            init_raw_labels = np.array([item[2] for item in data], dtype=int).reshape(-1, 1)
            goal_raw_labels = np.array([item[3] for item in data], dtype=int).reshape(-1, 1)

            # 创建并使用OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            init_stable_label_one_hot = encoder.fit_transform(init_raw_labels)
            goal_stable_label_one_hot = encoder.fit_transform(goal_raw_labels)

            self.inputs =  np.concatenate([init_positions_scaled, init_angles, init_stable_label_one_hot, 
                                          goal_positions_scaled, goal_angles, goal_stable_label_one_hot], axis=1)
            
        # 处理标签
        self.labels = np.array([self._create_target_vector(item[-1]) for item in data], dtype=np.float32)


    def _rotmat_to_euler(self, rotmat):
        """将旋转矩阵转换为欧拉角(rx, ry, rz)"""
        r = R.from_matrix(rotmat)
        return r.as_euler('xyz', degrees=False)
    
    def _convert_rotation(self, rotmat):
        """转换旋转矩阵为四元数或欧拉角"""
        if self.use_quaternion:
            return R.from_matrix(rotmat).as_quat()
        return R.from_matrix(rotmat).as_euler('xyz', degrees=False)

    def _create_target_vector(self, label_ids):
        target_vector = np.zeros(self.output_dim, dtype=np.float32)
        if label_ids == None:
            return target_vector
        target_vector[label_ids] = 1
        return target_vector

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_vector = self.inputs[idx].copy()
        label = self.labels[idx]

        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def save_scaler(self, scaler_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.position_scaler, f)

    @staticmethod
    def load_scaler(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)


def env_setup():
    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.unshow_cdprim()
    return robot


def collect_model_predictions(model_1, model_2, test_loader, grasp_file, device):
    """收集两个能量模型的预测结果"""
    model_1.eval()
    model_2.eval()
    all_labels, all_energies_1, all_energies_2 = [], [], []

    # 加载抓取候选
    with open(grasp_file, 'rb') as f:
        grasp_data = pickle.load(f)
        grasp_list = grasp_data._grasp_list

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="收集预测结果"):
            batch_size = inputs.shape[0]
            
            for batch_idx in range(batch_size):
                # 获取初始物体位姿
                init_obj_pos = inputs[batch_idx, :3]
                init_obj_rot = inputs[batch_idx, 3:7]
                
                # 获取目标物体位姿
                goal_obj_pos = inputs[batch_idx, 12:15]
                goal_obj_rot = inputs[batch_idx, 15:19]

                # 获取初始稳定标签
                init_stable_label_one_hot = inputs[batch_idx, 7:12]
                # 获取目标稳定标签
                goal_stable_label_one_hot = inputs[batch_idx, 12:17]
                
                # 为每个抓取候选构建特征
                model1_features = []
                model2_features = []
                
                for grasp in grasp_list:
                    # 获取抓取位姿
                    grasp_pos = grasp.ac_pos
                    grasp_rot = R.from_matrix(grasp.ac_rotmat).as_quat()  # 将旋转矩阵转换为四元数
                    
                    # 构建模型1的输入 [init_obj_pos, init_obj_rot, grasp_pos, grasp_rot，init_stable_label_one_hot]
                    model1_feature = np.concatenate([
                        init_obj_pos,
                        init_obj_rot,
                        grasp_pos,
                        grasp_rot,
                        init_stable_label_one_hot,
                    ])
                    
                    # 构建模型2的输入 [goal_obj_pos, goal_obj_rot, grasp_pos, grasp_rot，goal_stable_label_one_hot]
                    model2_feature = np.concatenate([
                        goal_obj_pos,
                        goal_obj_rot,
                        grasp_pos,
                        grasp_rot,
                        goal_stable_label_one_hot
                    ])
                    
                    model1_features.append(model1_feature)
                    model2_features.append(model2_feature)
                
                # 转换为tensor并移到设备
                model1_features = torch.FloatTensor(model1_features).to(device)
                model2_features = torch.FloatTensor(model2_features).to(device)
                
                # 计算能量
                energies_1 = -model_1(model1_features)  # 取负值，使得能量值越低表示越可能
                energies_2 = -model_2(model2_features)
                
                # 收集结果
                all_labels.append(labels[batch_idx].cpu().numpy())
                all_energies_1.append(energies_1.cpu().numpy())
                all_energies_2.append(energies_2.cpu().numpy())

    return (np.array(all_labels), 
            np.array(all_energies_1), 
            np.array(all_energies_2))

def evaluate_threshold_combination(all_labels, all_logits_1, all_logits_2, th1, th2):
    """评估两个模型在给定阈值下的组合性能
    
    Args:
        all_labels: shape [N, num_classes] 真实标签 (0或1)
        all_logits_1: shape [N, num_classes] 模型1的预测能量
        all_logits_2: shape [N, num_classes] 模型2的预测能量
        th1: 模型1的阈值
        th2: 模型2的阈值
    """
    predictions_1 = (all_logits_1 < th1).astype(float)
    predictions_2 = (all_logits_2 < th2).astype(float)
    predictions = np.logical_and(predictions_1, predictions_2).astype(float)

    flat_labels = all_labels.flatten()
    flat_predictions = predictions.flatten()

    return {
        "threshold_1": th1,
        "threshold_2": th2,
        "accuracy": accuracy_score(flat_labels, flat_predictions) * 100,
        "precision_macro": precision_score(flat_labels, flat_predictions, zero_division=0),
        "recall_macro": recall_score(flat_labels, flat_predictions, zero_division=0),
        "f1_macro": f1_score(flat_labels, flat_predictions, zero_division=0)
    }

def plot_heatmap(matrix, x_labels, y_labels, title, filename):
    """绘制并保存热力图"""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels, 
                     annot=False, fmt=".1f",
                     cmap="viridis",
                     annot_kws={"color": "black", "fontsize": 10})
    plt.title(title, fontsize=14)
    plt.xlabel("Model robot-table Threshold", fontsize=14)
    plt.ylabel("Model robot-table Threshold", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 手动设置颜色条的字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    # 调整子图的布局，确保所有元素都能显示
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def log_results_to_wandb(results_data, precision_matrix, recall_matrix, 
                        thresholds_1, thresholds_2, best_thresholds, best_f1):
    """记录结果到wandb并在本地绘制热力图"""
    try:
        # 创建热力图数据表
        # wandb.log({
        #     "Dataset Threshold Evaluation": wandb.Table(
        #         data=results_data,
        #         columns=["Threshold_1", "Threshold_2", "Accuracy", "Hamming_Loss", 
        #                 "Precision_Macro", "Recall_Macro", "F1_Macro"]
        #     ),
        #     "Dataset Best Thresholds": best_thresholds,
        #     "Dataset Best F1 Score": best_f1
        # })

        # 在本地绘制热力图
        plot_heatmap(precision_matrix, [f"{x:.2f}" for x in thresholds_1], [f"{x:.2f}" for x in thresholds_2],
                     "Precision Heatmap", "precision_heatmap.png")
        plot_heatmap(recall_matrix, [f"{x:.2f}" for x in thresholds_1], [f"{x:.2f}" for x in thresholds_2],
                     "Recall Heatmap", "recall_heatmap.png")

    except Exception as e:
        print(f"Warning: Failed to log to wandb: {str(e)}")

def test_AND_model_with_dataset(args, model_1, model_2, test_loader, grasp_file, device,
                            thresholds_1=np.arange(0.3, 0.9, 0.05),
                            thresholds_2=np.arange(0.3, 0.9, 0.05),
                            verbose=True):
    # 收集模型预测
    all_labels, all_energies_1, all_energies_2 = collect_model_predictions(
        model_1, model_2, test_loader, grasp_file,device
    )

    # 初始化结果存储
    threshold_results = {}
    best_f1 = -1
    best_thresholds = None
    best_results = None
    results_data = []
    
    # 创建性能矩阵
    precision_matrix = np.zeros((len(thresholds_1), len(thresholds_2)))
    recall_matrix = np.zeros((len(thresholds_1), len(thresholds_2)))

    # 测试不同阈值组合
    print("\n测试不同阈值组合的性能:") if verbose else None
    for i, th1 in enumerate(tqdm.tqdm(thresholds_1, desc="测试模型1阈值")):
        for j, th2 in enumerate(thresholds_2):
            # 评估当前阈值组合
            results = evaluate_threshold_combination(
                all_labels, all_energies_1, all_energies_2, th1, th2
            )
            
            # 更新结果
            results_data.append([th1, th2] + [results[k] for k in results.keys() 
                                            if k not in ['threshold_1', 'threshold_2']])
            threshold_results[(th1, th2)] = results
            precision_matrix[i, j] = results["precision_macro"]
            recall_matrix[i, j] = results["recall_macro"]

            # 更新最佳结果
            if results["f1_macro"] > best_f1:
                best_f1 = results["f1_macro"]
                best_thresholds = (th1, th2)
                best_results = results

            # 输出当前结果
            if verbose:
                print(f"\nThreshold_1 = {th1:.2f}, Threshold_2 = {th2:.2f}:")
                print(f"Accuracy: {results['accuracy']:.2f}%")
                print(f"Macro - P: {results['precision_macro']:.4f}, "
                      f"R: {results['recall_macro']:.4f}, "
                      f"F1: {results['f1_macro']:.4f}")

    # 记录结果到wandb
    log_results_to_wandb(results_data, precision_matrix, recall_matrix,
                        thresholds_1, thresholds_2, best_thresholds, best_f1)

    return best_results, threshold_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    parser.add_argument('--model_path', type=str, 
                        default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\Binary_ebm_model\best_model_grasp_ebm_v2_robot_table_withstablelabel_352_h5_4096_lr0.0005_t0.15_pos_ratio0.1_without_reg_rotstandard.pth',
                        help='模型路径')
    parser.add_argument('--test_dataset', type=str, 
                        default=r"H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\shared_grasp_stable_label_random_position_bottle_109.pickle",
                        help='测试数据集路径')
    parser.add_argument('--grasp_pickle_file', type=str, 
                        default=r"H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle",
                        help='抓取候选路径')
    parser.add_argument('--shared_testdataset', type=bool,  default=True, help='是否使用共享测试数据集')
    parser.add_argument('--input_dim', type=int, default=6, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=109, help='输出维度')
    parser.add_argument('--hidden_dims', type=list, default=[128, 256, 512, 256, 128], help='隐藏层维度')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='残差块数量')
    parser.add_argument('--dropout_rate', type=float, default=0.15, help='dropout率')

    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--train_split', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--normalize_data', type=bool, default=False, help='是否进行数据标准化')
    parser.add_argument('--stable_label', type=bool, default=True, help='是否使用稳定标签')
    parser.add_argument('--use_quaternion', type=bool, default=True, help='是否使用四元数')
    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='grasp_ebm_experiments', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str,
                        default='grasp_ebm_AND_evaluation_bottle_352',
                        help='wandb运行名称')

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载数据
    full_dataset = EBM_grasp_AND_dataset(
        args.output_dim, 
        args.test_dataset,
        args.grasp_pickle_file,
        normalize_data=args.normalize_data,
        stable_label=args.stable_label,
        use_quaternion=args.use_quaternion
    )

    # 计算数据集分割
    train_size = int(args.train_split * len(full_dataset))
    val_size = int(args.val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 设置输入维度 考虑one-hot编码维度
    n_categories = 5
    input_dim = (14 if args.use_quaternion else 12) + n_categories  # 物体位姿 + 抓取位姿 + one-hot

    # 已有数据集测试
    model_1 = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                            num_res_blocks=args.num_res_blocks, 
                                            dropout_rate=args.dropout_rate)
    model_2 = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                            num_res_blocks=args.num_res_blocks,
                                            dropout_rate=args.dropout_rate)
    
    checkpoint = torch.load(args.model_path)
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_2.load_state_dict(checkpoint['model_state_dict'])


    # 可以为两个模型设置不同的能量判断阈值
    thresholds_1 = np.arange(-0.1, 0.3, 0.02)
    thresholds_2 = np.arange(-0.1, 0.3, 0.02)

    # wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name)

    # 使用数据集测试模型
    print("\n使用数据集进行测试...")
    dataset_results = test_AND_model_with_dataset(
        args, model_1, model_2, train_loader, args.grasp_pickle_file,
        device='cpu',
        thresholds_1=thresholds_1,
        thresholds_2=thresholds_2
    )

