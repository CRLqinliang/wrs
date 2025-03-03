""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241205 Osaka Univ.
"""

from typing import List, Tuple, Optional, Union
import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
sys.path.append("E:/Qin/wrs")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.transform import Rotation as R
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import argparse, tqdm, wandb
from wrs.HuGroup_Qin.Shared_grasp_project.script.shared_grasp_network.binary_ebm_grasp.EBM_grasp_network_train import GraspEnergyNetwork

from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import random
import os

class SharedGraspEnergyDataset(Dataset):
    def __init__(self, data, grasp_pickle_file, use_quaternion=False, use_stable_label=True, 
                 grasp_type='robot_table', state_type='both'):
        """
        Args:
            data: 数据列表，每个item包含 
                  [[init_pos, init_rotmat], init_available_gids_robot_table, init_available_gids_table, init_available_gids_robot_without_table, init_stable_id,
                   [goal_pos, goal_rotmat], goal_available_gids_robot_table, goal_available_gids_table, goal_available_gids_robot_without_table, goal_stable_id,
                   common_id]
            grasp_pickle_file: 抓取候选文件路径
            use_quaternion: 是否使用四元数表示旋转。如果为False，则使用简化的(x,y,rz)表示
            use_stable_label: 是否使用stable_label作为特征。注意：当use_quaternion=False时必须为True
            grasp_type: 使用哪种抓取集合，可选 'robot_table', 'table', 'robot'
            state_type: 处理哪种状态，可选 'init', 'goal', 'both'
        """
        # 参数验证
        if not use_quaternion and not use_stable_label:
            raise ValueError("非四元数表示(use_quaternion=False)必须使用stable_label")
            
        self.use_quaternion = use_quaternion
        self.use_stable_label = use_stable_label
        self.data = data
        self.grasp_type = grasp_type
        self.state_type = state_type
        
        # 检查参数有效性
        valid_grasp_types = ['robot_table', 'table', 'robot']
        valid_state_types = ['init', 'goal', 'both']
        
        if grasp_type not in valid_grasp_types:
            raise ValueError(f"grasp_type 必须是 {valid_grasp_types} 之一")
        if state_type not in valid_state_types:
            raise ValueError(f"state_type 必须是 {valid_state_types} 之一")
            
        # 加载抓取候选
        with open(grasp_pickle_file, 'rb') as f:
            grasp_candidates = pickle.load(f)
            
        # 预处理抓取位姿数据 - 始终使用7维表示(pos + quaternion)
        grasp_poses = np.array([
            np.concatenate([
                np.array(grasp.ac_pos, dtype=np.float32).flatten(),
                R.from_matrix(grasp.ac_rotmat).as_quat()
            ]) for grasp in grasp_candidates
        ], dtype=np.float32)
        self.grasp_poses = torch.from_numpy(grasp_poses.copy())
        del grasp_poses, grasp_candidates
        
        # 只有在使用stable_label时才创建编码器
        if self.use_stable_label:
            # 创建物体stable_id的One-Hot编码器
            init_types = [item[4] for item in data]  # init_stable_id
            goal_types = [item[9] for item in data]  # goal_stable_id
            all_types = list(set(init_types + goal_types))
            self.obj_encoder = OneHotEncoder(sparse_output=False)
            self.obj_encoder.fit(np.array(all_types).reshape(-1, 1))
            del init_types, goal_types, all_types
        
        self.prepare_data()
    
    def _normalize_angle(self, angle):
        """将角度从[-π, π]归一化到[0, 1]范围"""
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return (angle + np.pi) / (2 * np.pi)
    
    def _convert_rotation(self, rotmat):
        """转换旋转矩阵为四元数或归一化的欧拉角"""
        if self.use_quaternion:
            return R.from_matrix(rotmat).as_quat()
        else:
            # 如果不使用四元数，只返回z轴旋转角度并归一化
            euler = R.from_matrix(rotmat).as_euler('zxy', degrees=False)
            return np.array([self._normalize_angle(euler[0])], dtype=np.float32)
    
    def _get_position(self, pos):
        """根据表示方式返回位置信息"""
        if self.use_quaternion:
            return pos.copy()  # 返回完整的xyz
        else:
            return pos[:2].copy()  # 只返回xy
        
    def _get_grasp_indices(self, item, state_type):
        """获取指定状态和抓取类型的抓取索引列表"""
        if state_type == 'init':
            if self.grasp_type == 'robot_table':
                return item[1]  # init_available_gids_robot_table
            elif self.grasp_type == 'table':
                return item[2]  # init_available_gids_table
            elif self.grasp_type == 'robot':
                return item[3]  # init_available_gids_robot_without_table
        else:  # state_type == 'goal'
            if self.grasp_type == 'robot_table':
                return item[6]  # goal_available_gids_robot_table
            elif self.grasp_type == 'table':
                return item[7]  # goal_available_gids_table
            elif self.grasp_type == 'robot':
                return item[8]  # goal_available_gids_robot_without_table
        return None
        
    def prepare_data(self):
        """准备训练数据"""
        # 计算特征维度
        if self.use_quaternion:
            pose_dim = 7  # pos(3) + quaternion(4)
        else:
            pose_dim = 3  # pos(2) + normalized_rz(1)
            
        grasp_pose_dim = 7  # 抓取位姿始终使用7维表示
        
        # 根据是否使用stable_label确定特征维度
        if self.use_stable_label:
            onehot_dim = len(self.obj_encoder.get_feature_names_out())
            single_state_dim = pose_dim + onehot_dim
        else:
            single_state_dim = pose_dim
            onehot_dim = 0
        
        # 处理state_type为'both'的情况时，特征维度需要包含两个状态
        if self.state_type == 'both':
            feature_dim = 2 * single_state_dim + grasp_pose_dim  # 初始状态+目标状态+抓取位姿
        else:
            feature_dim = single_state_dim + grasp_pose_dim  # 单状态+抓取位姿
        
        print(f"特征维度: {feature_dim}")
        
        # 计算总样本数
        total_samples = len(self.data) * len(self.grasp_poses)
        print(f"总样本数: {total_samples}")
        
        # 预分配数组
        all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)
        all_labels = np.zeros(total_samples, dtype=np.float32)
        
        # 批量处理数据
        current_idx = 0
        
        # 根据state_type的不同处理方式
        if self.state_type == 'both':
            # 同时处理初始和目标状态
            for item_idx, item in enumerate(self.data):
                # 获取物体初始状态位姿和稳定性ID
                init_obj_pos = self._get_position(np.array(item[0][0], dtype=np.float32))
                init_obj_rot = self._convert_rotation(np.array(item[0][1], dtype=np.float32))
                init_stable_id = item[4]
                init_obj_pose = np.concatenate([init_obj_pos, init_obj_rot])
                
                # 获取物体目标状态位姿和稳定性ID
                goal_obj_pos = self._get_position(np.array(item[5][0], dtype=np.float32))
                goal_obj_rot = self._convert_rotation(np.array(item[5][1], dtype=np.float32))
                goal_stable_id = item[9]
                goal_obj_pose = np.concatenate([goal_obj_pos, goal_obj_rot])
                
                # 从公共ID获取共同有效的抓取索引
                common_grasp_indices = item[-1]
                
                # 为该物体创建所有抓取的样本
                n_samples = len(self.grasp_poses)
                end_idx = current_idx + n_samples
                
                # 填充特征数组
                feature_start = 0
                
                # 添加初始状态物体位姿
                all_features[current_idx:end_idx, feature_start:feature_start+len(init_obj_pose)] = init_obj_pose
                feature_start += len(init_obj_pose)
                
                # 如果使用stable_label，添加初始状态的One-Hot编码
                if self.use_stable_label:
                    init_stable_onehot = self.obj_encoder.transform([[init_stable_id]]).copy()
                    all_features[current_idx:end_idx, feature_start:feature_start+onehot_dim] = init_stable_onehot
                    feature_start += onehot_dim
                    
                # 添加目标状态物体位姿
                all_features[current_idx:end_idx, feature_start:feature_start+len(goal_obj_pose)] = goal_obj_pose
                feature_start += len(goal_obj_pose)
                
                # 如果使用stable_label，添加目标状态的One-Hot编码
                if self.use_stable_label:
                    goal_stable_onehot = self.obj_encoder.transform([[goal_stable_id]]).copy()
                    all_features[current_idx:end_idx, feature_start:feature_start+onehot_dim] = goal_stable_onehot
                    feature_start += onehot_dim
                
                # 添加抓取位姿
                all_features[current_idx:end_idx, feature_start:] = self.grasp_poses.numpy()
                
                # 设置标签 - 共同有效的抓取标记为1，其余为0
                if common_grasp_indices:
                    all_labels[current_idx:end_idx][common_grasp_indices] = 1
                
                current_idx = end_idx
        else:
            # 原有逻辑 - 处理单一状态
            for item_idx, item in enumerate(self.data):
                state = 'init' if self.state_type == 'init' else 'goal'
                
                # 获取物体位姿和稳定性ID
                if state == 'init':
                    obj_pos = self._get_position(np.array(item[0][0], dtype=np.float32))
                    obj_rot = self._convert_rotation(np.array(item[0][1], dtype=np.float32))
                    stable_id = item[4]
                else:  # state == 'goal'
                    obj_pos = self._get_position(np.array(item[5][0], dtype=np.float32))
                    obj_rot = self._convert_rotation(np.array(item[5][1], dtype=np.float32))
                    stable_id = item[9]
                
                obj_pose = np.concatenate([obj_pos, obj_rot])
                
                # 获取该状态下的有效抓取索引
                grasp_indices = self._get_grasp_indices(item, state)
                if grasp_indices is None:
                    grasp_indices = []
                
                # 为该物体状态创建所有抓取的样本
                n_samples = len(self.grasp_poses)
                end_idx = current_idx + n_samples
                
                # 填充特征数组
                feature_start = 0
                
                # 添加物体位姿
                all_features[current_idx:end_idx, feature_start:feature_start+len(obj_pose)] = obj_pose
                feature_start += len(obj_pose)
                
                # 如果使用stable_label，添加One-Hot编码
                if self.use_stable_label:
                    stable_onehot = self.obj_encoder.transform([[stable_id]]).copy()
                    all_features[current_idx:end_idx, feature_start:feature_start+onehot_dim] = stable_onehot
                    feature_start += onehot_dim
                
                # 添加抓取位姿
                all_features[current_idx:end_idx, feature_start:] = self.grasp_poses.numpy()
                
                # 设置标签
                # 将有效抓取的标签设为1，其余为0
                if grasp_indices:
                    all_labels[current_idx:end_idx][grasp_indices] = 1
                
                current_idx = end_idx
        
        # 转换为tensor并确保数据独立
        self.all_features = torch.from_numpy(all_features.copy())
        self.all_labels = torch.from_numpy(all_labels.copy())
        
        # 打印数据集信息
        positive_count = torch.sum(self.all_labels == 1).item()
        print(f"总样本: {len(self.all_labels)}, 正样本: {positive_count}, 正样本比例: {positive_count/len(self.all_labels):.3f}")
        
        # 清理中间变量
        del all_features, all_labels
        gc.collect()
    
    def __len__(self):
        return len(self.all_features)
    
    def __getitem__(self, idx):
        return self.all_features[idx], self.all_labels[idx]
    
    
def collect_model_predictions_with_mode(models, test_loader, grasp_file, device, mode="standard"):
    """收集能量模型的预测结果，支持不同模式
    
    Args:
        models: 模型字典，包含不同模式需要的模型
            - 标准模式(standard): {'init': model1, 'goal': model2}
            - 扩展模式(extended): {'robot_init': model1, 'table_init': model2, 'robot_goal': model3, 'table_goal': model4}
        test_loader: 测试数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        mode: 评估模式，可以是 "standard"(标准，两个模型) 或 "extended"(扩展，四个模型)
        
    注意:
        此函数实现了能量输出的标准化。每个模型的能量输出将进行Z-Score标准化处理：
        normalized_energy = (energy - mean(energy)) / std(energy)
        这确保了来自不同模型的能量贡献在相加前具有可比性，防止某个模型主导最终决策。
    """
    # 将所有模型设置为评估模式
    for model in models.values():
        model.eval()
    
    all_labels = []
    all_energies_init = []  # init状态的能量
    all_energies_goal = []  # goal状态的能量
    all_combined_energies = []  # 全部能量的组合

    # 收集所有批次的能量值
    all_init_energies = []
    all_goal_energies = []
    
    # 扩展模式需要的列表
    if mode == "extended":
        all_robot_init_energies = []
        all_table_init_energies = []
        all_robot_goal_energies = []
        all_table_goal_energies = []

    # 第一次循环：收集所有能量值
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.cpu().numpy()
            
            # 判断是否使用quaternion表示
            feature_dim = inputs.shape[1]
            is_quaternion = (feature_dim > 22)
            
            # 确定各部分的维度
            obj_pose_dim = 7 if is_quaternion else 3
            stable_dim = 5  # 假设stable label的one-hot编码维度为5
            grasp_pose_dim = 7  # 抓取位姿始终为7维
            
            # 计算各部分在特征中的位置
            init_pose_end = obj_pose_dim
            init_stable_end = init_pose_end + stable_dim
            goal_pose_end = init_stable_end + obj_pose_dim
            goal_stable_end = goal_pose_end + stable_dim
            
            # 提取各个部分的特征
            init_obj_pose = inputs[:, :init_pose_end]
            init_stable = inputs[:, init_pose_end:init_stable_end]
            goal_obj_pose = inputs[:, init_stable_end:goal_pose_end]
            goal_stable = inputs[:, goal_pose_end:goal_stable_end]
            grasp_pose = inputs[:, goal_stable_end:]
            
            # 构建初始状态和目标状态的特征输入
            init_features = torch.cat([init_obj_pose, init_stable, grasp_pose], dim=1)
            goal_features = torch.cat([goal_obj_pose, goal_stable, grasp_pose], dim=1)
            
            # 根据不同模式计算能量并存储原始输出
            if mode == "standard":
                # 标准模式：使用两个模型分别预测init和goal状态
                init_energies = models['init'](init_features).cpu().numpy()
                goal_energies = models['goal'](goal_features).cpu().numpy()
                
                all_init_energies.append(init_energies)
                all_goal_energies.append(goal_energies)
                
            elif mode == "extended":
                # 扩展模式：使用四个模型预测
                robot_init_energies = models['robot_init'](init_features).cpu().numpy()
                table_init_energies = models['table_init'](init_features).cpu().numpy()
                robot_goal_energies = models['robot_goal'](goal_features).cpu().numpy()
                table_goal_energies = models['table_goal'](goal_features).cpu().numpy()
                
                all_robot_init_energies.append(robot_init_energies)
                all_table_init_energies.append(table_init_energies)
                all_robot_goal_energies.append(robot_goal_energies)
                all_table_goal_energies.append(table_goal_energies)
            
            all_labels.append(labels)

    # 合并所有批次的能量值
    all_labels = np.concatenate(all_labels)
    
    if mode == "standard":
        # 标准模式处理
        all_init_energies = np.concatenate(all_init_energies)
        all_goal_energies = np.concatenate(all_goal_energies)

        # 全局标准化
        init_energies_norm = (all_init_energies - np.mean(all_init_energies)) / (np.std(all_init_energies) + 1e-8)
        goal_energies_norm = (all_goal_energies - np.mean(all_goal_energies)) / (np.std(all_goal_energies) + 1e-8)

        # 合并结果
        all_energies_init = init_energies_norm
        all_energies_goal = goal_energies_norm
        all_combined_energies = init_energies_norm + goal_energies_norm
        
    elif mode == "extended":
        # 扩展模式处理
        all_robot_init_energies = np.concatenate(all_robot_init_energies)
        all_table_init_energies = np.concatenate(all_table_init_energies)
        all_robot_goal_energies = np.concatenate(all_robot_goal_energies)
        all_table_goal_energies = np.concatenate(all_table_goal_energies)

        # 全局标准化
        robot_init_norm = (all_robot_init_energies - np.mean(all_robot_init_energies)) / (np.std(all_robot_init_energies) + 1e-8)
        table_init_norm = (all_table_init_energies - np.mean(all_table_init_energies)) / (np.std(all_table_init_energies) + 1e-8)
        robot_goal_norm = (all_robot_goal_energies - np.mean(all_robot_goal_energies)) / (np.std(all_robot_goal_energies) + 1e-8)
        table_goal_norm = (all_table_goal_energies - np.mean(all_table_goal_energies)) / (np.std(all_table_goal_energies) + 1e-8)

        # 合并结果
        all_energies_init = robot_init_norm + table_init_norm
        all_energies_goal = robot_goal_norm + table_goal_norm
        all_combined_energies = all_energies_init + all_energies_goal
    
    return all_labels, all_energies_init, all_energies_goal, all_combined_energies


def evaluate_combined_threshold(all_labels, all_combined_energies, threshold):
    """评估组合能量阈值的性能"""
    y_true = []
    y_pred = []
    
    for idx in range(len(all_labels)):
        labels = all_labels[idx]
        energies = all_combined_energies[idx]
        
        # 获取预测
        pred = np.zeros_like(labels)
        mask = energies.flatten() < threshold
        pred.flat[mask] = 1
        
        y_true.append(labels)
        y_pred.append(pred)
    
    # 将列表转换为数组
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    
    # 计算评估指标
    accuracy = np.mean(np.all(y_true == y_pred, axis=1)) * 100
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 计算每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # 返回评估结果
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class
    }


def evaluate_with_threshold(models, data_loader, grasp_file, device, threshold, mode="standard"):
    """使用给定阈值在数据集上评估模型性能
    
    Args:
        models: 模型字典
        data_loader: 数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        threshold: 固定的能量阈值
        mode: 评估模式，可以是 "standard"(标准) 或 "extended"(扩展)
    
    Returns:
        evaluation_results: 包含各种评估指标的字典
    """
    # 收集模型预测
    all_labels, all_energies_init, all_energies_goal, all_combined_energies = collect_model_predictions_with_mode(
        models, data_loader, grasp_file, device, mode)
    
    # 使用固定阈值评估性能
    results = evaluate_combined_threshold(all_labels, all_combined_energies, threshold)
    
    print(f"\n使用阈值 {threshold:.3f} 的评估结果:")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision_macro']:.4f}")
    print(f"Recall: {results['recall_macro']:.4f}")
    print(f"F1 Score: {results['f1_macro']:.4f}")
    
    return results


def test_AND_model_with_dataset(models, test_loader, grasp_file, device, thresholds, mode="standard", verbose=True):
    """在测试集上寻找最佳阈值
    
    Args:
        models: 模型字典，格式取决于mode参数
        test_loader: 测试数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        thresholds: 用于评估的能量阈值列表
        mode: 评估模式，可以是 "standard"(标准，两个模型) 或 "extended"(扩展，四个模型)
        verbose: 是否输出详细信息
    """
    # 收集模型预测
    all_labels, all_energies_init, all_energies_goal, all_combined_energies = collect_model_predictions_with_mode(
        models, test_loader, grasp_file, device, mode)
    
    # 初始化结果数组
    precision_array = np.zeros(len(thresholds))
    recall_array = np.zeros(len(thresholds))
    f1_array = np.zeros(len(thresholds))
    accuracy_array = np.zeros(len(thresholds))
    
    # 存储所有结果
    results_data = []
    
    # 初始化最佳结果跟踪
    best_f1 = -1
    best_threshold = 0
    best_results = None
    
    # 遍历阈值
    for i, threshold in enumerate(tqdm.tqdm(thresholds, desc="测试阈值")):
        # 评估当前阈值
        results = evaluate_combined_threshold(
            all_labels, all_combined_energies, threshold
        )
        
        # 记录结果
        precision_array[i] = results['precision_macro']
        recall_array[i] = results['recall_macro']
        f1_array[i] = results['f1_macro']
        accuracy_array[i] = results['accuracy']
        
        results_data.append(results)
        
        # 更新最佳结果
        if results['f1_macro'] > best_f1:
            best_f1 = results['f1_macro']
            best_threshold = threshold
            best_results = results

        # 输出当前结果
        if verbose and i % 5 == 0:
            print(f"\nThreshold = {threshold:.2f}:")
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Macro - P: {results['precision_macro']:.4f}, "
                  f"R: {results['recall_macro']:.4f}, "
                  f"F1: {results['f1_macro']:.4f}")

    # 可视化F1分数随阈值变化
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_array, marker='o', linestyle='-', color='blue')
    plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.3f}')
    plt.grid(True)
    plt.xlabel('Combined Energy Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Combined Energy Threshold (Test Set)')
    plt.legend()
    plt.savefig('f1_vs_threshold_test.png')
    
    # 记录结果到wandb
    wandb.log({
        "test_best_f1": best_f1,
        "best_threshold": best_threshold,
        "test_precision_vs_threshold": wandb.plot.line_series(
            xs=thresholds.tolist(),
            ys=[precision_array.tolist()],
            keys=["Precision"],
            title="Test Precision vs Threshold",
            xname="Threshold"
        ),
        "test_recall_vs_threshold": wandb.plot.line_series(
            xs=thresholds.tolist(),
            ys=[recall_array.tolist()],
            keys=["Recall"],
            title="Test Recall vs Threshold",
            xname="Threshold"
        ),
        "test_f1_vs_threshold": wandb.plot.line_series(
            xs=thresholds.tolist(),
            ys=[f1_array.tolist()],
            keys=["F1"],
            title="Test F1 vs Threshold",
            xname="Threshold"
        ),
        "test_f1_vs_threshold_plot": wandb.Image('f1_vs_threshold_test.png')
    })

    threshold_results = {
        'precision': precision_array,
        'recall': recall_array,
        'f1': f1_array,
        'accuracy': accuracy_array,
        'thresholds': thresholds
    }

    return best_threshold, best_results, threshold_results


def load_raw_data(data_path, ratio=0.5):
    """加载原始数据
    Args:
        data_path: 数据文件路径
        ratio: 使用数据的比例，默认0.5表示使用后半部分数据
    """
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
        start_idx = int(len(raw_data) * ratio)
        raw_data = raw_data[start_idx:]
        print(f"加载数据: 总量 {len(raw_data)}, 数据比例 {1 - ratio:.1%}")
    return raw_data


def split_data_indices(total_size, train_split, val_split):
    """划分数据集索引"""
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    return {
        'train': indices[:train_size],
        'val': indices[train_size:train_size+val_size],
        'test': indices[train_size+val_size:]
    }


def create_datasets(raw_data, indices, args):
    # 创建训练集 - 利用init的数据（保证数据量一致）
    train_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )

    # 创建验证集和测试集
    val_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )
    
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, args):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * 2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,  # 添加这个参数
        prefetch_factor=2  # 添加这个参数
    )
    
    # 验证和测试加载器类似修改
   # val_sampler = BalancedBatchSampler(val_dataset, args.batch_size)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=args.batch_size * 2,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
  #  test_sampler = BalancedBatchSampler(test_dataset, args.batch_size)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size * 2,  # 添加批量大小，与验证集保持一致
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和测试共享抓取能量网络')
    
    # 数据参数
    parser.add_argument('--test_dataset', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/grasps/Bottle/SharedGraspNetwork_bottle_experiment_data_57.pickle',
                       help='测试数据集路径')
    parser.add_argument('--grasp_pickle_file', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/grasps/Bottle/bottle_grasp_57.pickle',
                       help='抓取候选文件路径')
    
    # 兼容两种模式的模型路径
    parser.add_argument('--model_mode', type=str, default='standard',
                       choices=['standard', 'extended'],
                       help='模型评估模式: standard(两个模型) 或 extended(四个模型)')
    
    # 标准模式的模型路径
    parser.add_argument('--model_init_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_init.pth',
                       help='初始状态模型路径 (标准模式)')
    parser.add_argument('--model_goal_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_goal.pth',
                       help='目标状态模型路径 (标准模式)')
    
    # 扩展模式的模型路径
    parser.add_argument('--model_robot_init_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_robot_init.pth',
                       help='robot初始状态模型路径 (扩展模式)')
    parser.add_argument('--model_table_init_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_table_init.pth',
                       help='table初始状态模型路径 (扩展模式)')
    parser.add_argument('--model_robot_goal_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_robot_goal.pth',
                       help='robot目标状态模型路径 (扩展模式)')
    parser.add_argument('--model_table_goal_path', type=str, 
                       default='E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project/model/feasible_best_model/best_model_grasp_table_goal.pth',
                       help='table目标状态模型路径 (扩展模式)')
    
    # 数据集参数
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批量大小')
    parser.add_argument('--data_ratio', type=float, default=0.5,
                       help='数据比例')
    parser.add_argument('--use_quaternion', type=int, default=1,
                       help='使用四元数 (1) 或欧拉角 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1,
                       help='使用稳定性标签 (1) 或不使用 (0)')
    parser.add_argument('--grasp_type', type=str, default='robot_table',
                       choices=['robot_table', 'table', 'robot'],
                       help='使用哪种抓取类型')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=17) 
    parser.add_argument('--output_dim', type=int, default=120)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='EBM_grasp_AND_evaluation')
    parser.add_argument('--wandb_name', type=str, default='ebm_and_evaluation')
    
    args = parser.parse_args()
    args.use_quaternion = bool(args.use_quaternion)
    args.use_stable_label = bool(args.use_stable_label)
    
    return args


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 加载原始数据
    raw_data = load_raw_data(args.test_dataset, args.data_ratio)

    indices = split_data_indices(len(raw_data), args.train_split, args.val_split)
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(raw_data, indices, args)
    
    # 及时清理原始数据
    del raw_data, indices
    gc.collect()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, args
    )

    # 设置输入维度 - 检查与训练时是否一致
    input_dim = args.input_dim  # 物体位姿 + 稳定标签 + 抓取位姿

    # 创建模型和加载权重 - 根据模式选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    if args.model_mode == 'standard':
        # 标准模式：使用两个模型
        models['init'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                         num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        models['goal'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                         num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        
        # 加载模型权重
        checkpoint_init = torch.load(args.model_init_path, map_location=device)
        models['init'].load_state_dict(checkpoint_init['model_state_dict'])
        models['init'] = models['init'].to(device).float()
        
        checkpoint_goal = torch.load(args.model_goal_path, map_location=device)
        models['goal'].load_state_dict(checkpoint_goal['model_state_dict'])
        models['goal'] = models['goal'].to(device).float()
        
        print(f"\n使用标准模式（两个模型）进行评估")
        print(f"初始状态模型: {os.path.basename(args.model_init_path)}")
        print(f"目标状态模型: {os.path.basename(args.model_goal_path)}")
        
    elif args.model_mode == 'extended':
        # 扩展模式：使用四个模型
        models['robot_init'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                               num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        models['table_init'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                               num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        models['robot_goal'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                               num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        models['table_goal'] = GraspEnergyNetwork(input_dim=input_dim, hidden_dims=args.hidden_dims,
                                               num_layers=args.num_layers, dropout_rate=args.dropout_rate)
        
        # 加载模型权重
        checkpoint_robot_init = torch.load(args.model_robot_init_path, map_location=device)
        models['robot_init'].load_state_dict(checkpoint_robot_init['model_state_dict'])
        models['robot_init'] = models['robot_init'].to(device).float()
        
        checkpoint_table_init = torch.load(args.model_table_init_path, map_location=device)
        models['table_init'].load_state_dict(checkpoint_table_init['model_state_dict'])
        models['table_init'] = models['table_init'].to(device).float()
        
        checkpoint_robot_goal = torch.load(args.model_robot_goal_path, map_location=device)
        models['robot_goal'].load_state_dict(checkpoint_robot_goal['model_state_dict'])
        models['robot_goal'] = models['robot_goal'].to(device).float()
        
        checkpoint_table_goal = torch.load(args.model_table_goal_path, map_location=device)
        models['table_goal'].load_state_dict(checkpoint_table_goal['model_state_dict'])
        models['table_goal'] = models['table_goal'].to(device).float()
        
        print(f"\n使用扩展模式（四个模型）进行评估")
        print(f"Robot初始状态模型: {os.path.basename(args.model_robot_init_path)}")
        print(f"Table初始状态模型: {os.path.basename(args.model_table_init_path)}")
        print(f"Robot目标状态模型: {os.path.basename(args.model_robot_goal_path)}")
        print(f"Table目标状态模型: {os.path.basename(args.model_table_goal_path)}")
    
    # 将所有模型设为评估模式
    for model in models.values():
        model.eval()

    # 设置组合能量阈值范围 - 由于对能量进行了标准化处理，能量值会围绕0分布
    if args.model_mode == 'standard':
        thresholds = np.arange(-10, -2, 0.1)  # 扩大范围
    elif args.model_mode == 'extended':
        thresholds = np.arange(-5, -4, 0.01)  # 扩大范围

    # 初始化wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # 第一步：在测试集上寻找最佳阈值
    print("\n在测试集上寻找最佳阈值...")
    best_threshold, test_results, threshold_results = test_AND_model_with_dataset(
        models, test_loader, args.grasp_pickle_file, device, thresholds, mode=args.model_mode
    )
    
    # 输出测试集上的最佳结果
    print("\n测试集上的最佳阈值结果:")
    print(f"组合能量阈值: {best_threshold:.3f}")
    print(f"准确率: {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision_macro']:.4f}")
    print(f"Recall: {test_results['recall_macro']:.4f}")
    print(f"F1 Score: {test_results['f1_macro']:.4f}")
    
    # 第二步：使用找到的最佳阈值在验证集上评估
    print("\n使用找到的最佳阈值在验证集上评估...")
    val_results = evaluate_with_threshold(
        models, val_loader, args.grasp_pickle_file, device, best_threshold, mode=args.model_mode
    )
    
    # 记录验证集结果到wandb
    wandb.log({
        "val_accuracy": val_results['accuracy'],
        "val_precision": val_results['precision_macro'],
        "val_recall": val_results['recall_macro'],
        "val_f1": val_results['f1_macro']
    })
    
    # 输出最终结果对比
    print("\n=== 结果对比 ===")
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"测试集 - F1: {test_results['f1_macro']:.4f}, Accuracy: {test_results['accuracy']:.2f}%")
    print(f"验证集 - F1: {val_results['f1_macro']:.4f}, Accuracy: {val_results['accuracy']:.2f}%")
    
    # 将最终结果保存到文件
    results_summary = {
        'best_threshold': best_threshold,
        'test_results': test_results,
        'val_results': val_results,
        'config': vars(args)
    }
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    
    # 保存结果摘要
    result_file = f'output/{args.wandb_name}_results.pickle'
    with open(result_file, 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\n结果已保存到 {result_file}")

    def check_dataset_distribution(dataset, name):
        labels = [label.item() for _, label in dataset]
        positive_ratio = sum(labels) / len(labels)
        print(f"{name} 数据集: 总样本 {len(labels)}, 正样本比例 {positive_ratio:.4f}")

    check_dataset_distribution(train_dataset, "训练")
    check_dataset_distribution(val_dataset, "验证")
    check_dataset_distribution(test_dataset, "测试")

