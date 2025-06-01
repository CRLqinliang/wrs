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
from torch.nn.functional import threshold

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
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

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


def evaluate_with_model_thresholds(models, data_loader, grasp_file, device, thresholds_dict, mode="standard"):
    """使用各模型自带阈值在数据集上评估模型性能
    
    Args:
        models: 模型字典
        data_loader: 数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        thresholds_dict: 各模型的阈值字典，格式与models相同
        mode: 评估模式，可以是 "standard"(标准) 或 "extended"(扩展)
    
    Returns:
        evaluation_results: 包含各种评估指标的字典
    """ 
    # 收集模型预测
    all_labels, all_combined_binary = collect_model_predictions_with_mode2(
        models, data_loader, grasp_file, device, thresholds_dict, mode
    )

    # all_labels, all_binary_predictions, all_energies, all_combined_binary = collect_model_predictions_with_mode(
    #     models, data_loader, grasp_file, device, thresholds_dict, mode
    # )
    
    # 评估二元预测结果
    results = evaluate_binary_predictions(all_labels, all_combined_binary)
    
    print(f"\n使用各模型自带阈值的评估结果:")
    print(f"准确率: {results['accuracy']:.2f}%")
    print(f"Binary - Precision: {results['precision']:.4f}")
    print(f"Binary - Recall: {results['recall']:.4f}")
    print(f"Binary - F1 Score: {results['f1']:.4f}")

    return results


def collect_model_predictions_with_mode2(models, test_loader, grasp_file, device, thresholds, mode="standard"):
    """收集能量模型的预测结果，使用各模型自带阈值进行逻辑AND操作
    
    Args:
        models: 模型字典，包含不同模式需要的模型
            - 标准模式(standard): {'init': model1, 'goal': model2}
            - 扩展模式(extended): {'robot_init': model1, 'table_init': model2, 'robot_goal': model3, 'table_goal': model4}
        test_loader: 测试数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        thresholds: 各模型的阈值字典，格式与models相同
        mode: 评估模式，可以是 "standard"(标准，两个模型) 或 "extended"(扩展，四个模型)
        
    注意:
        此函数不对能量进行标准化，而是直接使用各模型的阈值进行二元判断，然后通过逻辑AND操作得到最终结果。
    """
    # 将所有模型设置为评估模式
    for model in models.values():
        model.eval()
    
    all_labels = []
    all_binary_predictions = {}  # 存储各模型的二元预测结果
    all_energies = {}  # 存储各模型的原始能量值
    all_combined_binary = []  # 存储逻辑AND后的最终二元预测结果
    
    # 初始化模型列表
    if mode == "standard":
        model_keys = ['init', 'goal']
    else:  # mode == "extended"
        model_keys = ['robot_init', 'table_init', 'robot_goal', 'table_goal']
    
    # 初始化预测结果字典
    for key in model_keys:
        all_binary_predictions[key] = []
        all_energies[key] = []

    # 收集所有批次的预测结果
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

            
            if mode == "standard":
                # 标准模式：使用两个模型分别预测init和goal状态 - 能量归一化((利用在训练中获得到的归一化参数)
                init_energies = models['init'](init_features).cpu().numpy() 
                goal_energies = models['goal'](goal_features).cpu().numpy() 
                
                # 存储原始能量值
                # all_energies['init'].append(init_energies)
                # all_energies['goal'].append(goal_energies)
            
                # 根据阈值进行二元判断
                # combined_energies = init_energies + goal_energies
                # combined_binary = (combined_energies < thresholds['init']).astype(np.int32)

                init_binary = (init_energies < thresholds['init']).astype(np.int32)
                goal_binary = (goal_energies < thresholds['goal']).astype(np.int32)
                
                # 存储二元预测结果
                # all_binary_predictions['init'].append(init_binary)
                # all_binary_predictions['goal'].append(goal_binary)
                
                # 逻辑AND操作
                combined_binary = np.minimum(init_binary, goal_binary)
                
            elif mode == "extended":
                # 扩展模式：使用四个模型预测
                robot_init_energies = models['robot_init'](init_features).cpu().numpy()
                table_init_energies = models['table_init'](init_features).cpu().numpy()
                robot_goal_energies = models['robot_goal'](goal_features).cpu().numpy()
                table_goal_energies = models['table_goal'](goal_features).cpu().numpy()
                
                # 存储原始能量值
                all_energies['robot_init'].append(robot_init_energies)
                all_energies['table_init'].append(table_init_energies)
                all_energies['robot_goal'].append(robot_goal_energies)
                all_energies['table_goal'].append(table_goal_energies)
                
                # 根据阈值进行二元判断
                robot_init_binary = (robot_init_energies < thresholds['robot_init']).astype(np.int32)
                table_init_binary = (table_init_energies < thresholds['table_init']).astype(np.int32)
                robot_goal_binary = (robot_goal_energies < thresholds['robot_goal']).astype(np.int32)
                table_goal_binary = (table_goal_energies < thresholds['table_goal']).astype(np.int32)
                
                # 存储二元预测结果
                all_binary_predictions['robot_init'].append(robot_init_binary)
                all_binary_predictions['table_init'].append(table_init_binary)
                all_binary_predictions['robot_goal'].append(robot_goal_binary)
                all_binary_predictions['table_goal'].append(table_goal_binary)
                
                # 逻辑AND操作 - 先合并init和goal的结果
                init_binary = np.minimum(robot_init_binary, table_init_binary)
                goal_binary = np.minimum(robot_goal_binary, table_goal_binary)
                combined_binary = np.minimum(init_binary, goal_binary)
            
            # 存储最终的二元预测结果和标签
            all_combined_binary.append(combined_binary)
            all_labels.append(labels)

    # 合并所有批次的结果
    all_labels = np.concatenate(all_labels)
    all_combined_binary = np.concatenate(all_combined_binary)
    
    # 合并各模型的预测结果
    # for key in model_keys:
    #     all_binary_predictions[key] = np.concatenate(all_binary_predictions[key])
    #     all_energies[key] = np.concatenate(all_energies[key])
    
    return all_labels,  all_combined_binary


def evaluate_binary_predictions(all_labels, all_combined_binary):
    """评估二元预测结果的性能
    
    Args:
        all_labels: 真实标签
        all_combined_binary: 通过逻辑AND操作得到的二元预测结果
        
    Returns:
        evaluation_results: 包含各种评估指标的字典
    """
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_combined_binary) * 100
    precision = precision_score(all_labels, all_combined_binary, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_combined_binary, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_combined_binary, average='binary', zero_division=0)
    
    # 返回评估结果
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# - - - - - -- - - - - - - 

def test_AND_model_with_fixed_thresholds(models, test_loader, device, thresholds_dict, mode="standard"):
    """使用固定阈值评估模型在测试集和验证集上的性能"""
    # 在测试集上评估
    print("\n在测试集上评估...")
    # 使用与optimal_f1_threshold相同的函数收集预测
    all_labels, _, _, all_combined_energies = collect_model_predictions_with_mode(models, test_loader, device, mode)

    # 用归一化之后的阈值进行评估
    threshold = thresholds_dict['init']
    # threshold = thresholds_dict['init'] + thresholds_dict['goal']
    test_results = evaluate_combined_threshold(all_labels, all_combined_energies, threshold)
    
        
    # 输出结果
    print(f"\n使用验证集上最佳F1分数对应的阈值，在测试集上的评价结果:")
    print(f"准确率: {test_results['accuracy']:.2f}%")
    print(f"Precision: {test_results['precision_binary']:.4f}")
    print(f"Recall: {test_results['recall_binary']:.4f}")
    print(f"F1 Score: {test_results['f1_binary']:.4f}")
    
    return test_results


def collect_model_predictions_with_mode(models, test_loader, device, mode="standard"):
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
    num = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            num += 1
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
    print(f"收集了 {num} 个批次的能量值")
    # 合并所有批次的能量值
    all_labels = np.concatenate(all_labels)

    if mode == "standard":
        # 标准模式处理
        all_init_energies = np.concatenate(all_init_energies)
        all_goal_energies = np.concatenate(all_goal_energies)

        # 不进行标准化
        all_combined_energies = all_init_energies + all_goal_energies

    elif mode == "extended":
        # 扩展模式处理
        all_robot_init_energies = np.concatenate(all_robot_init_energies)
        all_table_init_energies = np.concatenate(all_table_init_energies)
        all_robot_goal_energies = np.concatenate(all_robot_goal_energies)
        all_table_goal_energies = np.concatenate(all_table_goal_energies)

        # 全局标准化
        robot_init_norm = (all_robot_init_energies - np.mean(all_robot_init_energies)) / (
                    np.std(all_robot_init_energies) + 1e-8)
        table_init_norm = (all_table_init_energies - np.mean(all_table_init_energies)) / (
                    np.std(all_table_init_energies) + 1e-8)
        robot_goal_norm = (all_robot_goal_energies - np.mean(all_robot_goal_energies)) / (
                    np.std(all_robot_goal_energies) + 1e-8)
        table_goal_norm = (all_table_goal_energies - np.mean(all_table_goal_energies)) / (
                    np.std(all_table_goal_energies) + 1e-8)

        # 合并结果
        all_energies_init = robot_init_norm + table_init_norm
        all_energies_goal = robot_goal_norm + table_goal_norm
        all_combined_energies = all_energies_init + all_energies_goal

    return all_labels, all_init_energies, all_goal_energies, all_combined_energies


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
    precision_binary = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall_binary = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average='binary', zero_division=0)


    # 返回评估结果
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision_binary": precision_binary,
        "recall_binary": recall_binary,
        "f1_binary": f1_binary
    }


def optimal_f1_threshold(models, val_loader, device, mode="standard"):
    """直接计算最佳F1分数对应的阈值
    
    Args:
        models: 模型字典
        test_loader: 测试数据加载器
        grasp_file: 抓取文件路径
        device: 计算设备
        mode: 评估模式
    """
    
    # 收集模型预测
    all_labels, all_energies_init, all_energies_goal, all_combined_energies = collect_model_predictions_with_mode(
        models, val_loader, device, mode)
    
    # 计算精确率-召回率曲线
    # 注意：能量值越低表示越可能是正样本，所以需要取负值
    precision, recall, thresholds = precision_recall_curve(all_labels, -all_combined_energies)
    
    # 计算每个阈值点的F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 找到F1分数最大的索引
    best_idx = np.argmax(f1_scores)
    
    # 获取对应的阈值
    # 注意：precision_recall_curve返回的thresholds比precision和recall少一个元素
    if best_idx < len(thresholds):
        best_threshold = -thresholds[best_idx]  # 需要转回原始能量的符号
    else:
        # 如果最佳点是最后一个，则使用一个极小值
        best_threshold = np.min(all_combined_energies) - 1.0
    
    # 评估最佳阈值
    best_results = evaluate_combined_threshold(all_labels, all_combined_energies, best_threshold)
    best_f1 = best_results['f1_binary']
    
    print(f"最佳F1分数: {best_f1:.4f}, 对应阈值: {best_threshold:.4f}")
    
    return best_results, best_threshold, best_f1


def optimal_separate_thresholds(models, val_loader, test_loader, grasp_file, device, mode="standard"):
    """使用相同阈值进行init和goal的判断，通过百分位数采样找到最优F1对应的阈值"""
    # 在验证集上收集模型预测
    print("\n在验证集上寻找最佳阈值...")
    all_labels, all_energies_init, all_energies_goal, _ = collect_model_predictions_with_mode(
        models, val_loader, grasp_file, device, mode)
    
    # 计算init和goal能量的范围
    init_percentiles = np.percentile(all_energies_init, np.linspace(0, 100, 100))
    goal_percentiles = np.percentile(all_energies_goal, np.linspace(0, 100, 100))
    
    # 合并并去重所有可能的阈值点
    thresholds = np.unique(np.concatenate([init_percentiles, goal_percentiles]))
    print(f"采样得到 {len(thresholds)} 个候选阈值点")
    
    # 计算每个阈值点的F1分数
    f1_scores = []
    for threshold in thresholds:
        init_binary = (all_energies_init < threshold).astype(np.int32)
        goal_binary = (all_energies_goal < threshold).astype(np.int32)
        combined_binary = np.minimum(init_binary, goal_binary)
        
        # 计算F1分数
        f1 = f1_score(all_labels, combined_binary, average='binary', zero_division=0)
        f1_scores.append(f1)
    
    # 找到最佳F1分数对应的阈值
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # 使用最佳阈值计算验证集上的指标
    init_binary = (all_energies_init < best_threshold).astype(np.int32)
    goal_binary = (all_energies_goal < best_threshold).astype(np.int32)
    combined_binary = np.minimum(init_binary, goal_binary)
    
    best_val_results = {
        "accuracy": accuracy_score(all_labels, combined_binary) * 100,
        "precision": precision_score(all_labels, combined_binary, average='binary', zero_division=0),
        "recall": recall_score(all_labels, combined_binary, average='binary', zero_division=0),
        "f1": f1_score(all_labels, combined_binary, average='binary', zero_division=0)
    }
    
    print(f"\n验证集上的最佳结果:")
    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"验证集准确率: {best_val_results['accuracy']:.2f}%")
    print(f"验证集Precision: {best_val_results['precision']:.4f}")
    print(f"验证集Recall: {best_val_results['recall']:.4f}")
    print(f"验证集F1 Score: {best_val_results['f1']:.4f}")

    # 在测试集上进行最终评估
    print("\n在测试集上进行最终评估...")
    test_labels, test_energies_init, test_energies_goal, _ = collect_model_predictions_with_mode(
        models, test_loader, grasp_file, device, mode)
    
    # 使用找到的最佳阈值进行预测
    test_init_binary = (test_energies_init < best_threshold).astype(np.int32)
    test_goal_binary = (test_energies_goal < best_threshold).astype(np.int32)
    test_combined_binary = np.minimum(test_init_binary, test_goal_binary)
    
    # 计算测试集上的评估指标
    test_results = {
        "accuracy": accuracy_score(test_labels, test_combined_binary) * 100,
        "precision": precision_score(test_labels, test_combined_binary, average='binary', zero_division=0),
        "recall": recall_score(test_labels, test_combined_binary, average='binary', zero_division=0),
        "f1": f1_score(test_labels, test_combined_binary, average='binary', zero_division=0)
    }
    
    print(f"\n测试集上的最终结果:")
    print(f"测试集准确率: {test_results['accuracy']:.2f}%")
    print(f"测试集Precision: {test_results['precision']:.4f}")
    print(f"测试集Recall: {test_results['recall']:.4f}")
    print(f"测试集F1 Score: {test_results['f1']:.4f}")
    
    # 返回相同的阈值
    best_thresholds = {
        "init": best_threshold,
        "goal": best_threshold
    }
    
    return best_val_results, test_results, best_thresholds


# - - -

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


def split_data_indices(total_size, train_split, val_split, test_split):
    """划分数据集索引"""
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    return {
        'train': indices[:train_size],
        'val': indices[train_size:train_size+val_size],
        'test': indices[train_size+val_size:train_size+val_size+test_size]
    }


def create_datasets(raw_data, indices, args):
    # 创建训练集 - 利用init的数据（保证数据量一致）
    # train_dataset = SharedGraspEnergyDataset(
    #     [raw_data[i] for i in indices['train']],
    #     args.grasp_pickle_file,
    #     use_quaternion=args.use_quaternion,
    #     use_stable_label=args.use_stable_label,
    #     grasp_type=args.grasp_type,
    #     state_type='both'
    # )

    # 创建验证集和测试集
    val_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['val']],
        args.val_grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )
    
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.test_grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )
    
    return  val_dataset, test_dataset


def create_data_loaders( val_dataset, test_dataset, args):
    """创建数据加载器"""
    # train_loader = DataLoader(
    #     train_dataset,
    #     shuffle=False,
    #     batch_size=args.batch_size * 2,
    #     num_workers=2,
    #     pin_memory=True,
    #     persistent_workers=True,  # 添加这个参数
    #     prefetch_factor=2  # 添加这个参数
    # )
    
    # 验证和测试加载器类似修改
    # val_sampler = BalancedBatchSampler(val_dataset, args.batch_size)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
  #  test_sampler = BalancedBatchSampler(test_dataset, args.batch_size)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,  # 添加批量大小，与验证集保持一致
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return val_loader, test_loader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和测试共享抓取能量网络')
    
    # 数据参数
    parser.add_argument('--test_dataset', type=str, 
                       default=rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_922.pickle',
                       help='测试数据集路径')
    parser.add_argument('--test_grasp_pickle_file', type=str, 
                       default=rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_922.pickle',
                       help='抓取候选文件路径')
    
    
    parser.add_argument('--validate_dataset_path',type=str, default=rf"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_352.pickle",
                        help='验证数据集路径')
    parser.add_argument('--val_grasp_pickle_file',type=str, default=rf"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_352.pickle",
                        help='验证抓取文件路径')
    
    
    # 兼容两种模式的模型路径
    parser.add_argument('--model_mode', type=str, default='standard',
                       choices=['standard', 'extended'],
                       help='模型评估模式: standard(两个模型) 或 extended(四个模型)')
    
    # 标准模式的模型路径
    parser.add_argument('--model_init_path', type=str, 
                       default=rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_352_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl1_grobot_table_stinit.pth',
                       help='初始状态模型路径 (标准模式)')
    parser.add_argument('--model_goal_path', type=str, 
                       default=rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_352_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl1_grobot_table_stinit.pth',
                       help='目标状态模型路径 (标准模式)')
    
    # 扩展模式的模型路径
    parser.add_argument('--seed', type = int, default=42)
    

    # 数据集参数
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批量大小')
    parser.add_argument('--data_ratio', type=float, default=0.0,
                       help='数据比例')
    parser.add_argument('--use_quaternion', type=int, default=1,
                       help='使用四元数 (1) 或欧拉角 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1,
                       help='使用稳定性标签 (1) 或不使用 (0)')

    parser.add_argument('--grasp_type', type=str, default='robot_table',
                       choices=['robot_table', 'table', 'robot'],
                       help='使用哪种抓取类型')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=19)
    parser.add_argument('--output_dim', type=int, default=1)
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
    random.seed(args.seed)

    # 加载测试数据
    raw_data = load_raw_data(args.test_dataset, 0.0) # 100% 的数据测试；
    print(f"测试数据集长度: {len(raw_data)}")

    # 加载验证数据
    raw_data_validate = load_raw_data(args.validate_dataset_path, args.data_ratio)
    print(f"验证数据集长度: {len(raw_data_validate)}")

    # 划分数据集索引
    indices = split_data_indices(len(raw_data), 0.1, 0.1, 0.1) # 1K 数据集测试
    valide_indices = split_data_indices(len(raw_data_validate), args.train_split, args.val_split /2, args.test_split) # 10K 数据集验证阈值
    valide_indices['val'] = valide_indices['val'][:10000]

    # 创建数据集
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.test_grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )

    val_dataset_validate = SharedGraspEnergyDataset(
        [raw_data_validate[i] for i in valide_indices['val']],
        args.val_grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type='both'
    )

    # 及时清理原始数据
    del raw_data, raw_data_validate, indices, valide_indices
    gc.collect()
    
    # 创建数据加载器
    val_loader, test_loader = create_data_loaders(val_dataset_validate, test_dataset, args)

    # 设置输入维度 - 检查与训练时是否一致
    input_dim = args.input_dim  # 物体位姿 + 稳定标签 + 抓取位姿

    # 创建模型和加载权重 - 根据模式选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    thresholds_dict = {}
    

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
    
    # 设置各模型的阈值 - 从检查点中获取或使用默认值
    thresholds_dict['init'] = checkpoint_init.get('optimal_threshold', -5.0)
    thresholds_dict['goal'] = checkpoint_goal.get('optimal_threshold', -5.0)
    
    print(f"\n使用标准模式（两个模型）进行评估")
    print(f"初始状态模型: {os.path.basename(args.model_init_path)}, 阈值: {thresholds_dict['init']:.3f}")
    print(f"目标状态模型: {os.path.basename(args.model_goal_path)}, 阈值: {thresholds_dict['goal']:.3f}")
        

    # 将所有模型设为评估模式
    for model in models.values():
        model.eval()

    # 初始化wandb
    # wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # 添加阈值信息到wandb配置
    # wandb.config.update({"thresholds": thresholds_dict})

    # 使用自带阈值评估模型 - 不进行归一化：
    # test_results = evaluate_with_model_thresholds(
    #     models, test_loader, args.grasp_pickle_file,
    #       device, thresholds_dict, mode=args.model_mode
    # )


    # 使用固定阈值评估模型
    # test_results = test_AND_model_with_fixed_thresholds(
    #     models, test_loader, val_loader, args.grasp_pickle_file, device, thresholds_dict, mode=args.model_mode
    # )

    # 在val上找到能量加和之后的最佳阈值，并在test上测试
    best_results, best_threshold, best_f1 = optimal_f1_threshold(models, val_loader,  device, args.model_mode )
    thresholds_dict['init'] = best_threshold
    thresholds_dict['goal'] = best_threshold
    test_results = test_AND_model_with_fixed_thresholds(models, test_loader, device, thresholds_dict, mode=args.model_mode)

    # 在val上找到在binary之前的最佳阈值，并在test上测试
    # best_val_results, test_results, best_thresholds = optimal_separate_thresholds(
    #     models, val_loader, test_loader, args.grasp_pickle_file, device, args.model_mode
    # )


