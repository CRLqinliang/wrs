""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241205 Osaka Univ.
"""

from typing import List, Tuple, Optional, Union
import sys
import pickle
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
sys.path.append("E:/Qin/wrs")
import argparse, tqdm, wandb
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, average_precision_score, classification_report
)
from wrs.HuGroup_Qin.Shared_grasp_project.script.shared_grasp_network.feasible_grasp.grasp_network_train import GraspNetwork
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import OneHotEncoder

# 常量定义
OBJECT_MESH_PATH = Path(r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")


class SharedGraspBinaryDataset(Dataset):
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
    


def collect_model_predictions_with_mode(models, test_loader, device, mode="standard", model_thresholds=None):
    """收集模型的预测结果，支持不同模式
    
    Args:
        models: 模型字典，格式取决于mode参数
            - standard: {'init': model1, 'goal': model2}
            - extended: {'robot_init': model1, 'table_init': model2, 'robot_goal': model3, 'table_goal': model4}
        test_loader: 测试数据加载器
        device: 计算设备
        mode: 评估模式，"standard"(两个模型) 或 "extended"(四个模型)
        model_thresholds: 模型阈值字典
    """
    # 检查阈值字典是否存在
    if model_thresholds is None:
        model_thresholds = {k: 0.5 for k in models.keys()}  # 默认所有模型阈值为0.5
    
    # 将所有模型设置为评估模式
    for model in models.values():
        model.eval()
    
    all_labels = []
    all_probs_init = []
    all_probs_goal = []
    all_combined_probs = []
    all_binary_preds = []  # 新增：存储二值化预测结果

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="收集预测结果"):
            inputs = inputs.to(device).float()
            labels = labels.cpu().numpy()
            
            # 判断是否使用quaternion表示
            feature_dim = inputs.shape[1]

            # 确定各部分的维度
            obj_pose_dim = 7 
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
                # 标准模式：使用两个模型
                init_probs = torch.sigmoid(models['init'](init_features)).cpu().numpy()
                goal_probs = torch.sigmoid(models['goal'](goal_features)).cpu().numpy()
                
                # 使用各自模型的阈值进行二值化
                init_binary = (init_probs >= model_thresholds['init']).astype(float)
                goal_binary = (goal_probs >= model_thresholds['goal']).astype(float)
                
                # 合并二值化结果（AND操作）
                combined_binary = np.minimum(init_binary, goal_binary)
                
                all_probs_init.append(init_probs)
                all_probs_goal.append(goal_probs)
                all_binary_preds.append(combined_binary)
                
            elif mode == "extended":
                # 扩展模式：使用四个模型
                robot_init_probs = torch.sigmoid(models['robot_init'](init_features)).cpu().numpy()
                table_init_probs = torch.sigmoid(models['table_init'](init_features)).cpu().numpy()
                robot_goal_probs = torch.sigmoid(models['robot_goal'](goal_features)).cpu().numpy()
                table_goal_probs = torch.sigmoid(models['table_goal'](goal_features)).cpu().numpy()
                
                # 使用各自模型的阈值进行二值化
                robot_init_binary = (robot_init_probs >= model_thresholds['robot_init']).astype(float)
                table_init_binary = (table_init_probs >= model_thresholds['table_init']).astype(float)
                robot_goal_binary = (robot_goal_probs >= model_thresholds['robot_goal']).astype(float)
                table_goal_binary = (table_goal_probs >= model_thresholds['table_goal']).astype(float)
                
                # 初始状态的二值化结果 = robot_init AND table_init
                init_binary = np.minimum(robot_init_binary, table_init_binary)
                # 目标状态的二值化结果 = robot_goal AND table_goal
                goal_binary = np.minimum(robot_goal_binary, table_goal_binary)
                
                # 合并初始状态和目标状态的二值化结果
                combined_binary = np.minimum(init_binary, goal_binary)
                
                # 为保持接口一致，仍然保存原始概率
                init_probs = np.minimum(robot_init_probs, table_init_probs)
                goal_probs = np.minimum(robot_goal_probs, table_goal_probs)
                
                all_probs_init.append(init_probs)
                all_probs_goal.append(goal_probs)
                all_binary_preds.append(combined_binary)
            
            all_labels.append(labels)

    # 合并所有批次的概率值
    all_labels = np.concatenate(all_labels)
    all_probs_init = np.concatenate(all_probs_init)
    all_probs_goal = np.concatenate(all_probs_goal)
    all_binary_preds = np.concatenate(all_binary_preds)
    

    return all_labels, all_probs_init, all_probs_goal, all_combined_probs, all_binary_preds


def test_AND_model_with_dataset(models, test_loader, device, thresholds=None, mode="standard", 
                                verbose=False, model_thresholds=None):
    """在测试集上评估模型
    
    Args:
        models: 模型字典
        test_loader: 测试数据加载器
        device: 计算设备
        thresholds: 不再使用，保留参数以兼容接口
        mode: 评估模式，"standard"(标准) 或 "extended"(扩展)
        verbose: 是否输出详细信息
        model_thresholds: 模型阈值字典
    """
    # 收集模型预测
    all_labels, all_probs_init, all_probs_goal, all_combined_probs, all_binary_preds = collect_model_predictions_with_mode(
        models, test_loader, device, mode, model_thresholds
    )
    
    # 评估使用模型自身阈值的二值化结果
    binary_accuracy = accuracy_score(all_labels, all_binary_preds) * 100
    binary_precision = precision_score(all_labels, all_binary_preds, average='binary', zero_division=0)
    binary_recall = recall_score(all_labels, all_binary_preds, average='binary', zero_division=0)
    binary_f1 = f1_score(all_labels, all_binary_preds, average='binary', zero_division=0)
    
    print("\n使用模型自身阈值的二值化结果:")
    print(f"Accuracy: {binary_accuracy:.2f}%")
    print(f"Precision: {binary_precision:.4f}, Recall: {binary_recall:.4f}, F1: {binary_f1:.4f}")
    
    # 创建结果字典
    binary_results = {
        "threshold": "model_optimal",
        "accuracy": binary_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1
    }
    
    # 可视化混淆矩阵
    if verbose:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(all_labels, all_binary_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # 确保output/plots目录存在
        os.makedirs('output/plots', exist_ok=True)
        plt_path = f'output/plots/confusion_matrix_{mode}.png'
        plt.savefig(plt_path)
        
        # 如果wandb可用，记录结果
        if wandb.run is not None:
            wandb.log({
                "binary_f1": binary_f1,
                "binary_precision": binary_precision,
                "binary_recall": binary_recall, 
                "binary_accuracy": binary_accuracy,
                "confusion_matrix": wandb.Image(plt_path)
            })
    
    # 返回结果
    threshold_results = {
        'binary_results': binary_results,
        'binary_preds': all_binary_preds,
        'true_labels': all_labels
    }

    return "model_optimal", binary_results, threshold_results



def load_raw_data(data_path, data_ratio=0.5):
    """加载原始数据
    
    Args:
        data_path: 数据文件路径
        data_ratio: 使用数据的比例，默认0.5表示使用一半数据
    """
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
        
        # 如果data_ratio是整数，则直接使用该数量的数据
        if isinstance(data_ratio, int) and data_ratio > 1:
            if data_ratio > len(raw_data):
                print(f"警告: 请求的数据量({data_ratio})超过可用数据({len(raw_data)})")
                data_ratio = len(raw_data)
            # 使用前data_ratio条数据
            raw_data = raw_data[:data_ratio]
            print(f"加载数据: 总量 {len(raw_data)}, 使用前 {data_ratio} 条")
        else:
            # 使用比例
            if data_ratio <= 0:
                start_idx = 0
            else:
                start_idx = int(len(raw_data) * (1 - data_ratio))
            raw_data = raw_data[start_idx:]
            print(f"加载数据: 总量 {len(raw_data)}, 数据比例 {data_ratio:.2f}")
            
    return raw_data


def split_data_indices(data_len, train_split=0.7, val_split=0.15, seed=42):
    """划分数据集索引
    
    Args:
        data_len: 数据总长度
        train_split: 训练集比例
        val_split: 验证集比例
        seed: 随机种子
    
    Returns:
        dict: 包含训练、验证和测试集索引的字典
    """
    import random
    random.seed(seed)
    
    indices = list(range(data_len))
    random.shuffle(indices)
    
    train_size = int(train_split * data_len)
    val_size = int(val_split * data_len)
    
    return {
        'train': indices[:train_size],
        'val': indices[train_size:train_size+val_size],
        'test': indices[train_size+val_size:]
    }


def find_optimal_threshold_with_pr_curve(models, val_loader, device, mode="standard"):
    """使用PR曲线寻找合并预测后F1分数最大的共同阈值
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        device: 计算设备
        mode: 评估模式，"standard"(两个模型) 或 "extended"(四个模型)
    
    Returns:
        dict 或 float: 如果是standard模式返回单一阈值，extended模式返回阈值字典
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # 将所有模型设置为评估模式
    for model in models.values():
        model.eval()
    
    # 收集验证集上的预测结果
    all_labels = []
    model_predictions = {model_name: [] for model_name in models.keys()}
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader, desc="收集验证集预测结果"):
            inputs = inputs.to(device).float()
            labels = labels.cpu().numpy()
            
            # 判断是否使用quaternion表示
            feature_dim = inputs.shape[1]

            # 确定各部分的维度
            obj_pose_dim = 7 
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
                # 标准模式：使用两个模型
                model_predictions['init'].append(torch.sigmoid(models['init'](init_features)).cpu().numpy())
                model_predictions['goal'].append(torch.sigmoid(models['goal'](goal_features)).cpu().numpy())
                
            elif mode == "extended":
                # 扩展模式：使用四个模型
                model_predictions['robot_init'].append(torch.sigmoid(models['robot_init'](init_features)).cpu().numpy())
                model_predictions['table_init'].append(torch.sigmoid(models['table_init'](init_features)).cpu().numpy())
                model_predictions['robot_goal'].append(torch.sigmoid(models['robot_goal'](goal_features)).cpu().numpy())
                model_predictions['table_goal'].append(torch.sigmoid(models['table_goal'](goal_features)).cpu().numpy())
            
            all_labels.append(labels)
    
    # 合并所有批次的预测结果
    all_labels = np.concatenate(all_labels)
    for model_name in model_predictions:
        model_predictions[model_name] = np.concatenate(model_predictions[model_name])
    
    if mode == "standard":
        # 标准模式：为两个模型找一个共同的最佳阈值
        # 创建一个阈值列表进行搜索
        thresholds = np.linspace(0.01, 0.99, 99)  # 从0.01到0.99的99个阈值点
        best_f1 = 0
        best_threshold = 0.5  # 默认阈值
        
        # 对每个阈值计算合并后的F1分数
        for threshold in thresholds:
            # 使用当前阈值对两个模型的预测进行二值化
            init_binary = (model_predictions['init'] >= threshold).astype(float)
            goal_binary = (model_predictions['goal'] >= threshold).astype(float)
            
            # 合并预测结果（AND操作）
            combined_binary = np.minimum(init_binary, goal_binary)
            
            # 计算F1分数
            current_f1 = f1_score(all_labels, combined_binary, average='binary', zero_division=0)
            
            # 更新最佳阈值
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        
        print(f"找到的共同最佳阈值: {best_threshold:.4f}, F1分数: {best_f1:.4f}")
        
        # 对于标准模式，返回单一阈值
        return best_threshold
        
    elif mode == "extended":
        # 扩展模式：仍然为每个模型单独找最佳阈值
        optimal_thresholds = {}
        
        for model_name, predictions in model_predictions.items():
            # 使用PR曲线寻找最佳阈值
            precision, recall, thresholds = precision_recall_curve(all_labels, predictions)
            
            # 计算每个阈值对应的F1分数
            # 注意：precision_recall_curve返回的thresholds比precision和recall少一个元素
            # 所以我们需要在thresholds末尾添加一个0，以匹配precision和recall的长度
            thresholds = np.append(thresholds, 0)
            
            # 计算F1分数 (2 * precision * recall) / (precision + recall)
            # 避免除零错误
            f1_scores = np.zeros_like(precision)
            valid_indices = (precision + recall) > 0
            f1_scores[valid_indices] = 2 * (precision[valid_indices] * recall[valid_indices]) / (precision[valid_indices] + recall[valid_indices])
            
            # 找到最大F1分数对应的索引
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
            
            optimal_thresholds[model_name] = best_threshold
            print(f"模型 {model_name} 的最佳阈值: {best_threshold:.4f}, F1分数: {best_f1:.4f}")
        
        return optimal_thresholds


def test_AND_model_with_optimal_threshold(models, val_loader, test_loader, device, mode="standard", verbose=True):
    """使用验证集上找到的最佳阈值在测试集上评估模型
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        mode: 评估模式，"standard"(标准) 或 "extended"(扩展)
        verbose: 是否输出详细信息
    
    Returns:
        tuple: (最佳阈值字典或单一阈值, 评估结果, 详细结果)
    """
    # 在验证集上寻找最佳阈值
    print("\n在验证集上寻找最佳阈值...")
    optimal_threshold = find_optimal_threshold_with_pr_curve(models, val_loader, device, mode)
    
    # 在测试集上使用最佳阈值评估模型
    print("\n使用最佳阈值在测试集上评估模型...")
    
    # 根据模式处理阈值
    if mode == "standard" and not isinstance(optimal_threshold, dict):
        # 标准模式下，将单一阈值转换为字典格式
        model_thresholds = {'init': optimal_threshold, 'goal': optimal_threshold}
    else:
        # 扩展模式或已经是字典格式
        model_thresholds = optimal_threshold
    
    _, best_results, threshold_results = test_AND_model_with_dataset(
        models, test_loader, device, mode=mode, model_thresholds=model_thresholds
    )
    
    # 输出结果
    print("\n使用验证集优化阈值的测试结果:")
    print(f"最佳阈值: {optimal_threshold}")
    print(f"准确率: {best_results['accuracy']:.2f}%")
    print(f"Binary_Precision: {best_results['binary_precision']:.4f}")
    print(f"Binary_Recall: {best_results['binary_recall']:.4f}")
    print(f"Binary_F1 Score: {best_results['binary_f1']:.4f}")
    
    # 可视化混淆矩阵
    if verbose:
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(threshold_results['true_labels'], threshold_results['binary_preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Optimal Threshold)')
        
        # 确保output/plots目录存在
        os.makedirs('output/plots', exist_ok=True)
        plt_path = f'output/plots/confusion_matrix_optimal_{mode}.png'
        plt.savefig(plt_path)
        
        # 如果wandb可用，记录结果
        if wandb.run is not None:
            wandb.log({
                "optimal_f1": best_results['binary_f1'],
                "optimal_precision": best_results['binary_precision'],
                "optimal_recall": best_results['binary_recall'], 
                "optimal_accuracy": best_results['accuracy'],
                "optimal_confusion_matrix": wandb.Image(plt_path),
                "optimal_threshold": optimal_threshold
            })
    
    return optimal_threshold, best_results, threshold_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='二分类抓取网络AND评估')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_922.pickle",
                        help='数据集路径')
    parser.add_argument('--grasp_pickle_file', type=str,  default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_922.pickle",
                        help='抓取候选文件路径')
    
    # 模型模式参数
    parser.add_argument('--model_mode', type=str, default='standard',
                        choices=['standard', 'extended'],
                        help='模型评估模式: standard(两个模型) 或 extended(四个模型)')
    
    parser.add_argument('--model_init_path', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_BC_SharedGraspNetwork_bottle_experiment_data_352_h3_b2048_lr0.001_r75000_s0.7_q1_sl1_seed42.pth",
                        help='初始状态模型路径 (标准模式)')
    parser.add_argument('--model_goal_path', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_BC_SharedGraspNetwork_bottle_experiment_data_352_h3_b2048_lr0.001_r75000_s0.7_q1_sl1_seed42.pth",
                        help='目标状态模型路径 (标准模式)')
    
    # 扩展模式的模型路径
    parser.add_argument('--model_robot_init_path', type=str, default=None,
                        help='robot初始状态模型路径 (扩展模式)')
    parser.add_argument('--model_table_init_path', type=str, default=None,
                        help='table初始状态模型路径 (扩展模式)')
    parser.add_argument('--model_robot_goal_path', type=str, default=None,
                        help='robot目标状态模型路径 (扩展模式)')
    parser.add_argument('--model_table_goal_path', type=str, default=None,
                        help='table目标状态模型路径 (扩展模式)')
    
    # 数据集参数
    parser.add_argument('--val_split', type=float, default=0.5,
                        help='验证集比例')
    parser.add_argument('--test_split', type=float, default=0.5,
                        help='测试集比例')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='批量大小')
    parser.add_argument('--data_ratio', type=float, default=0.7,
                        help='数据比例')
    parser.add_argument('--use_quaternion', type=int, default=1,
                        help='使用四元数 (1) 或欧拉角 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1,
                        help='使用稳定性标签 (1) 或不使用 (0)')
    parser.add_argument('--grasp_type', type=str, default='robot_table',
                        choices=['robot_table', 'table', 'robot'],
                        help='使用哪种抓取类型')
    parser.add_argument('--state_type', type=str, default='both',
                        choices=['both', 'init', 'goal'],
                        help='使用哪种状态类型')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=19,
                        help='输入维度')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512],
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='网络层数')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout率')
    
    # wandb参数
    parser.add_argument('--wandb_project', type=str, default='bc_grasp_AND_evaluation',
                        help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default='bc_and_evaluation',
                        help='wandb运行名称')
    parser.add_argument('--no_wandb', type=bool, default=True,
                        help='不使用wandb记录')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    
    # 新增：使用验证集优化阈值
    parser.add_argument('--use_optimal_threshold', type=bool, default=True,
                       help='使用验证集优化阈值')
    
    args = parser.parse_args()
    args.use_quaternion = bool(args.use_quaternion)
    args.use_stable_label = bool(args.use_stable_label)
    
    return args


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    raw_data = load_raw_data(args.dataset, args.data_ratio)
    
    # 划分数据集
    print("\n划分数据集...")
    data_len = len(raw_data)
    
    # 计算验证集和测试集大小
    val_size = int(args.val_split * data_len)
    test_size = int(args.test_split * data_len)
    
    # 确保验证集和测试集大小不超过数据总量
    if val_size + test_size > data_len:
        print(f"警告: 验证集({val_size})和测试集({test_size})总和超过数据总量({data_len})")
        # 按比例调整
        total = args.val_split + args.test_split
        val_size = int((args.val_split / total) * data_len * 0.9)  # 留10%余量
        test_size = int((args.test_split / total) * data_len * 0.9)
        print(f"调整后: 验证集({val_size}), 测试集({test_size})")
    
    # 使用随机种子确保可重复性
    generator = torch.Generator().manual_seed(args.seed)
    
    # 随机划分数据集
    indices = torch.randperm(data_len, generator=generator).tolist()
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size+test_size]
    
    # 创建验证集和测试集
    val_data = [raw_data[i] for i in val_indices]
    test_data = [raw_data[i] for i in test_indices]
    
    print(f"数据集划分: 总数据 {data_len}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    
    # 创建验证数据集
    print("\n创建验证数据集...")
    val_dataset = SharedGraspBinaryDataset(
        val_data,
        args.grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type=args.state_type
    )
    
    # 创建测试数据集
    print("\n创建测试数据集...")
    test_dataset = SharedGraspBinaryDataset(
        test_data,
        args.grasp_pickle_file,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
        grasp_type=args.grasp_type,
        state_type=args.state_type
    )
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 清理不再需要的变量
    del raw_data, val_data, test_data
    gc.collect()
    
    # 初始化模型
    models = {}
    model_thresholds = {}  # 存储每个模型的optimal_threshold
    
    if args.model_mode == 'standard':
        # 标准模式：加载两个模型(init和goal)
        print("\n使用标准模式（两个模型）进行评估")
        
        # 检查路径是否提供
        if args.model_init_path is None or args.model_goal_path is None:
            raise ValueError("标准模式下需要提供model_init_path和model_goal_path参数")
        
        # 计算输入维度 - 此处获取模型需要的实际输入维度
        if args.use_quaternion:
            obj_dim = 7  # 位置(3) + 四元数(4)
        else:
            obj_dim = 3  # 位置(2) + 欧拉角(1)
            
        if args.use_stable_label:
            # 获取one-hot编码的实际维度
            onehot_dim = len(test_dataset.obj_encoder.categories_[0])
            obj_dim += onehot_dim
            
        # 加上抓取姿态的维度
        grasp_dim = 7  # 抓取位姿(位置3 + 四元数4)
        input_dim = obj_dim + grasp_dim
        
        # 创建并加载模型
        models['init'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
        )
        
        models['goal'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
        )
        # 加载模型权重和阈值
        
        checkpoint_init = torch.load(args.model_init_path, map_location=device)
        models['init'].load_state_dict(checkpoint_init['model_state_dict'])
        models['init'] = models['init'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['init'] = checkpoint_init.get('optimal_threshold', 0.5)
        
        checkpoint_goal = torch.load(args.model_goal_path, map_location=device)
        models['goal'].load_state_dict(checkpoint_goal['model_state_dict'])
        models['goal'] = models['goal'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['goal'] = checkpoint_goal.get('optimal_threshold', 0.5)
        
        print(f"已加载初始状态模型: {os.path.basename(args.model_init_path)}, 阈值: {model_thresholds['init']}")
        print(f"已加载目标状态模型: {os.path.basename(args.model_goal_path)}, 阈值: {model_thresholds['goal']}")
        
    elif args.model_mode == 'extended':
        # 扩展模式：加载四个模型
        print("\n使用扩展模式（四个模型）进行评估")
        
        # 检查路径是否提供
        required_paths = [
            args.model_robot_init_path, args.model_table_init_path,
            args.model_robot_goal_path, args.model_table_goal_path
        ]
        if any(path is None for path in required_paths):
            raise ValueError("扩展模式下需要提供robot_init, table_init, robot_goal, table_goal四个模型路径")
        
        # 计算输入维度 - 与标准模式相同
        if args.use_quaternion:
            obj_dim = 7  # 位置(3) + 四元数(4)
        else:
            obj_dim = 3  # 位置(2) + 欧拉角(1)
            
        if args.use_stable_label:
            # 获取one-hot编码的实际维度
            onehot_dim = len(test_dataset.obj_encoder.categories_[0])
            obj_dim += onehot_dim
            
        # 加上抓取姿态的维度
        grasp_dim = 7  # 抓取位姿(位置3 + 四元数4)
        input_dim = obj_dim + grasp_dim
        
        # 创建四个模型
        models['robot_init'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
        
        models['table_init'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
        
        models['robot_goal'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
        
        models['table_goal'] = GraspNetwork(
            input_dim=input_dim, 
            hidden_dims=args.hidden_dims,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate
        )
        
        # 加载模型权重和阈值
        checkpoint_robot_init = torch.load(args.model_robot_init_path, map_location=device)
        models['robot_init'].load_state_dict(checkpoint_robot_init['model_state_dict'])
        models['robot_init'] = models['robot_init'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['robot_init'] = checkpoint_robot_init.get('optimal_threshold', 0.5)

        checkpoint_table_init = torch.load(args.model_table_init_path, map_location=device)
        models['table_init'].load_state_dict(checkpoint_table_init['model_state_dict'])
        models['table_init'] = models['table_init'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['table_init'] = checkpoint_table_init.get('optimal_threshold', 0.5)

        checkpoint_robot_goal = torch.load(args.model_robot_goal_path, map_location=device)
        models['robot_goal'].load_state_dict(checkpoint_robot_goal['model_state_dict'])
        models['robot_goal'] = models['robot_goal'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['robot_goal'] = checkpoint_robot_goal.get('optimal_threshold', 0.5)

        checkpoint_table_goal = torch.load(args.model_table_goal_path, map_location=device)
        models['table_goal'].load_state_dict(checkpoint_table_goal['model_state_dict'])
        models['table_goal'] = models['table_goal'].to(device).float()
        # 加载optimal_threshold，如果不存在则使用默认值0.5
        model_thresholds['table_goal'] = checkpoint_table_goal.get('optimal_threshold', 0.5)

        
        # 将模型移至设备并设置为评估模式
        models['robot_init'] = models['robot_init'].to(device).eval()
        models['table_init'] = models['table_init'].to(device).eval()
        models['robot_goal'] = models['robot_goal'].to(device).eval()
        models['table_goal'] = models['table_goal'].to(device).eval()
        
        print(f"已加载Robot初始状态模型: {os.path.basename(args.model_robot_init_path)}, 阈值: {model_thresholds['robot_init']}")
        print(f"已加载Table初始状态模型: {os.path.basename(args.model_table_init_path)}, 阈值: {model_thresholds['table_init']}")
        print(f"已加载Robot目标状态模型: {os.path.basename(args.model_robot_goal_path)}, 阈值: {model_thresholds['robot_goal']}")
        print(f"已加载Table目标状态模型: {os.path.basename(args.model_table_goal_path)}, 阈值: {model_thresholds['table_goal']}")
    

    # 初始化wandb
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # 评估模型性能
    print("\n评估模型性能...")
    
    # 根据是否使用验证集优化阈值选择评估方法
    if args.use_optimal_threshold:
        # 使用验证集优化阈值
        optimal_threshold, best_results, threshold_results = test_AND_model_with_optimal_threshold(
            models, val_loader, test_loader, device, mode=args.model_mode
        )
        
        # 记录最终结果
        if not args.no_wandb:
            if args.model_mode == "standard" and not isinstance(optimal_threshold, dict):
                # 标准模式下记录单一阈值
                wandb.log({
                    "final_optimal_threshold": optimal_threshold,
                    "final_accuracy": best_results['accuracy'],
                    "final_precision": best_results['binary_precision'],
                    "final_recall": best_results['binary_recall'],
                    "final_f1": best_results['binary_f1']
                })
            else:
                # 扩展模式下记录阈值字典
                wandb.log({
                    "final_optimal_thresholds": optimal_threshold,
                    "final_accuracy": best_results['accuracy'],
                    "final_precision": best_results['binary_precision'],
                    "final_recall": best_results['binary_recall'],
                    "final_f1": best_results['binary_f1']
                })
        
        # 将结果保存到文件
        results_summary = {
            'optimal_threshold': optimal_threshold,
            'best_results': best_results,
            'threshold_results': threshold_results,
            'config': vars(args)
        }
    else:
        # 使用模型自身阈值
        best_threshold, best_results, threshold_results = test_AND_model_with_dataset(
            models, test_loader, device, mode=args.model_mode, model_thresholds=model_thresholds
        )
        
        # 输出结果
        print("\n测试集上的结果:")
        print(f"使用模型自身最佳阈值")
        print(f"准确率: {best_results['accuracy']:.2f}%")
        print(f"Binary_Precision: {best_results['binary_precision']:.4f}")
        print(f"Binary_Recall: {best_results['binary_recall']:.4f}")
        print(f"Binary_F1 Score: {best_results['binary_f1']:.4f}")
        
        # 记录最终结果
        if not args.no_wandb:
            if isinstance(best_threshold, str):
                threshold_value = best_threshold  # 如果是字符串，直接使用
            else:
                threshold_value = float(best_threshold)  # 否则转换为浮点数
                
            wandb.log({
                "final_best_threshold": threshold_value,
                "final_accuracy": best_results['accuracy'],
                "final_precision": best_results['binary_precision'],
                "final_recall": best_results['binary_recall'],
                "final_f1": best_results['binary_f1']
            })
        
        # 将结果保存到文件
        results_summary = {
            'best_threshold': best_threshold,
            'best_results': best_results,
            'threshold_results': threshold_results,
            'config': vars(args)
        }
    
    # 创建输出目录
    os.makedirs('output/BC_AND_evaluation', exist_ok=True)
    
    # 保存结果
    result_file = f'output/BC_AND_evaluation/{args.wandb_name}_results.pickle'
    with open(result_file, 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\n结果已保存到 {result_file}")


if __name__ == '__main__':
    main()