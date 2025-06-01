import os
import sys
sys.path.append("E:/Qin/wrs")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import numpy as np
import wandb
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R
import argparse
import random
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import gc
import wrs.basis.robot_math as rm
import re

# class SharedGraspEnergyDataset(Dataset):
#     def __init__(self, data, grasp_pickle_file, use_quaternion=False, use_stable_label=True):
#         """
#         Args:
#             data: 数据列表，每个item包含 [[init_pos, init_rotmat], [goal_pos, goal_rotmat], init_stable_id, goal_stable_id, common_id]
#             grasp_pickle_file: 抓取候选文件路径
#             use_quaternion: 是否使用四元数表示旋转。如果为False，则使用简化的(x,y,rz)表示
#             use_stable_label: 是否使用stable_label作为特征。注意：当use_quaternion=False时必须为True
#         """
#         # 参数验证
#         if not use_quaternion and not use_stable_label:
#             raise ValueError("非四元数表示(use_quaternion=False)必须使用stable_label")
            
#         self.use_quaternion = use_quaternion
#         self.use_stable_label = use_stable_label
#         self.data = data
        
#         # 加载抓取候选
#         with open(grasp_pickle_file, 'rb') as f:
#             grasp_candidates = pickle.load(f)
            
#         # 预处理抓取位姿数据 - 始终使用7维表示(pos + quaternion)
#         grasp_poses = np.array([
#             np.concatenate([
#                 np.array(grasp.ac_pos, dtype=np.float32).flatten(),
#                 R.from_matrix(grasp.ac_rotmat).as_quat()
#             ]) for grasp in grasp_candidates
#         ], dtype=np.float32)
#         self.grasp_poses = torch.from_numpy(grasp_poses.copy())
#         del grasp_poses, grasp_candidates
        
#         # 只有在使用stable_label时才创建编码器
#         if self.use_stable_label:
#             # 创建物体stable_id的One-Hot编码器
#             init_types = [item[4] for item in data]  # init_stable_id
#             goal_types = [item[9] for item in data]  # goal_stable_id
#             all_types = list(set(init_types + goal_types))
#             self.obj_encoder = OneHotEncoder(sparse_output=False)
#             self.obj_encoder.fit(np.array(all_types).reshape(-1, 1))
#             del init_types, goal_types, all_types
        
#         self.prepare_data()
    
#     def _normalize_angle(self, angle):
#         """将角度从[-π, π]归一化到[0, 1]范围"""
#         angle = (angle + np.pi) % (2 * np.pi) - np.pi
#         return (angle + np.pi) / (2 * np.pi)
    
#     def _convert_rotation(self, rotmat):
#         """转换旋转矩阵为四元数或归一化的欧拉角"""
#         if self.use_quaternion:
#             return R.from_matrix(rotmat).as_quat()
#         else:
#             # 如果不使用四元数，只返回z轴旋转角度并归一化
#             euler = R.from_matrix(rotmat).as_euler('zxy', degrees=False)
#             return np.array([self._normalize_angle(euler[0])], dtype=np.float32)
    
#     def _get_position(self, pos):
#         """根据表示方式返回位置信息"""
#         if self.use_quaternion:
#             return pos.copy()  # 返回完整的xyz
#         else:
#             return pos[:2].copy()  # 只返回xy
        
#     def prepare_data(self):
#         """准备训练数据"""
#         # 预先计算总样本数
#         total_samples = len(self.data) * len(self.grasp_poses)
        
#         # 计算特征维度
#         if self.use_quaternion:
#             pose_dim = 7  # pos(3) + quaternion(4)
#         else:
#             pose_dim = 3  # pos(2) + normalized_rz(1)
            
#         grasp_pose_dim = 7  # 抓取位姿始终使用7维表示
        
#         # 根据是否使用stable_label确定特征维度
#         if self.use_stable_label:
#             onehot_dim = len(self.obj_encoder.get_feature_names_out())
#             feature_dim = pose_dim * 2 + onehot_dim * 2 + grasp_pose_dim
#         else:
#             feature_dim = pose_dim * 2 + grasp_pose_dim
        
#         # 预分配数组
#         all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)
#         all_labels = np.zeros(total_samples, dtype=np.float32)
        
#         # 批量处理数据
#         current_idx = 0
#         for item in self.data:
#             # 处理初始位姿
#             init_pos = self._get_position(np.array(item[0][0], dtype=np.float32))
#             init_rot = self._convert_rotation(np.array(item[0][1], dtype=np.float32))
#             init_pose = np.concatenate([init_pos, init_rot])
            
#             # 处理目标位姿
#             goal_pos = self._get_position(np.array(item[5][0], dtype=np.float32))
#             goal_rot = self._convert_rotation(np.array(item[5][1], dtype=np.float32))
#             goal_pose = np.concatenate([goal_pos, goal_rot])
            
#             # 计算当前样本范围
#             n_samples = len(self.grasp_poses)
#             end_idx = current_idx + n_samples
            
#             # 填充特征数组
#             feature_start = 0
            
#             # 添加初始位姿
#             all_features[current_idx:end_idx, feature_start:feature_start+len(init_pose)] = init_pose
#             feature_start += len(init_pose)

#             # 添加目标位姿
#             all_features[current_idx:end_idx, feature_start:feature_start+len(goal_pose)] = goal_pose
#             feature_start += len(goal_pose)
            
#             # 如果使用stable_label，添加One-Hot编码
#             if self.use_stable_label:
#                 init_type_onehot = self.obj_encoder.transform([[item[4]]]).copy()
#                 all_features[current_idx:end_idx, feature_start:feature_start+len(init_type_onehot[0])] = init_type_onehot
#                 feature_start += len(init_type_onehot[0])


#             if self.use_stable_label:
#                 goal_type_onehot = self.obj_encoder.transform([[item[9]]]).copy()
#                 all_features[current_idx:end_idx, feature_start:feature_start+len(goal_type_onehot[0])] = goal_type_onehot
#                 feature_start += len(goal_type_onehot[0])
                
#                 del init_type_onehot, goal_type_onehot
            
#             # 添加抓取位姿
#             all_features[current_idx:end_idx, feature_start:] = self.grasp_poses.numpy()
            
#             # 设置标签
#             if item[-1]:  # common_id
#                 all_labels[current_idx:end_idx][item[-1]] = 1
            
#             # 清理临时变量
#             del init_pos, init_rot, init_pose, goal_pos, goal_rot, goal_pose
#             current_idx = end_idx
        
#         # 转换为tensor并确保数据独立
#         self.all_features = torch.from_numpy(all_features.copy())
#         self.all_labels = torch.from_numpy(all_labels.copy())
        
#         # 清理中间变量
#         del all_features, all_labels
#         gc.collect()
    
#     def __len__(self):
#         return len(self.all_features)
    
#     def __getitem__(self, idx):
#         return self.all_features[idx], self.all_labels[idx]


# class SharedGraspEnergyDataset(Dataset):
#     def __init__(self, data, grasp_pickle_file, use_quaternion=True, use_stable_label=True):
#         """
#         Args:
#             data: 数据列表，每个item包含
#             [[init_pos, init_rotmat], init_available_gids_robot_table, init_available_gids_table, init_available_gids_robot_without_table, init_stable_id,
#             [goal_pos, goal_rotmat], goal_available_gids_robot_table, goal_available_gids_table, goal_available_gids_robot_without_table, goal_stable_id,
#             common_id]
#             grasp_pickle_file: 抓取候选文件路径
#             use_quaternion: 是否使用四元数表示旋转。如果为False，则使用简化的(x,y,rz)表示
#             use_stable_label: 是否使用stable_label作为特征。注意：当use_quaternion=False时必须为True
#         """
#         # 参数验证
#         if not use_quaternion and not use_stable_label:
#             raise ValueError("非四元数表示(use_quaternion=False)必须使用stable_label")
            
#         self.use_quaternion = use_quaternion
#         self.use_stable_label = use_stable_label
#         self.data = data
        
#         # 加载抓取候选
#         with open(grasp_pickle_file, 'rb') as f:
#             grasp_candidates = pickle.load(f)
            
#         # 预处理抓取位姿数据 - 始终使用7维表示(pos + quaternion)
#         grasp_poses = np.array([
#             np.concatenate([
#                 np.array(grasp.ac_pos, dtype=np.float32).flatten(),
#                 R.from_matrix(grasp.ac_rotmat).as_quat()
#             ]) for grasp in grasp_candidates
#         ], dtype=np.float32)
#         self.grasp_poses = torch.from_numpy(grasp_poses.copy())
#         del grasp_poses, grasp_candidates
        
#         # 只有在使用stable_label时才创建编码器
#         if self.use_stable_label:
#             # 创建物体stable_id的One-Hot编码器 
#             init_types = [item[4] for item in data]  # init_stable_id
#             goal_types = [item[9] for item in data]  # goal_stable_id
#             all_types = list(set(init_types + goal_types))
#             self.obj_encoder = OneHotEncoder(sparse_output=False)
#             self.obj_encoder.fit(np.array(all_types).reshape(-1, 1))
#             del init_types, goal_types, all_types
        
#         self.prepare_data()
    
#     def _normalize_angle(self, angle):
#         """将角度从[-π, π]归一化到[0, 1]范围"""
#         angle = (angle + np.pi) % (2 * np.pi) - np.pi
#         return (angle + np.pi) / (2 * np.pi)
    
#     def _convert_rotation(self, rotmat):
#         """转换旋转矩阵为四元数或归一化的欧拉角"""
#         if self.use_quaternion:
#             return R.from_matrix(rotmat).as_quat()
#         else:
#             # 如果不使用四元数，只返回z轴旋转角度并归一化
#             euler = R.from_matrix(rotmat).as_euler('zxy', degrees=False)
#             return np.array([self._normalize_angle(euler[0])], dtype=np.float32)
    
#     def _get_position(self, pos):
#         """根据表示方式返回位置信息"""
#         if self.use_quaternion:
#             return pos.copy()  # 返回完整的xyz
#         else:
#             return pos[:2].copy()  # 只返回xy
        
#     def prepare_data(self):
#         """准备训练数据"""
#         # 预先计算总样本数
#         total_samples = len(self.data) * len(self.grasp_poses)
        
#         # 计算特征维度
#         if self.use_quaternion:
#             pose_dim = 7  # pos(3) + quaternion(4)
#         else:
#             pose_dim = 3  # pos(2) + normalized_rz(1)
            
#         grasp_pose_dim = 7  # 抓取位姿始终使用7维表示
        
#         # 根据是否使用stable_label确定特征维度
#         if self.use_stable_label:
#             onehot_dim = len(self.obj_encoder.get_feature_names_out())
#             feature_dim = pose_dim * 2 + onehot_dim * 2 + grasp_pose_dim
#         else:
#             feature_dim = pose_dim * 2 + grasp_pose_dim
        
#         # 预分配数组
#         all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)
#         all_labels = np.zeros(total_samples, dtype=np.float32)
        
#         # 批量处理数据
#         current_idx = 0
#         for item in self.data:
#             # 处理初始位姿
#             init_pos = self._get_position(np.array(item[0][0], dtype=np.float32))
#             init_rot = self._convert_rotation(np.array(item[0][1], dtype=np.float32))
#             init_pose = np.concatenate([init_pos, init_rot])
            
#             # 处理目标位姿
#             goal_pos = self._get_position(np.array(item[5][0], dtype=np.float32))
#             goal_rot = self._convert_rotation(np.array(item[5][1], dtype=np.float32))
#             goal_pose = np.concatenate([goal_pos, goal_rot])
            
#             # 计算当前样本范围
#             n_samples = len(self.grasp_poses)
#             end_idx = current_idx + n_samples
            
#             # 填充特征数组
#             feature_start = 0
            
#             # 添加初始位姿 7 维
#             all_features[current_idx:end_idx, feature_start:feature_start+len(init_pose)] = init_pose
#             feature_start += len(init_pose)

#             # 添加目标位姿 - 7 维
#             all_features[current_idx:end_idx, feature_start:feature_start + len(goal_pose)] = goal_pose
#             feature_start += len(goal_pose)

#             if self.use_stable_label:
#                 init_type_onehot = self.obj_encoder.transform([[item[4]]]).copy()
#                 all_features[current_idx:end_idx, feature_start:feature_start+len(init_type_onehot[0])] = init_type_onehot
#                 feature_start += len(init_type_onehot[0])
#                 del init_type_onehot

#             # 如果使用stable_label，添加目标状态的One-Hot编码
#             if self.use_stable_label:
#                 goal_type_onehot = self.obj_encoder.transform([[item[9]]]).copy()
#                 all_features[current_idx:end_idx, feature_start:feature_start+len(goal_type_onehot[0])] = goal_type_onehot
#                 feature_start += len(goal_type_onehot[0])
#                 del goal_type_onehot
            
#             # 添加抓取位姿
#             all_features[current_idx:end_idx, feature_start:] = self.grasp_poses.numpy()
            
#             # 设置标签
#             if item[-1]:  # common_id
#                 all_labels[current_idx:end_idx][item[-1]] = 1
            
#             # 清理临时变量
#             del init_pos, init_rot, init_pose, goal_pos, goal_rot, goal_pose
#             current_idx = end_idx
        
#         # 转换为tensor并确保数据独立
#         self.all_features = torch.from_numpy(all_features.copy())
#         self.all_labels = torch.from_numpy(all_labels.copy())
        
#         # 清理中间变量
#         del all_features, all_labels
#         gc.collect()
    
#     def __len__(self):
#         return len(self.all_features)
    
#     def __getitem__(self, idx):
#         return self.all_features[idx], self.all_labels[idx]   
    

class SharedGraspEnergyDataSynthesizeDataset(Dataset):
    def __init__(self, data, grasp_pickle_file, use_quaternion=False, use_stable_label=True):
        """
        Args:
            data: 数据列表，每个item包含 [[init_pos, init_rotmat], init_feasible_grasp_id, init_stable_id, [goal_pos, goal_rotmat], shared_grasp_id]
            grasp_pickle_file: 抓取候选文件路径
            use_quaternion: 是否使用四元数表示旋转。如果为False，则使用简化的(x,y,rz)表示
            use_stable_label: 是否使用stable_label作为特征。注意：当use_quaternion=False时必须为True
        """
        # 参数验证
        if not use_quaternion and not use_stable_label:
            raise ValueError("非四元数表示(use_quaternion=False)必须使用stable_label")
            
        self.use_quaternion = use_quaternion
        self.use_stable_label = use_stable_label
        self.data = data
        
       # 加载所有抓取候选并合并
        all_grasp_candidates = []
        grasp_files_list = grasp_pickle_file[0].split(',')
        
        for grasp_file in grasp_files_list:
            with open(grasp_file, 'rb') as f:
                grasp_candidates = pickle.load(f)
                all_grasp_candidates.extend(grasp_candidates)
                print(f"从 {os.path.basename(grasp_file)} 加载了 {len(grasp_candidates)} 个抓取候选")
            
        # 预处理抓取位姿数据 - 现在是8维表示(pos + quaternion + normalized_width)
        grasp_poses = np.array([
            np.concatenate([
                np.array(grasp.ac_pos, dtype=np.float32).flatten(),
                R.from_matrix(grasp.ac_rotmat).as_quat()
            ]) for grasp in all_grasp_candidates
        ], dtype=np.float32)
        
        print(f"合并后总抓取候选数量: {len(grasp_poses)}")
        self.grasp_poses = torch.from_numpy(grasp_poses.copy())
        del grasp_poses, all_grasp_candidates
        
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
        
    def prepare_data(self):
        """准备训练数据"""
        # 预先计算总样本数
        total_samples = len(self.data) * len(self.grasp_poses)
        
        # 计算特征维度
        if self.use_quaternion:
            pose_dim = 7  # pos(3) + quaternion(4)
        else:
            pose_dim = 3  # pos(2) + normalized_rz(1)
            
        grasp_pose_dim = 7  # 抓取位姿始终使用7维表示
        
        # 根据是否使用stable_label确定特征维度
        if self.use_stable_label:
            onehot_dim = len(self.obj_encoder.get_feature_names_out())
            feature_dim = pose_dim * 2 + onehot_dim * 2 + grasp_pose_dim
        else:
            feature_dim = pose_dim * 2 + grasp_pose_dim
        
        # 预分配数组
        all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)
        all_labels = np.zeros(total_samples, dtype=np.float32)
        
        # 批量处理数据
        current_idx = 0
        for item in self.data:
            # 处理初始位姿
            init_pos = self._get_position(np.array(item[0][0], dtype=np.float32))
            init_rot = self._convert_rotation(np.array(item[0][1], dtype=np.float32))
            init_pose = np.concatenate([init_pos, init_rot])
            
            # 处理目标位姿
            goal_pos = self._get_position(np.array(item[3][0], dtype=np.float32))
            goal_rot = self._convert_rotation(np.array(item[3][1], dtype=np.float32))
            goal_pose = np.concatenate([goal_pos, goal_rot])
            
            # 计算当前样本范围
            n_samples = len(self.grasp_poses)
            end_idx = current_idx + n_samples
            
            # 填充特征数组
            feature_start = 0
            
            # 添加初始位姿
            all_features[current_idx:end_idx, feature_start:feature_start+len(init_pose)] = init_pose
            feature_start += len(init_pose)

            # 添加目标位姿
            all_features[current_idx:end_idx, feature_start:feature_start+len(goal_pose)] = goal_pose
            feature_start += len(goal_pose)
            
            # 如果使用stable_label，添加One-Hot编码
            if self.use_stable_label:
                init_type_onehot = self.obj_encoder.transform([[item[4]]]).copy()
                all_features[current_idx:end_idx, feature_start:feature_start+len(init_type_onehot[0])] = init_type_onehot
                feature_start += len(init_type_onehot[0])


            if self.use_stable_label:
                goal_type_onehot = self.obj_encoder.transform([[item[9]]]).copy()
                all_features[current_idx:end_idx, feature_start:feature_start+len(goal_type_onehot[0])] = goal_type_onehot
                feature_start += len(goal_type_onehot[0])
                
                del init_type_onehot, goal_type_onehot
            
            # 添加抓取位姿
            all_features[current_idx:end_idx, feature_start:] = self.grasp_poses.numpy()
            
            # 设置标签
            if item[-1]:  # common_id
                all_labels[current_idx:end_idx][item[-1]] = 1
            
            # 清理临时变量
            del init_pos, init_rot, init_pose, goal_pos, goal_rot, goal_pose
            current_idx = end_idx
        
        # 转换为tensor并确保数据独立
        self.all_features = torch.from_numpy(all_features.copy())
        self.all_labels = torch.from_numpy(all_labels.copy())
        
        # 清理中间变量
        del all_features, all_labels
        gc.collect()
    
    def __len__(self):
        return len(self.all_features)
    
    def __getitem__(self, idx):
        return self.all_features[idx], self.all_labels[idx]
   

class GraspEnergyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, num_layers=3, dropout_rate=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        # 确保hidden_dims是列表且长度等于num_layers
        if len(hidden_dims) != num_layers:
            hidden_dims = [hidden_dims[0]] * num_layers
        
        # 构建MLP层
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.SELU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # 创建模型
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # SELU激活函数推荐的初始化方法
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


class EnergyBasedLoss(nn.Module):
    def __init__(self, temperature=1.0, ml_coeff=1.0, l2_coeff=0.2):
        super().__init__()
        self.temperature = temperature
        self.ml_coeff = ml_coeff
        self.l2_coeff = l2_coeff

    def forward(self, energies, labels):
        """
        Args:
            energies: 模型输出的能量值 [batch_size, 1]
            labels: 二值标签 [batch_size]
        Returns:
            loss: 标量损失值
        """
        # 分离正样本和负样本
        pos_mask = labels.bool()
        neg_mask = ~pos_mask

        pos_energies = energies[pos_mask]
        neg_energies = energies[neg_mask]

        # 1. 对数配分函数 logZ（考虑所有样本）
        log_Z = torch.logsumexp(-energies.squeeze() / self.temperature, dim=0)

        # 2. 正样本的负对数似然损失
        pos_log_likelihood = -pos_energies.mean() / self.temperature - log_Z
        nll_loss = -pos_log_likelihood

        # 3. 对比散度损失（类似 contrastive divergence）
        contrastive_loss = (pos_energies.mean() - neg_energies.mean()) / self.temperature

        # 4. L2 正则化项（对正负样本分别正则化）
        l2_reg = self.l2_coeff * (pos_energies.pow(2).mean() + neg_energies.pow(2).mean())

        # 5. 总损失（结合最大似然和正则化）
        total_loss = self.ml_coeff * (nll_loss + contrastive_loss) + l2_reg

        return total_loss


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 获取所有标签
        self.labels = torch.tensor([sample[1].item() for sample in dataset])
        
        # 计算数据集中的实际正样本比例
        self.actual_pos_ratio = float(torch.sum(self.labels == 1)) / len(self.labels)
        
        # 使用实际正样本比例来确定每个batch中的正样本数量
        self.pos_samples_per_batch = int(batch_size * self.actual_pos_ratio)
        self.neg_samples_per_batch = batch_size - self.pos_samples_per_batch
        
        # 分离正负样本的索引
        self.pos_indices = torch.where(self.labels == 1)[0]
        self.neg_indices = torch.where(self.labels == 0)[0]
        
        # 确保所有索引都是长整型
        self.pos_indices = self.pos_indices.long()
        self.neg_indices = self.neg_indices.long()
        
        # 打印采样器信息
        print(f"\n采样器信息:")
        print(f"数据集大小: {len(dataset)}")
        print(f"正样本数量: {len(self.pos_indices)}")
        print(f"负样本数量: {len(self.neg_indices)}")
        print(f"实际正样本比例: {self.actual_pos_ratio:.3f}")
        print(f"每个batch中的正样本数量: {self.pos_samples_per_batch}")
        print(f"每个batch中的负样本数量: {self.neg_samples_per_batch}")
        
        # 计算能够构建的完整batch数量
        self.num_batches = min(
            len(self.pos_indices) // self.pos_samples_per_batch,
            len(self.neg_indices) // self.neg_samples_per_batch
        )
        
        # 添加数据量检查
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise RuntimeError("正样本或负样本数量为0, 无法进行平衡采样")
            
        # 使用replacement来处理样本不足的情况
        self.use_replacement = (len(self.pos_indices) < self.pos_samples_per_batch or 
                              len(self.neg_indices) < self.neg_samples_per_batch)
    
    def __iter__(self):
        # 根据是否需要replacement选择不同的采样策略
        if self.use_replacement:
            for _ in range(self.num_batches):
                # 有放回采样
                batch_pos = self.pos_indices[torch.randint(len(self.pos_indices), 
                          (self.pos_samples_per_batch,))]
                batch_neg = self.neg_indices[torch.randint(len(self.neg_indices), 
                          (self.neg_samples_per_batch,))]
                
                # 合并并打乱
                batch_indices = torch.cat([batch_pos, batch_neg])
                batch_indices = batch_indices[torch.randperm(len(batch_indices))]
                
                yield batch_indices
        else:
            # 无放回采样
            pos_indices = self.pos_indices[torch.randperm(len(self.pos_indices))]
            neg_indices = self.neg_indices[torch.randperm(len(self.neg_indices))]
            
            for i in range(self.num_batches):
                batch_pos = pos_indices[i * self.pos_samples_per_batch : 
                                      (i + 1) * self.pos_samples_per_batch]
                batch_neg = neg_indices[i * self.neg_samples_per_batch : 
                                      (i + 1) * self.neg_samples_per_batch]
                
                batch_indices = torch.cat([batch_pos, batch_neg])
                batch_indices = batch_indices[torch.randperm(len(batch_indices))]
                
                yield batch_indices
    
    def __len__(self):
        return self.num_batches


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_path, early_stop_patience=50):
    model.to(device)
    best_val_metric = 0
    patience_counter = 0
    best_threshold = 0  # 初始化最佳阈值
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_energies_list = []
        train_labels_list = []
        batch_count = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            energies = model(inputs)
            loss = criterion(energies, labels)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()  # 保存当前loss值
            epoch_loss += current_loss
            batch_count += 1
            
            # 收集训练数据时立即转为numpy并释放tensor
            train_energies_list.append(energies.detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())
            
            # 及时清理GPU内存
            del energies, loss, inputs, labels
            
            train_loop.set_postfix({'loss': current_loss})
        
        # 计算训练指标
        train_energies = np.concatenate(train_energies_list)
        train_labels = np.concatenate(train_labels_list)
        train_metrics = calculate_metrics(train_energies, train_labels)
        avg_train_loss = epoch_loss / batch_count

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_energies_list = []
        val_labels_list = []
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for val_inputs, val_labels in val_loop:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_energies = model(val_inputs)
                current_val_loss = criterion(val_energies, val_labels).item()
                val_loss += current_val_loss
                
                # 收集验证数据时立即转为numpy并释放tensor
                val_energies_list.append(val_energies.cpu().numpy())
                val_labels_list.append(val_labels.cpu().numpy())
                
                # 及时清理GPU内存
                del val_energies, val_inputs, val_labels
                
                val_loop.set_postfix({'loss': current_val_loss})
        
        # 计算验证指标
        val_energies = np.concatenate(val_energies_list)
        val_labels = np.concatenate(val_labels_list)
        val_metrics = calculate_metrics(val_energies, val_labels)
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 记录详细指标到wandb
        wandb.log({
            # 基础训练指标
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            
            # 训练集详细指标
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics['precision'],
            "train/recall": train_metrics['recall'],
            "train/f1": train_metrics['f1'],
            "train/auc": train_metrics['auc'],
            "train/pos_energy_mean": train_metrics['pos_energy_mean'],
            "train/pos_energy_std": train_metrics['pos_energy_std'],
            "train/neg_energy_mean": train_metrics['neg_energy_mean'],
            "train/neg_energy_std": train_metrics['neg_energy_std'],
            "train/energy_gap": train_metrics['energy_gap'],
            "train/optimal_threshold": train_metrics['optimal_threshold'],
            
            # 验证集详细指标
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "val/f1": val_metrics['f1'],
            "val/auc": val_metrics['auc'],
            "val/pos_energy_mean": val_metrics['pos_energy_mean'],
            "val/pos_energy_std": val_metrics['pos_energy_std'],
            "val/neg_energy_mean": val_metrics['neg_energy_mean'],
            "val/neg_energy_std": val_metrics['neg_energy_std'],
            "val/energy_gap": val_metrics['energy_gap'],
            "val/optimal_threshold": val_metrics['optimal_threshold']
        })

        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Recall:{train_metrics['recall']:.4f}, Precision:{train_metrics['precision']:.4f},"
              f" F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Recall:{val_metrics['recall']:.4f}, Precision:{val_metrics['precision']:.4f},"
              f" F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"Energy Gap - Train: {train_metrics['energy_gap']:.4f}, Val: {val_metrics['energy_gap']:.4f}")
        print(f"Optimal Threshold - Train: {train_metrics['optimal_threshold']:.4f}, Val: {val_metrics['optimal_threshold']:.4f}")

        # 验证阶段后的早停检查
        if val_metrics['f1'] > best_val_metric:  # 使用F1分数作为早停标准
            best_val_metric = val_metrics['f1']
            best_threshold = val_metrics['optimal_threshold']  # 保存最佳阈值
            patience_counter = 0
            
            # 保存模型和验证集确定的最佳阈值
            save_info = {
                'model_state_dict': model.state_dict(),
                'best_val_metric': best_val_metric,
                'epoch': epoch,
                'optimal_threshold': best_threshold,  # 保存验证集确定的最佳阈值
                'val_metrics': val_metrics,  # 保存完整的验证集指标
                'train_metrics': train_metrics  # 保存完整的训练集指标(可选)
            }
            print(f"\n保存最佳模型 - 验证集F1: {best_val_metric:.4f}, 最佳阈值: {best_threshold:.4f}")
            torch.save(save_info, save_path)
            
            # 额外分析：使用验证集阈值在训练集上的表现
            train_with_val_threshold = calculate_metrics_with_fixed_threshold(
                train_energies, train_labels, best_threshold)
            print(f"使用验证集阈值在训练集上的F1: {train_with_val_threshold['f1']:.4f}")
            
            # 记录到wandb
            wandb.log({
                "val/best_val_f1": best_val_metric,
                "val/best_threshold": best_threshold,
                "train/train_f1_with_val_threshold": train_with_val_threshold['f1']
            })
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n早停: 验证集F1分数 {early_stop_patience} 个epoch没有改善")
            break

        # 清理内存
        del train_energies_list, train_labels_list, train_energies, train_labels
        del val_energies_list, val_labels_list, val_energies, val_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    # 训练结束后，加载最佳模型并返回
    best_model_info = torch.load(save_path)
    model.load_state_dict(best_model_info['model_state_dict'])
    print(f"\n训练完成 - 加载最佳模型(Epoch {best_model_info['epoch']})")
    print(f"最佳验证集F1: {best_model_info['best_val_metric']:.4f}")
    print(f"最佳阈值: {best_model_info['optimal_threshold']:.4f}")
    
    return model, best_model_info['optimal_threshold']


def calculate_metrics_with_fixed_threshold(energies, labels, threshold):
    """使用固定阈值计算评估指标"""
    # 确保输入是连续的numpy数组
    energies = np.ascontiguousarray(energies).flatten()
    labels = np.ascontiguousarray(labels).flatten()
    
    # 使用固定阈值进行预测
    predictions = (energies <= threshold)
    
    # 计算指标
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_metrics(energies, labels):
    """计算评估指标并找到最佳阈值"""
    # 确保输入是连续的numpy数组
    energies = np.ascontiguousarray(energies).flatten()
    labels = np.ascontiguousarray(labels).flatten()
    
    # 分离正负样本能量
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    pos_energies = energies[pos_mask]
    neg_energies = energies[neg_mask]
    
    # 计算能量统计量
    pos_energy_mean = np.mean(pos_energies) if len(pos_energies) > 0 else 0
    pos_energy_std = np.std(pos_energies) if len(pos_energies) > 0 else 0
    neg_energy_mean = np.mean(neg_energies) if len(neg_energies) > 0 else 0
    neg_energy_std = np.std(neg_energies) if len(neg_energies) > 0 else 0
    energy_gap = neg_energy_mean - pos_energy_mean
    
    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(labels, -energies)  # 注意这里用-energies，因为能量越低越可能是正样本
    roc_auc = auc(fpr, tpr)
    
    # 找到最佳阈值（使F1分数最大化）
    precisions, recalls, thresholds_pr = precision_recall_curve(labels, -energies)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = -thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0
    
    # 使用最佳阈值计算预测结果
    predictions = (energies <= best_threshold)
    
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    # 计算各种指标
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'pos_energy_mean': pos_energy_mean,
        'pos_energy_std': pos_energy_std,
        'neg_energy_mean': neg_energy_mean,
        'neg_energy_std': neg_energy_std,
        'energy_gap': energy_gap,
        'optimal_threshold': best_threshold
    }


def evaluate_model(model, test_loader, device, args):
    """评估模型性能"""
    # 加载保存的模型信息
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_threshold = checkpoint['optimal_threshold']  # 获取验证集确定的最佳阈值
    
    print(f"加载模型 - 最佳F1: {checkpoint['best_val_metric']:.4f}, Epoch: {checkpoint['epoch']}")
    print(f"使用验证集确定的最佳阈值: {val_threshold:.4f}")
    
    model = model.to(device)
    model.eval()
    test_energies = []
    test_labels = []
    
    # 收集所有预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs = inputs.to(device)
            energies = model(inputs).cpu().numpy().flatten()
            test_energies.extend(energies)
            test_labels.extend(labels.numpy().flatten())
    
    test_energies = np.array(test_energies)
    test_labels = np.array(test_labels)
    
    # 使用验证集阈值评估测试集
    test_metrics_with_val_threshold = calculate_metrics_with_fixed_threshold(
        test_energies, test_labels, val_threshold)
    
    print("\n=== 使用验证集阈值评估测试集 ===")
    print(f"阈值: {val_threshold:.4f}")
    print(f"Accuracy: {test_metrics_with_val_threshold['accuracy']:.4f}")
    print(f"Precision: {test_metrics_with_val_threshold['precision']:.4f}")
    print(f"Recall: {test_metrics_with_val_threshold['recall']:.4f}")
    print(f"F1 Score: {test_metrics_with_val_threshold['f1']:.4f}")
    
    # 仅供参考：计算测试集自身的最佳阈值和性能
    test_metrics = calculate_metrics(test_energies, test_labels)
    
    print("\n=== 测试集自身的最佳阈值(仅供参考) ===")
    print(f"最佳阈值: {test_metrics['optimal_threshold']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # 分析能量分布
    analyze_energy_distributions(test_energies, test_labels, val_threshold)
    
    return test_metrics_with_val_threshold


def analyze_energy_distributions(energies, labels, threshold):
    """分析能量分布"""
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    pos_energies = energies[pos_mask]
    neg_energies = energies[neg_mask]
    
    print("\n=== 能量分布统计 ===")
    print(f"正样本数量: {len(pos_energies)}")
    print(f"负样本数量: {len(neg_energies)}")
    print(f"正样本能量 - 均值: {np.mean(pos_energies):.4f}, 标准差: {np.std(pos_energies):.4f}")
    print(f"负样本能量 - 均值: {np.mean(neg_energies):.4f}, 标准差: {np.std(neg_energies):.4f}")
    print(f"能量差距: {np.mean(neg_energies) - np.mean(pos_energies):.4f}")
    
    # 分析阈值效果
    below_threshold_pos = np.sum(pos_energies <= threshold)
    below_threshold_neg = np.sum(neg_energies <= threshold)
    
    print(f"\n=== 阈值 {threshold:.4f} 的效果 ===")
    print(f"正样本低于阈值比例: {below_threshold_pos/len(pos_energies):.4f} ({below_threshold_pos}/{len(pos_energies)})")
    print(f"负样本低于阈值比例: {below_threshold_neg/len(neg_energies):.4f} ({below_threshold_neg}/{len(neg_energies)})")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def get_id_regex(file_path):
    match = re.search(r'_(\d+)\.pickle$', file_path)
    if match:
        return int(match.group(1))
    return None


def load_raw_data(data_paths, ratio=0.5, data_range=None):
    """加载多个数据集并合并（使用numpy优化）"""
    all_raw_data = []
    data_path_list = data_paths[0].split(',')
    grasp_candidiate_num = 0
    
    for index, data_path in enumerate(data_path_list):
        if index != 0:
            grasp_candidiate = get_id_regex(data_path_list[index-1])
            grasp_candidiate_num += grasp_candidiate
            
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

            # if ratio is not None:
            #     start_idx = int(len(raw_data) * ratio)
            #     raw_data = raw_data[start_idx:]

            if data_range is not None:
                raw_data = raw_data[:data_range]
                print(f"加载数据: 数据总量 {len(raw_data)}")

            # 用于保证每个object的grasp_number数目一致且训练数据大小一致； 180个grasp id和 150K个feasible grasp/ shared grasp 数据；
            #  - - - - - 
           
            for item in raw_data:
                if isinstance(item[1], (list, np.ndarray)):
                    # 转换为numpy数组进行批量操作
                    indices = list(filter(lambda x: x < 180, item[1]))
                    if len(indices) == 0:
                        item[1] = indices
                        continue
                    if grasp_candidiate_num > 0:
                        indices = [x + grasp_candidiate_num for x in indices]
                    item[1] = indices
                elif isinstance(item[1], int):
                    item[1] = list(filter(lambda x: x < 180, item[1]))
                    if len(item[1]) == 0:
                        continue
                    if grasp_candidiate_num > 0:
                        item[1] = [x + grasp_candidiate_num for x in item[1]]
                        
                        
            # 随机从raw_data中选择一个样本来构造shard grasp样本 - 仅仅针对用feasible grasp数据构造shared grasp数据 (训练feasible 的时候需要注释掉)
            for item in raw_data:
                random_index = np.random.randint(0, len(raw_data))
                target_item = raw_data[random_index]
                item.extend([target_item[0]])
                item.append(list(set(item[1]) & set(target_item[1])) if item[1] is not None and target_item[1] is not None else None)
            #  - - - - - 

            all_raw_data.extend(raw_data)
            print(f"从 {os.path.basename(data_path)} 加载了 {len(raw_data)} 个样本")
            
            # 及时清理内存
            del raw_data
            gc.collect()
    
    print(f"合并后总数据量: {len(all_raw_data)}")
    return all_raw_data
    

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
    # 创建训练集
    train_dataset = SharedGraspEnergyDataSynthesizeDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )

    # 使用训练集的标准化器创建验证集和测试集
    val_dataset = SharedGraspEnergyDataSynthesizeDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )
    
    test_dataset = SharedGraspEnergyDataSynthesizeDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, args):
    """创建数据加载器"""
    train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # 添加这个参数
        prefetch_factor=2  # 添加这个参数
    )
    
    # 验证和测试加载器类似修改
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader


def calculate_input_dim(args):
    """根据参数计算输入维度"""
    # 抓取位姿始终是7维 (pos + quaternion)
    grasp_pose_dim = 7
    
    # 计算物体位姿维度
    if args.use_quaternion:
        pose_dim = 7  # pos(3) + quaternion(4)
    else:
        pose_dim = 3  # pos(2) + normalized_rz(1)
    
    # 计算稳定性标签维度
    if args.use_stable_label:
        stable_label_dim = 5  # 假设有5个稳定性类别
        return pose_dim * 2 + stable_label_dim * 2 + grasp_pose_dim
    else:
        return pose_dim * 2 + grasp_pose_dim


def setup_training(args):
    """设置模型、损失函数、优化器等训练组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 自动计算输入维度
    input_dim = calculate_input_dim(args)
    print(f"计算得到的输入维度: {input_dim}")
    
    model = GraspEnergyNetwork(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )
    
    # 初始化损失函数
    criterion = EnergyBasedLoss(temperature=args.temperature)
    
    # 初始化优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr
    )
    
    return model, criterion, optimizer, scheduler, device


def parse_args():
    parser = argparse.ArgumentParser(description='Energy-based Grasp Model')
    
    # 数据相关参数
    parser.add_argument('--data_path', nargs='+', type=str)
    parser.add_argument('--grasp_data_path', nargs='+', type=str)
    parser.add_argument('--model_save_path', type=str,
                       default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_352_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl1_grobot_table_stinit.pth')
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--pos_ratio', type=float, default=0.33)
    parser.add_argument('--data_range', type=int, default=None, help='数据采样个数')
    parser.add_argument('--data_ratio', type=float, default=None, help='数据采样比例')
    
    # 模型结构参数 - 更新为简化的MLP参数
    parser.add_argument('--input_dim', type=int, default=19)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512])
    parser.add_argument('--num_layers', type=int, default=3)  # 替换num_res_blocks
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--use_quaternion', type=int, default=1,
                       help='使用四元数表示 (1) 或简化表示 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1,
                       help='使用稳定性标签 (1) 或不使用 (0)')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    
    # 损失函数参数
    parser.add_argument('--temperature', type=float, default=0.1)

    # 学习率调度器参数
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_model', type=bool, default=True)
    parser.add_argument('--wandb_project', type=str, default='grasp_ebm')
    parser.add_argument('--wandb_name', type=str, default='grasp_random_position_bottle_robot_57_ebm_selu')
    
    args = parser.parse_args()
    # 将整数转换为布尔值
    args.use_quaternion = bool(args.use_quaternion)
    args.use_stable_label = bool(args.use_stable_label)
    
    return args


def main():
    # 解析参数并设置随机种子
    args = parse_args()
    set_seed(args.seed)
    
    # 初始化wandb
    # wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # 加载原始数据
    raw_data = load_raw_data(args.data_path, args.data_ratio, args.data_range)
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
    
    # 设置训练组件
    model, criterion, optimizer, scheduler, device = setup_training(args)
    
    if args.train_model:
    # 训练模型
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            save_path=args.model_save_path,
            early_stop_patience=args.early_stop_patience
        )
        print("开始评估模型...")
        evaluate_model(model, test_loader, device, args)
    else:
        # 评估模型
        print("开始评估模型...")
        evaluate_model(model, test_loader, device, args)


if __name__ == '__main__':
    main()
