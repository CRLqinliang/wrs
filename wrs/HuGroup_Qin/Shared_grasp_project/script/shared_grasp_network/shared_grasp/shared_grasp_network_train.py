import os
import sys
sys.path.append("E:/Qin/wrs/wrs/HuGroup_Qin/Shared_grasp_project")
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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, average_precision_score, classification_report
)
import argparse
import random, gc
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


class SharedGraspEnergyDataset(Dataset):
    def __init__(self, data, grasp_pickle_file, use_quaternion=True, use_stable_label=True):
        """
        Args:
            data: 数据列表，每个item包含
            [[init_pos, init_rotmat], init_available_gids_robot_table, init_available_gids_table, init_available_gids_robot_without_table, init_stable_id,
            [goal_pos, goal_rotmat], goal_available_gids_robot_table, goal_available_gids_table, goal_available_gids_robot_without_table, goal_stable_id,
            common_id]
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
            goal_pos = self._get_position(np.array(item[5][0], dtype=np.float32))
            goal_rot = self._convert_rotation(np.array(item[5][1], dtype=np.float32))
            goal_pose = np.concatenate([goal_pos, goal_rot])
            
            # 计算当前样本范围
            n_samples = len(self.grasp_poses)
            end_idx = current_idx + n_samples
            
            # 填充特征数组
            feature_start = 0
            
            # 添加初始位姿 7 维
            all_features[current_idx:end_idx, feature_start:feature_start+len(init_pose)] = init_pose
            feature_start += len(init_pose)

            # 添加目标位姿 - 7 维
            all_features[current_idx:end_idx, feature_start:feature_start + len(goal_pose)] = goal_pose
            feature_start += len(goal_pose)

            if self.use_stable_label:
                init_type_onehot = self.obj_encoder.transform([[item[4]]]).copy()
                all_features[current_idx:end_idx, feature_start:feature_start+len(init_type_onehot[0])] = init_type_onehot
                feature_start += len(init_type_onehot[0])
                del init_type_onehot

            # 如果使用stable_label，添加目标状态的One-Hot编码
            if self.use_stable_label:
                goal_type_onehot = self.obj_encoder.transform([[item[9]]]).copy()
                all_features[current_idx:end_idx, feature_start:feature_start+len(goal_type_onehot[0])] = goal_type_onehot
                feature_start += len(goal_type_onehot[0])
                del goal_type_onehot
            
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


class GraspNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, num_layers=3, dropout_rate=0.1):
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
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 创建模型
        self.model = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # ReLU激活函数推荐的初始化方法
                nn.init.kaiming_normal_(m.weight, nonlinearity='selu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


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
    best_threshold = 0.5  # 初始化最佳阈值
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_outputs_list = []
        train_labels_list = []
        batch_count = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # 收集训练数据
            train_outputs_list.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())
            
            del outputs, loss, inputs, labels
            
            train_loop.set_postfix({'loss': current_loss})
        
        # 计算训练指标
        train_outputs = np.concatenate(train_outputs_list)
        train_labels = np.concatenate(train_labels_list)
        train_metrics = calculate_metrics(train_outputs, train_labels)
        avg_train_loss = epoch_loss / batch_count

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_outputs_list = []
        val_labels_list = []
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for val_inputs, val_labels in val_loop:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device).float()
                val_outputs = model(val_inputs)
                current_val_loss = criterion(val_outputs.squeeze(), val_labels).item()
                val_loss += current_val_loss
                
                val_outputs_list.append(torch.sigmoid(val_outputs).cpu().numpy())
                val_labels_list.append(val_labels.cpu().numpy())
                
                del val_outputs, val_inputs, val_labels
                
                val_loop.set_postfix({'loss': current_val_loss})
        
        # 计算验证指标
        val_outputs = np.concatenate(val_outputs_list)
        val_labels = np.concatenate(val_labels_list)
        val_metrics = calculate_metrics(val_outputs, val_labels)
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics['binary_precision'],
            "train/recall": train_metrics['binary_recall'],
            "train/f1": train_metrics['binary_f1'],
            "train/auc": train_metrics['auc'],
            
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['binary_precision'],
            "val/recall": val_metrics['binary_recall'],
            "val/f1": val_metrics['binary_f1'],
            "val/auc": val_metrics['auc'],
        })

        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Recall:{train_metrics['binary_recall']:.4f}, Precision:{train_metrics['binary_precision']:.4f},"
              f" F1: {train_metrics['binary_f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Recall:{val_metrics['binary_recall']:.4f}, Precision:{val_metrics['binary_precision']:.4f},"
              f" F1: {val_metrics['binary_f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"Optimal Threshold - Train: {train_metrics['optimal_threshold']:.4f}, Val: {val_metrics['optimal_threshold']:.4f}")

        # 验证阶段后的早停检查
        if val_metrics['binary_f1'] > best_val_metric:  # 使用F1分数作为早停标准
            best_val_metric = val_metrics['binary_f1']
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
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n早停: 验证集F1分数 {early_stop_patience} 个epoch没有改善")
            break

        # 清理内存
        del train_outputs_list, train_labels_list, train_outputs, train_labels
        del val_outputs_list, val_labels_list, val_outputs, val_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    # 训练结束后，加载最佳模型并返回
    best_model_info = torch.load(save_path)
    model.load_state_dict(best_model_info['model_state_dict'])
    print(f"\n训练完成 - 加载最佳模型(Epoch {best_model_info['epoch']})")
    print(f"最佳验证集F1: {best_model_info['best_val_metric']:.4f}")
    print(f"最佳阈值: {best_model_info['optimal_threshold']:.4f}")
    
    return model, best_model_info['optimal_threshold']


def calculate_metrics_with_fixed_threshold(outputs, labels, threshold):
    """使用固定阈值计算评估指标（仅binary模式）"""
    # 确保输入是连续的numpy数组
    outputs = np.ascontiguousarray(outputs).flatten()
    labels = np.ascontiguousarray(labels).flatten()
    
    # 使用固定阈值进行预测
    predictions = (outputs <= threshold)
    
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    # 计算binary指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels)
    
    return {
        'threshold': threshold,
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': f1,
        'binary_accuracy': accuracy
    }


def calculate_metrics(outputs, labels):
    """计算二分类评估指标"""
    # 确保输入是numpy数组
    outputs = np.ascontiguousarray(outputs).flatten()
    labels = np.ascontiguousarray(labels).flatten()
    
    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(labels, outputs)
    roc_auc = auc(fpr, tpr)
    
    # 找到最佳阈值（使F1分数最大化）
    precisions, recalls, thresholds_pr = precision_recall_curve(labels, outputs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5
    
    # 使用最佳阈值计算预测结果
    predictions = (outputs >= best_threshold)
    
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
        'binary_precision': precision,
        'binary_recall': recall,
        'binary_f1': f1,
        'auc': roc_auc,
        'optimal_threshold': best_threshold
    }


def optimal_f1_threshold(models, val_loader, device):
    """直接计算最佳F1分数对应的阈值
    
    Args:
        models: 模型字典
        test_loader: 测试数据加载器
        device: 计算设备
    """
    
    # 收集模型预测
    all_labels, all_combined_energies = collect_model_predictions_with_mode(models, val_loader, device)
    
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
    print(f"最佳阈值: {best_threshold:.4f}")
    return best_threshold


def collect_model_predictions_with_mode(models, val_loader, device):

    # 将所有模型设置为评估模式
    for model in models.values():
        model.eval()

    all_labels = []
    all_combined_energies = []  # 全部能量的组合

    # 收集所有能量值
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.cpu().numpy()

            # 确定各部分的维度
            obj_pose_dim = 7 * 2
            grasp_pose_dim = 7  # 抓取位姿始终为7维

            # 计算各部分在特征中的位置
            obj_pose_end = obj_pose_dim
            grasp_pose_end = obj_pose_end + grasp_pose_dim

            # 提取各个部分的特征
            obj_pose = inputs[:, :obj_pose_end]
            grasp_pose = inputs[:, obj_pose_end:grasp_pose_end]

            # 构建初始状态和目标状态的特征输入
            input_features = torch.cat([obj_pose, grasp_pose], dim=1)

            # 标准模式：使用两个模型分别预测init和goal状态
            energies = model(input_features).cpu().numpy()
            all_combined_energies.append(energies)

            all_labels.append(labels)

    # 合并所有批次的能量值
    all_labels = np.concatenate(all_labels)
    all_combined_energies = np.concatenate(all_combined_energies)

    return all_labels, all_combined_energies


def evaluate_model(model, test_loader, device, args, threshold=None):
    """评估模型性能"""
    # 加载保存的模型信息
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if threshold is None:
        val_threshold = checkpoint['optimal_threshold']  # 获取验证集确定的最佳阈值
    else:
        val_threshold = threshold
    
    model = model.to(device)
    model.eval()
    test_outputs = []
    test_labels = []
    
    # 收集所有预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs)).cpu().numpy()
            test_outputs.extend(outputs)
            test_labels.extend(labels.numpy())
    
    test_outputs = np.array(test_outputs)
    test_labels = np.array(test_labels)
    
    # 使用验证集阈值评估测试集
    test_metrics_val_threshold = calculate_metrics_with_fixed_threshold(test_outputs, test_labels, val_threshold)

    print("\n=== 测试集评估结果（使用验证集阈值） ===")
    print(f"阈值: {val_threshold:.4f}")
    print(f"Binary Precision: {test_metrics_val_threshold['binary_precision']:.4f}")
    print(f"Binary Recall: {test_metrics_val_threshold['binary_recall']:.4f}")
    print(f"Binary F1 Score: {test_metrics_val_threshold['binary_f1']:.4f}")

    return test_metrics_val_threshold


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
    

def get_model(args):
    """根据参数选择网络架构"""
    input_dim = calculate_input_dim(args)
    return GraspNetwork(input_dim=input_dim, output_dim=args.output_dim, 
                        hidden_dims=args.hidden_dims, num_layers=args.num_layers, dropout_rate=args.dropout_rate)


def load_raw_data(data_path, ratio=0.5, data_range=None):
    """加载原始数据
    Args:
        data_path: 数据文件路径
        ratio: 使用数据的比例，默认0.5表示使用后半部分数据
    """
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f)
        if ratio is not None:
            start_idx = int(len(raw_data) * ratio)
            raw_data = raw_data[start_idx:]
            print(f"加载数据: 总量 {len(raw_data)}, 起始点数据比例 { ratio:.1%}")
            return raw_data
        # if data_range is not None:
        #     raw_data = raw_data[:data_range]
        #     print(f"加载数据: 数据总量 {len(raw_data)}")
        #     return raw_data


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
    train_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label
    )

    # 使用训练集的标准化器创建验证集和测试集
    val_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label
    )
    
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, args):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # 添加这个参数
        prefetch_factor=2  # 添加这个参数
    )
    
    # 验证和测试加载器类似修改
    val_sampler = BalancedBatchSampler(val_dataset, args.batch_size)
    val_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
  #  test_sampler = BalancedBatchSampler(test_dataset, args.batch_size)
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                      default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_922.pickle',
                      help='训练数据路径')
    parser.add_argument('--model_save_path', type=str,
                      default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\Binary_ebm_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_57_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl0.pth',
                      help='模型保存路径')
    parser.add_argument('--grasp_data_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_57.pickle',
                        help='抓取数据路径')
    
    # 训练相关参数
    parser.add_argument('--random_seed', type=int, default=22, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=4, help='工作线程数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--train_split', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--data_ratio', type=float, default=0.9, help='数据比例')
    parser.add_argument('--data_range', type=int, default=None, help='数据范围')
    
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=21, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度')
    parser.add_argument('--hidden_dims', type=list, default=[512, 512, 512], help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='隐藏层数量')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout率')
    
    # 训练控制参数
    parser.add_argument('--early_stop_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--scheduler_patience', type=int, default=30, help='学习率调度器耐心值')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率调度器衰减因子')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率')

    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='shared_grasp_experiments_paper', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, 
                     default='BC_shared',
                      help='wandb运行名称')
    parser.add_argument('--use_quaternion', type=int, default=1,
                       help='使用四元数表示 (1) 或简化表示 (0)')
    parser.add_argument('--use_stable_label', type=int, default=0,
                       help='使用稳定性标签 (1) 或不使用 (0)')
    
    parser.add_argument('--train', type=bool, default=False,
                       help='是否训练')
    
    args = parser.parse_args()
    # 将整数转换为布尔值
    args.use_quaternion = bool(args.use_quaternion)
    args.use_stable_label = bool(args.use_stable_label)
    
    return args

if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    set_seed(args.random_seed)

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
    
    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args)
    
    # normal BCE loss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True,
        min_lr=args.min_lr
    )

    # 初始化wandb
    # try:
    #     wandb.init(
    #         project=args.wandb_project,
    #         name=args.wandb_name,
    #         config=vars(args)  # 将所有参数记录到wandb
    #     )
    # except Exception as e:
    #     print(f"Failed to initialize Weights and Biases: {e}")

    # 训练模型
    best_threshold = 0.5  # 默认阈值
    if args.train:
        model, best_threshold = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, num_epochs=args.num_epochs,  save_path=args.model_save_path, early_stop_patience=args.early_stop_patience
        )
       # 加载已训练的模型和最佳阈值
        print("\n在测试集上评估模型:")
        test_results = evaluate_model(model, test_loader, device, args)
    else:
        # 加载已训练的模型和最佳阈值
        print("\n在测试集上评估模型:")

        # 评估最佳阈值
        best_threshold = optimal_f1_threshold(model, val_loader, device)
        test_results = evaluate_model(model, test_loader, device, args, best_threshold)

    




