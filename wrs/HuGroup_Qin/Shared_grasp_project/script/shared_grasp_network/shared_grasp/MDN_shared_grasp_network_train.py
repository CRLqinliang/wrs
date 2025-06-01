import sys
sys.path.append("E:/Qin/wrs")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import wandb
from scipy.spatial.transform import Rotation as R
import argparse
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import gc


# 保持与EBM版本相同的数据集类
class SharedGraspEnergyDataset(Dataset):
    def __init__(self, data, grasp_pickle_file, use_quaternion=False, use_stable_label=True):
        """
        Args:
            data: 数据列表，每个item包含 [[init_pos, init_rotmat], [goal_pos, goal_rotmat], init_stable_id, goal_stable_id, common_id]
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
            
            # 添加初始位姿
            all_features[current_idx:end_idx, feature_start:feature_start+len(init_pose)] = init_pose
            feature_start += len(init_pose)
            
            # 添加目标位姿
            all_features[current_idx:end_idx, feature_start:feature_start+len(goal_pose)] = goal_pose
            feature_start += len(goal_pose)
            
            # 如果使用stable_label，添加One-Hot编码
            if self.use_stable_label:
                init_type_onehot = self.obj_encoder.transform([[item[4]]]).copy()
                goal_type_onehot = self.obj_encoder.transform([[item[9]]]).copy()
                
                all_features[current_idx:end_idx, feature_start:feature_start+len(init_type_onehot[0])] = init_type_onehot
                feature_start += len(init_type_onehot[0])
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


# 混合密度网络 - 替代EBM网络
class GraspMDNetwork(nn.Module):
    def __init__(self, input_dim, n_mixtures=5, hidden_dims=None, num_layers=3, dropout_rate=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        
        # 确保hidden_dims是列表且长度等于num_layers
        if len(hidden_dims) != num_layers:
            hidden_dims = [hidden_dims[0]] * num_layers
        
        # 构建特征提取层
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
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # MDN输出层 - 对于二分类问题，我们使用1维作为输出目标
        # 每个混合分量需要：pi(权重)、mu(均值)、sigma(标准差)
        self.pi_layer = nn.Linear(hidden_dims[-1], n_mixtures)
        self.mu_layer = nn.Linear(hidden_dims[-1], n_mixtures)
        self.sigma_layer = nn.Linear(hidden_dims[-1], n_mixtures)
        
        self.n_mixtures = n_mixtures
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更稳定的初始化方法
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 特别初始化sigma层，确保初始值合理
        nn.init.normal_(self.sigma_layer.weight, mean=0, std=0.01)
        nn.init.constant_(self.sigma_layer.bias, 0.1)  # 改为更小的初始值

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 计算混合权重 (softmax确保和为1)
        pi = F.softmax(self.pi_layer(features), dim=1)
        
        # 计算均值
        mu = self.mu_layer(features)
        
        # 计算标准差 (需要确保为正)
        sigma = torch.exp(self.sigma_layer(features))
        
        return pi, mu, sigma


# MDN损失函数
class MDNLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pi, mu, sigma, target):
        # 添加数值检查
        assert not torch.isnan(mu).any(), "mu contains NaN values"
        assert not torch.isnan(sigma).any(), "sigma contains NaN values"
        assert not torch.isnan(pi).any(), "pi contains NaN values"
        
        # 确保sigma为正且不太小
        sigma = torch.clamp(sigma, min=1e-4)
        
        # 其余代码保持不变
        target = target.float().view(-1, 1)
        normal_dist = torch.distributions.Normal(mu, sigma)
        prob_density = normal_dist.log_prob(target)
        weighted_prob = prob_density + torch.log(pi + self.epsilon)
        max_weighted_prob = torch.max(weighted_prob, dim=1, keepdim=True)[0]
        log_prob = max_weighted_prob + torch.log(
            torch.sum(torch.exp(weighted_prob - max_weighted_prob), dim=1, keepdim=True) + self.epsilon
        )
        nll = -log_prob.mean()
        return nll


def calculate_metrics(pi, mu, sigma, labels, threshold=None):
    """计算评估指标并找到最佳阈值"""
    device = pi.device
    
    # 确保输入是连续的张量
    pi = pi.detach()
    mu = mu.detach()
    sigma = sigma.detach()
    labels = labels.detach().float().view(-1)
    
    # 计算目标为1的概率
    target = torch.ones((pi.shape[0], 1), device=device)
    normal_dist = torch.distributions.Normal(mu, sigma)
    prob_density = torch.exp(normal_dist.log_prob(target))
    weighted_prob = pi * prob_density
    probs = torch.sum(weighted_prob, dim=1).cpu().numpy()
    
    # 将标签转换为NumPy数组
    labels_np = labels.cpu().numpy()
    
    # 分离正负样本
    pos_mask = (labels_np == 1)
    neg_mask = (labels_np == 0)
    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]
    
    # 计算概率统计量
    pos_prob_mean = np.mean(pos_probs) if len(pos_probs) > 0 else 0
    pos_prob_std = np.std(pos_probs) if len(pos_probs) > 0 else 0
    neg_prob_mean = np.mean(neg_probs) if len(neg_probs) > 0 else 0
    neg_prob_std = np.std(neg_probs) if len(neg_probs) > 0 else 0
    prob_gap = pos_prob_mean - neg_prob_mean
    
    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    roc_auc = auc(fpr, tpr)
    
    # 如果没有提供阈值，找到最佳阈值（使F1分数最大化）
    if threshold is None:
        precisions, recalls, thresholds_pr = precision_recall_curve(labels_np, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5
    else:
        best_threshold = threshold
    
    # 使用阈值计算预测结果
    predictions = (probs >= best_threshold).astype(np.int32)
    
    # 计算混淆矩阵
    tp = np.sum((predictions == 1) & (labels_np == 1))
    fp = np.sum((predictions == 1) & (labels_np == 0))
    fn = np.sum((predictions == 0) & (labels_np == 1))
    tn = np.sum((predictions == 0) & (labels_np == 0))
    
    # 计算各种指标
    accuracy = (tp + tn) / len(labels_np)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'pos_prob_mean': pos_prob_mean,
        'pos_prob_std': pos_prob_std,
        'neg_prob_mean': neg_prob_mean,
        'neg_prob_std': neg_prob_std,
        'prob_gap': prob_gap,
        'optimal_threshold': best_threshold
    }


def calculate_metrics_with_fixed_threshold(pi, mu, sigma, labels, threshold):
    """使用固定阈值计算评估指标"""
    device = pi.device
    
    # 确保输入是连续的张量
    pi = pi.detach()
    mu = mu.detach()
    sigma = sigma.detach()
    labels = labels.detach().float().view(-1)
    
    # 计算目标为1的概率
    target = torch.ones((pi.shape[0], 1), device=device)
    normal_dist = torch.distributions.Normal(mu, sigma)
    prob_density = torch.exp(normal_dist.log_prob(target))
    weighted_prob = pi * prob_density
    probs = torch.sum(weighted_prob, dim=1).cpu().numpy()
    
    # 将标签转换为NumPy数组
    labels_np = labels.cpu().numpy()
    
    # 使用固定阈值进行预测
    predictions = (probs >= threshold)
    
    # 计算指标
    tp = np.sum((predictions == 1) & (labels_np == 1))
    fp = np.sum((predictions == 1) & (labels_np == 0))
    fn = np.sum((predictions == 0) & (labels_np == 1))
    tn = np.sum((predictions == 0) & (labels_np == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels_np)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


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
        train_pi_list = []
        train_mu_list = []
        train_sigma_list = []
        train_labels_list = []
        batch_count = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            pi, mu, sigma = model(inputs)
            loss = criterion(pi, mu, sigma, labels)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()  # 保存当前loss值
            epoch_loss += current_loss
            batch_count += 1
            
            # 收集训练数据时立即转为CPU张量并释放GPU内存
            train_pi_list.append(pi.detach().cpu())
            train_mu_list.append(mu.detach().cpu())
            train_sigma_list.append(sigma.detach().cpu())
            train_labels_list.append(labels.cpu())
            
            # 及时清理GPU内存
            del pi, mu, sigma, loss, inputs, labels
            
            train_loop.set_postfix({'loss': current_loss})
        
        # 计算训练指标
        train_pi = torch.cat(train_pi_list).to(device)
        train_mu = torch.cat(train_mu_list).to(device)
        train_sigma = torch.cat(train_sigma_list).to(device)
        train_labels = torch.cat(train_labels_list).to(device)
        train_metrics = calculate_metrics(train_pi, train_mu, train_sigma, train_labels)
        avg_train_loss = epoch_loss / batch_count
        
        # 清理训练指标计算后的内存
        del train_pi_list, train_mu_list, train_sigma_list, train_labels_list
        del train_pi, train_mu, train_sigma, train_labels

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_pi_list = []
        val_mu_list = []
        val_sigma_list = []
        val_labels_list = []
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for val_inputs, val_labels in val_loop:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_pi, val_mu, val_sigma = model(val_inputs)
                current_val_loss = criterion(val_pi, val_mu, val_sigma, val_labels).item()
                val_loss += current_val_loss
                
                # 收集验证数据
                val_pi_list.append(val_pi.cpu())
                val_mu_list.append(val_mu.cpu())
                val_sigma_list.append(val_sigma.cpu())
                val_labels_list.append(val_labels.cpu())
                
                # 及时清理GPU内存
                del val_pi, val_mu, val_sigma, val_inputs, val_labels
                
                val_loop.set_postfix({'loss': current_val_loss})
        
        # 计算验证指标
        val_pi = torch.cat(val_pi_list).to(device)
        val_mu = torch.cat(val_mu_list).to(device)
        val_sigma = torch.cat(val_sigma_list).to(device)
        val_labels = torch.cat(val_labels_list).to(device)
        val_metrics = calculate_metrics(val_pi, val_mu, val_sigma, val_labels)
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
            "train/pos_prob_mean": train_metrics['pos_prob_mean'],
            "train/pos_prob_std": train_metrics['pos_prob_std'],
            "train/neg_prob_mean": train_metrics['neg_prob_mean'],
            "train/neg_prob_std": train_metrics['neg_prob_std'],
            "train/prob_gap": train_metrics['prob_gap'],
            "train/optimal_threshold": train_metrics['optimal_threshold'],
            
            # 验证集详细指标
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "val/f1": val_metrics['f1'],
            "val/auc": val_metrics['auc'],
            "val/pos_prob_mean": val_metrics['pos_prob_mean'],
            "val/pos_prob_std": val_metrics['pos_prob_std'],
            "val/neg_prob_mean": val_metrics['neg_prob_mean'],
            "val/neg_prob_std": val_metrics['neg_prob_std'],
            "val/prob_gap": val_metrics['prob_gap'],
            "val/optimal_threshold": val_metrics['optimal_threshold']
        })

        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Recall:{train_metrics['recall']:.4f}, Precision:{train_metrics['precision']:.4f},"
              f" F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Recall:{val_metrics['recall']:.4f}, Precision:{val_metrics['precision']:.4f},"
              f" F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        print(f"Prob Gap - Train: {train_metrics['prob_gap']:.4f}, Val: {val_metrics['prob_gap']:.4f}")
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
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n早停: 验证集F1分数 {early_stop_patience} 个epoch没有改善")
            break

        # 清理内存
        del val_pi_list, val_mu_list, val_sigma_list, val_labels_list
        del val_pi, val_mu, val_sigma, val_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    # 训练结束后，加载最佳模型并返回
    best_model_info = torch.load(save_path)
    model.load_state_dict(best_model_info['model_state_dict'])
    print(f"\n训练完成 - 加载最佳模型(Epoch {best_model_info['epoch']})")
    print(f"最佳验证集F1: {best_model_info['best_val_metric']:.4f}")
    print(f"最佳阈值: {best_model_info['optimal_threshold']:.4f}")
    
    return model, best_model_info['optimal_threshold']


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
    test_pi_list = []
    test_mu_list = []
    test_sigma_list = []
    test_labels_list = []
    
    # 收集所有预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs = inputs.to(device)
            pi, mu, sigma = model(inputs)
            
            test_pi_list.append(pi.cpu())
            test_mu_list.append(mu.cpu())
            test_sigma_list.append(sigma.cpu())
            test_labels_list.append(labels)
            
            # 清理GPU内存
            del pi, mu, sigma, inputs
    
    # 将所有数据移到GPU上进行一次性评估
    test_pi = torch.cat(test_pi_list).to(device)
    test_mu = torch.cat(test_mu_list).to(device)
    test_sigma = torch.cat(test_sigma_list).to(device)
    test_labels = torch.cat(test_labels_list).to(device)
    
    # 使用验证集阈值评估测试集
    test_metrics_with_val_threshold = calculate_metrics_with_fixed_threshold(
        test_pi, test_mu, test_sigma, test_labels, val_threshold)
    
    print("\n=== 使用验证集阈值评估测试集 ===")
    print(f"阈值: {val_threshold:.4f}")
    print(f"Accuracy: {test_metrics_with_val_threshold['accuracy']:.4f}")
    print(f"Precision: {test_metrics_with_val_threshold['precision']:.4f}")
    print(f"Recall: {test_metrics_with_val_threshold['recall']:.4f}")
    print(f"F1 Score: {test_metrics_with_val_threshold['f1']:.4f}")
    
    # 仅供参考：计算测试集自身的最佳阈值和性能
    test_metrics = calculate_metrics(test_pi, test_mu, test_sigma, test_labels)
    
    print("\n=== 测试集自身的最佳阈值(仅供参考) ===")
    print(f"最佳阈值: {test_metrics['optimal_threshold']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # 分析概率分布
    analyze_probability_distributions(test_pi, test_mu, test_sigma, test_labels, val_threshold)
    
    # 清理内存
    del test_pi_list, test_mu_list, test_sigma_list, test_labels_list
    del test_pi, test_mu, test_sigma, test_labels
    
    return test_metrics_with_val_threshold


def analyze_probability_distributions(pi, mu, sigma, labels, threshold):
    """分析概率分布"""
    device = pi.device
    
    # 计算目标为1的概率
    target = torch.ones((pi.shape[0], 1), device=device)
    normal_dist = torch.distributions.Normal(mu, sigma)
    prob_density = torch.exp(normal_dist.log_prob(target))
    weighted_prob = pi * prob_density
    probs = torch.sum(weighted_prob, dim=1).cpu().numpy()
    
    # 将标签转换为NumPy数组
    labels_np = labels.cpu().numpy().flatten()
    
    # 分离正负样本概率
    pos_mask = (labels_np == 1)
    neg_mask = (labels_np == 0)
    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]
    
    print("\n=== 概率分布统计 ===")
    print(f"正样本数量: {len(pos_probs)}")
    print(f"负样本数量: {len(neg_probs)}")
    print(f"正样本概率 - 均值: {np.mean(pos_probs):.4f}, 标准差: {np.std(pos_probs):.4f}")
    print(f"负样本概率 - 均值: {np.mean(neg_probs):.4f}, 标准差: {np.std(neg_probs):.4f}")
    print(f"概率差距: {np.mean(pos_probs) - np.mean(neg_probs):.4f}")
    
    # 分析阈值效果
    above_threshold_pos = np.sum(pos_probs >= threshold)
    above_threshold_neg = np.sum(neg_probs >= threshold)
    
    print(f"\n=== 阈值 {threshold:.4f} 的效果 ===")
    print(f"正样本高于阈值比例: {above_threshold_pos/len(pos_probs):.4f} ({above_threshold_pos}/{len(pos_probs)})")
    print(f"负样本高于阈值比例: {above_threshold_neg/len(neg_probs):.4f} ({above_threshold_neg}/{len(neg_probs)})")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

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
        print(f"加载数据: 总量 {len(raw_data)}, 起始比例 {1 - ratio:.1%}")
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
    # 创建训练集
    train_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )

    # 创建验证集和测试集
    val_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )
    
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_data_path,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label,
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, args):
    """创建数据加载器"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 验证和测试加载器
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
        pin_memory=args.pin_memory
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
    
    # 创建MDN模型
    model = GraspMDNetwork(
        input_dim=input_dim,
        n_mixtures=args.n_mixtures,
        hidden_dims=args.hidden_dims,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate
    )
    
    # 初始化损失函数
    criterion = MDNLoss()
    
    # 使用更小的初始学习率
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 添加梯度裁剪
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    
    # 初始化学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr
    )
    
    return model, criterion, optimizer, scheduler, device


def parse_args():
    parser = argparse.ArgumentParser(description='Mixture Density Network for Grasp Model')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                       default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\grasp_random_position_bottle_robot_57.pickle')
    parser.add_argument('--grasp_data_path', type=str,
                       default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_57.pickle')
    parser.add_argument('--model_save_path', type=str,
                       default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\Binary_mdn_model\best_model_grasp_robot_57_mdn_selu.pth')
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--data_ratio', type=float, default=0.5)
    
    # 模型结构参数
    parser.add_argument('--input_dim', type=int, default=12)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--n_mixtures', type=int, default=5,
                       help='混合密度网络的高斯混合组件数量')
    parser.add_argument('--use_quaternion', type=int, default=1,
                       help='使用四元数表示 (1) 或简化表示 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1,
                       help='使用稳定性标签 (1) 或不使用 (0)')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop_patience', type=int, default=20)

    # 学习率调度器参数
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_model', type=bool, default=True)
    parser.add_argument('--wandb_project', type=str, default='grasp_mdn')
    parser.add_argument('--wandb_name', type=str, default='grasp_random_position_bottle_robot_57_mdn_selu')
    
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
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # 加载原始数据
    raw_data = load_raw_data(args.data_path, args.data_ratio)
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
        evaluate_model(model, val_loader, device, args)
    else:
        # 评估模型
        print("开始评估模型...")
        evaluate_model(model, val_loader, device, args)


if __name__ == '__main__':
    main()