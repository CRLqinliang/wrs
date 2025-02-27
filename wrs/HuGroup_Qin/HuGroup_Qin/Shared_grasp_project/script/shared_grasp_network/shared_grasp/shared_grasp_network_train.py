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
import random
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

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


class GraspingDataset(Dataset):
    def __init__(self, output_dim, pickle_file, data_ratio=0.5, stable_label=True):
        self.output_dim = output_dim
        self.stable_label = stable_label

        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                # 根据data_ratio选择数据
                start_idx = int(len(data) * (data_ratio))
                data = data[start_idx:]
                print(f"加载数据: 总量 {len(data)}, 使用比例 {1 - data_ratio:.1%}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading file {pickle_file}: {e}")

        # 提取初始位姿和目标位姿
        init_poses = []
        goal_poses = []
        for item in data:
            init_pos = np.array(item[0][0]).flatten()
            init_rot = self._rotmat_to_quaternion(item[0][1])
            init_poses.append(np.concatenate([init_pos, init_rot]))
            goal_pos = np.array(item[5][0]).flatten()
            goal_rot = self._rotmat_to_quaternion(item[5][1])
            goal_poses.append(np.concatenate([goal_pos, goal_rot]))

        init_poses = np.array(init_poses, dtype=np.float32)
        goal_poses = np.array(goal_poses, dtype=np.float32)
        
        # 分离位置和角度数据
        init_positions = init_poses[:, :3]
        init_angles = init_poses[:, 3:]
        goal_positions = goal_poses[:, :3]
        goal_angles = goal_poses[:, 3:]
        

        # 不进行标准化，直接使用原始数据
        init_positions_scaled = init_positions
        goal_positions_scaled = goal_positions
        self.position_scaler = None

        # 创建物体stable_placement_id类型的One-Hot编码器
        obj_types = [item[4] for item in data]  # stable_placement_id
        self.obj_encoder = OneHotEncoder(sparse_output=False)
        self.obj_encoder.fit(np.array(obj_types).reshape(-1, 1))


        # 是否增加稳定标签到输入里面
        if self.stable_label:
            # 获取原始标签并重塑为二维数组
            init_raw_labels = np.array([item[4] for item in data], dtype=int).reshape(-1, 1)
            goal_raw_labels = np.array([item[9] for item in data], dtype=int).reshape(-1, 1)
       
            # 转换标签为one-hot编码
            init_stable_label_one_hot = self.obj_encoder.fit_transform(init_raw_labels)
            goal_stable_label_one_hot = self.obj_encoder.fit_transform(goal_raw_labels)

            # 连接所有输入特征
            self.inputs = np.concatenate([
                init_positions_scaled, 
                init_angles, 
                init_stable_label_one_hot,  # 现在会是一个5维的one-hot向量
                goal_positions_scaled, 
                goal_angles,
                goal_stable_label_one_hot   # 现在会是一个5维的one-hot向量
            ], axis=1)
        else:
            # 组合所有特征
            self.inputs = np.concatenate([
                init_positions_scaled,
                init_angles,
                goal_positions_scaled,
                goal_angles
            ], axis=1).astype(np.float32)
            
        # 处理标签
        self.labels = np.array([self._create_target_vector(item[-1]) for item in data], dtype=np.float32)

    def _rotmat_to_quaternion(self, rotmat):
        """将旋转矩阵转换为四元数"""
        r = R.from_matrix(rotmat)
        return r.as_quat()

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


class BalancedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.5, neg_weight=1.0, zero_label_weight=1.05):
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.zero_label_weight = zero_label_weight

    def forward(self, inputs, targets):
        # 计算样本权重
        positive_samples = targets.sum(dim=1)
        is_zero_label = (positive_samples == 0)
        
        # 对全0样本使用轻微的权重提升
        sample_weights = torch.ones(targets.shape[0], device=targets.device)
        sample_weights[is_zero_label] = self.zero_label_weight
        
        # 对正负样本分别计算权重
        weights = torch.where(
            targets == 1,
            torch.ones_like(targets) * self.pos_weight,  # 正样本权重
            torch.ones_like(targets) * self.neg_weight   # 负样本权重
        )
        
        # 应用样本权重
        weights = weights * sample_weights.view(-1, 1)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            reduction='none'
        )
        
        return (weights * bce_loss).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.7): 
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class WeightedFocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=4.0, neg_weight=1.0):
        super(WeightedFocalBCELoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def forward(self, inputs, targets):
        # 计算基础BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # 计算预测概率
        pt = torch.exp(-bce_loss)
        
        # 分配样本权重
        weights = torch.where(
            targets == 1,
            torch.ones_like(targets) * self.pos_weight,
            torch.ones_like(targets) * self.neg_weight
        )
        
        # 计算focal loss
        focal_loss = weights * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()


# Multi-label classification network
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
                # SELU激活函数推荐的初始化方法
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, best_model_path, num_epochs=50, early_stop_patience=100
):
    model.to(device)
    best_val_f1 = 0.0
    best_val_metrics = None
    best_threshold = 0.5
    epochs_no_improve = 0
    early_stop = False
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_metrics': [],
        'thresholds': []
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break

        # 训练阶段
        model.train()
        running_loss = 0.0
        num_batches = 0

        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            train_pbar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / num_batches
        history['train_loss'].append(avg_loss)

        # 评估阶段
        with torch.no_grad():
            model.eval()
            
            # 寻找最佳阈值
            threshold = find_optimal_threshold(model, val_loader, device)
            history['thresholds'].append(threshold)
            
            # 使用最佳阈值评估模型
            train_results = evaluate_model(model, train_loader, device, threshold)
            val_results = evaluate_model(model, val_loader, device, threshold)
            history['val_metrics'].append(val_results)

        # 打印当前epoch的训练结果
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        print(f'Loss: {avg_loss:.4f}, Threshold: {threshold:.4f}')
        print(f'Train - P: {train_results["precision_macro"]:.4f}, R: {train_results["recall_macro"]:.4f}, F1: {train_results["f1_macro"]:.4f}')
        print(f'Val   - P: {val_results["precision_macro"]:.4f}, R: {val_results["recall_macro"]:.4f}, F1: {val_results["f1_macro"]:.4f}')

        # 记录到wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_loss,
            "Threshold": threshold,
            "Train Precision": train_results["precision_macro"],
            "Train Recall": train_results["recall_macro"],
            "Train F1": train_results["f1_macro"],
            "Train PR-AUC": train_results["pr_auc_macro"],
            "Val Precision": val_results["precision_macro"],
            "Val Recall": val_results["recall_macro"], 
            "Val F1": val_results["f1_macro"],
            "Val PR-AUC": val_results["pr_auc_macro"],
            "Validation Accuracy": val_results["accuracy"],
            "Validation Hamming Loss": val_results["hamming_loss"],
            "Learning Rate": optimizer.param_groups[0]['lr']
        })

        # 模型保存和早停
        scheduler.step(val_results["f1_macro"])

        if val_results["f1_macro"] > best_val_f1:
            best_val_f1 = val_results["f1_macro"]
            best_val_metrics = val_results
            best_threshold = threshold
            
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_f1,
                'optimal_threshold': best_threshold,
                'val_metrics': best_val_metrics
            }
            torch.save(checkpoint, best_model_path)
            
            print(f"最佳模型已保存，F1: {best_val_f1:.4f}, 阈值: {best_threshold:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                early_stop = True
                print(f"Early stopping triggered at epoch {epoch + 1}")

        # 每个epoch结束后清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练结束，加载最佳模型
    print(f"\n训练完成! 最佳验证F1: {best_val_f1:.4f}, 最佳阈值: {best_threshold:.4f}")
    
    # 返回训练历史和最佳阈值
    return history, best_threshold


def evaluate_model(model, data_loader, device, threshold=0.7):
    model.eval()
    all_labels = []
    all_predictions = []
    all_logits = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            predicted = (outputs > threshold).float()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_logits = np.vstack(all_logits)

    # 统计每个类别的正样本数量
    num_positives_per_class = np.sum(all_labels, axis=0) > 0
    valid_classes = num_positives_per_class > 0  # 有正样本的类别索引

    # 忽略无正样本类别
    if not np.any(valid_classes):
        raise ValueError("No valid classes with positive samples.")

    filtered_labels = all_labels[:, valid_classes]
    filtered_predictions = all_predictions[:, valid_classes]
    filtered_logits = all_logits[:, valid_classes]

    # 计算各类指标（仅针对有正样本的类别）
    results = {
        "accuracy": accuracy_score(all_labels, all_predictions) * 100,
        "hamming_loss": hamming_loss(all_labels, all_predictions),
        "precision_macro": precision_score(filtered_labels, filtered_predictions, average='macro', zero_division=0),
        "recall_macro": recall_score(filtered_labels, filtered_predictions, average='macro', zero_division=0),
        "f1_macro": f1_score(filtered_labels, filtered_predictions, average='macro', zero_division=0),
        "precision_micro": precision_score(filtered_labels, filtered_predictions, average='micro', zero_division=0),
        "recall_micro": recall_score(filtered_labels, filtered_predictions, average='micro', zero_division=0),
        "f1_micro": f1_score(filtered_labels, filtered_predictions, average='micro', zero_division=0),
        "pr_auc_macro": average_precision_score(filtered_labels, filtered_logits, average='macro'),
        "pr_auc_micro": average_precision_score(filtered_labels, filtered_logits, average='micro')
    }

    # 打印分类报告（仅针对有正样本的类别）
    class_report = classification_report(filtered_labels, filtered_predictions, zero_division=0)
    print(f"Validation Accuracy: {results['accuracy']:.2f}%")
    print(f"Hamming Loss: {results['hamming_loss']:.4f}")
    print(f"Precision (macro): {results['precision_macro']:.2f}, Recall (macro): {results['recall_macro']:.2f}, F1 Score (macro): {results['f1_macro']:.2f}")
    print(f"Precision (micro): {results['precision_micro']:.2f}, Recall (micro): {results['recall_micro']:.2f}, F1 Score (micro): {results['f1_micro']:.2f}")
    print(f"PR-AUC (macro): {results['pr_auc_macro']:.4f}, PR-AUC (micro): {results['pr_auc_micro']:.4f}")
    # print("\nClassification Report:\n", class_report)

    return results


def find_optimal_threshold(model, val_loader, device):
        """动态阈值选择策略，直接寻找最佳F1分数的阈值"""
        # 收集预测结果
        model.eval()
        with torch.no_grad():
            all_labels, all_outputs = [], []
            for inputs, labels in val_loader:
                outputs = torch.sigmoid(model(inputs.to(device)))
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        # 处理数据
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        valid_classes = np.sum(all_labels, axis=0) > 0
        filtered_labels = all_labels[:, valid_classes]
        filtered_outputs = all_outputs[:, valid_classes]

        # 直接寻找最佳F1分数的阈值
        best_f1 = -float('inf')
        best_threshold = 0.5
        
        # 遍历可能的阈值
        for th in np.arange(0.01, 0.99, 0.01):
            predictions = (filtered_outputs > th).astype(float)
            f1 = f1_score(filtered_labels, predictions, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        
        return best_threshold


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                      default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\argu_xyt_stable_label_common_grasp_random_position_bottle_109.pickle',
                      help='训练数据路径')
    parser.add_argument('--model_save_path', type=str,
                      default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\shared_best_model\shared_grasp_argu_xyt_stable_label_common_grasp_random_position_bottle_109.pth',
                      help='模型保存路径')
    parser.add_argument('--grasp_data_path', type=str, default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle',
                        help='抓取数据路径')
    
    # 训练相关参数
    parser.add_argument('--random_seed', type=int, default=22, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--train_split', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='数据比例')
    
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=12, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=57, help='输出维度')
    parser.add_argument('--hidden_dims', type=list, default=[512, 512, 512], help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='隐藏层数量')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout率')
    
    # 训练控制参数
    parser.add_argument('--early_stop_patience', type=int, default=300, help='早停耐心值')
    parser.add_argument('--scheduler_patience', type=int, default=30, help='学习率调度器耐心值')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率调度器衰减因子')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率')

    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='regrasp', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, 
                     default='shared_grasp_mlp_bottle_+xy+xyt+same_pos_1.0_57',
                      help='wandb运行名称')
    parser.add_argument('--stable_label', type=bool, default=True,
                       help='是否增加稳定标签到输入里面')
    
    return parser.parse_args()


def get_model(args):
    """根据参数选择网络架构"""
    return GraspNetwork(input_dim=args.input_dim, output_dim=args.output_dim, 
                        hidden_dims=args.hidden_dims, num_layers=args.num_layers, dropout_rate=args.dropout_rate)



if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    set_seed(args.random_seed)
    
    # 加载数据集
    full_dataset = GraspingDataset(
        args.output_dim, 
        args.data_path,
        data_ratio=args.data_ratio,
        stable_label=args.stable_label
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
    
    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args)
    
    # normal BCE loss
    criterion = nn.BCEWithLogitsLoss()
    # criterion = WeightedFocalBCELoss()
    # criterion = FocalBCELoss(gamma=args.gamma, alpha=args.alpha)
    
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
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)  # 将所有参数记录到wandb
        )
    except Exception as e:
        print(f"Failed to initialize Weights and Biases: {e}")

    # 训练模型
    history, best_threshold = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.model_save_path, args.num_epochs, args.early_stop_patience
    )

    # # 加载最佳模型并评估
    # from wrs.HuGroup_Qin.Shared_grasp_project.network.MlpBlock import GraspingNetwork

    # # # 新数据测试
    # # model = GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)
    # # model.load_state_dict(torch.load(args.model_save_path))
    # # results, best_threshold, best_f1 = evaluate_model_performance(args, model)
    # # print(results)

    # # 已有数据集测试
    # model = GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)
    # model.load_state_dict(torch.load(args.model_save_path))

    # # 加载测试数据集
    # test_dataset = GraspingDataset(args.output_dim, args.data_path)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # # 使用数据集测试模型
    # print("\n使用数据集进行测试...")
    # dataset_results = test_model_with_dataset(model, test_loader, device='cpu')




