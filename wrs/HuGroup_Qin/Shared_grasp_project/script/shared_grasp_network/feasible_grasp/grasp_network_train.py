import os
import sys
sys.path.append("H:/Qin/wrs")
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
from tqdm.auto import tqdm  # 改用这种方式导入tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GraspingDataset(Dataset):
    def __init__(self, output_dim, use_stable_label, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading file {pickle_file}: {e}")

        self.output_dim = output_dim
        self.use_stable_label = use_stable_label

        # 提取初始位姿和目标位姿
        init_poses = []
        for item in data:
            # 确保position是一维数组
            init_pos = np.array(item[0][0]).flatten()
            init_rot = self._rotmat_to_euler(item[0][1])
            init_poses.append(np.concatenate([init_pos, init_rot]))

        init_poses = np.array(init_poses, dtype=np.float32)

        # 分离位置和角度数据
        init_positions = init_poses[:, :3]  # x,y,z
        init_angles = init_poses[:, 3:]     # rx,ry,rz

        # 组合所有特征
        self.inputs = np.concatenate([
            init_positions,
            init_angles,
        ], axis=1)

        # 处理标签
        if self.use_stable_label:
            stable_labels = np.array([item[1] for item in data], dtype=int).reshape(-1, 1)
            encoder = OneHotEncoder(sparse=False)
            stable_labels_one_hot = encoder.fit_transform(stable_labels)
            self.inputs = np.concatenate([
                init_positions,
                init_angles,
                stable_labels_one_hot
            ], axis=1)
        # 处理标签
        self.labels = np.array([self._create_target_vector(item[-1]) for item in data], dtype=np.float32)

    def _rotmat_to_euler(self, rotmat):
        """将旋转矩阵转换为欧拉角(rx, ry, rz)"""
        r = R.from_matrix(rotmat)
        return r.as_euler('xyz', degrees=False)

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
    def __init__(self, gamma=1.5, alpha=0.6):  # 降低gamma，调整alpha
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, best_model_path, eval_th=0.7, num_epochs=50, early_stop_patience=100,
    min_precision=0.75, min_recall=0.4
):
    model.to(device)
    train_losses = []
    val_metrics = []  
    best_val_metric = 0.0  
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break

        # 训练阶段
        model.train()
        running_loss = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches

        # 评估阶段
        with torch.no_grad():
            model.eval()
            train_results = evaluate_model(model, train_loader, device, eval_th)
            val_results = evaluate_model(model, val_loader, device, eval_th)

        # 打印当前epoch的训练结果
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        print(f'Loss: {avg_loss:.4f}')
        print(f'Train - P: {train_results["precision_macro"]:.4f}, R: {train_results["recall_macro"]:.4f}, F1: {train_results["f1_macro"]:.4f}')
        print(f'Val   - P: {val_results["precision_macro"]:.4f}, R: {val_results["recall_macro"]:.4f}, F1: {val_results["f1_macro"]:.4f}')

        # 记录训练指标
        train_losses.append(avg_loss)
        val_metrics.append(val_results)

        # 记录到wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_loss,
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

        # 动态调整阈值
        if (epoch + 1) % 5 == 0:
            current_precision = val_results["precision_macro"]
            precision_target = min(0.8, current_precision + 0.05)
            
            eval_th = find_optimal_threshold(
                model, val_loader, device, 
                min_precision=min_precision,
                min_recall=min_recall,
                precision_target=precision_target
            )
            print(f"Epoch {epoch + 1}: 调整阈值为 {eval_th:.3f}")

        # 模型保存和早停
        scheduler.step(val_results["precision_macro"])
        if val_results["precision_macro"] > best_val_metric:
            best_val_metric = val_results["precision_macro"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Precision: {best_val_metric:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                early_stop = True
                print(f"Early stopping triggered at epoch {epoch + 1}")

    return train_losses, val_metrics


def evaluate_model(model, data_loader, device, threshold=0.55):
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
    num_positives_per_class = np.sum(all_labels, axis=0) 
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
    print("\nClassification Report:\n", class_report)

    return results


def find_optimal_threshold(model, val_loader, device, 
                         min_precision=0.75,
                         min_recall=0.4,
                         thresholds=np.arange(0.2, 0.8, 0.01),
                         precision_target=0.8):
    """动态阈值选择策略"""
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
    
    # 搜索最优阈值
    best_metrics = None
    search_range = thresholds
    
    for _ in range(2):  # 最多尝试两轮搜索
        best_score = -float('inf')
        for th in search_range:
            predictions = (filtered_outputs > th).astype(float)
            precision = precision_score(filtered_labels, predictions, average='macro', zero_division=0)
            recall = recall_score(filtered_labels, predictions, average='macro', zero_division=0)
            
            if precision >= min_precision and recall >= min_recall:
                # 使用加权F1分数评估
                precision_weight = 1.5 if precision < precision_target else 1.0
                weighted_f1 = ((1 + precision_weight) * precision * recall) / \
                            (precision_weight * precision + recall + 1e-6)
                
                if weighted_f1 > best_score:
                    best_score = weighted_f1
                    best_metrics = {
                        'threshold': th,
                        'precision': precision,
                        'recall': recall,
                        'f1': weighted_f1
                    }
        
        if best_metrics:
            break
            
        # 如果第一轮没找到合适的阈值，扩大搜索范围
        search_range = np.arange(0.1, 0.95, 0.01)
    
    # 如果仍然没找到合适的阈值，使用backup策略
    if not best_metrics:
        for th in sorted(thresholds, reverse=True):
            predictions = (filtered_outputs > th).astype(float)
            precision = precision_score(filtered_labels, predictions, average='macro', zero_division=0)
            if precision >= min_precision:
                return th
    
    return best_metrics['threshold'] if best_metrics else 0.5


def test_model_with_dataset(model, test_loader, device,
                            thresholds=np.arange(0.3, 0.9, 0.05),
                            verbose=True):
    """
    使用现有数据集测试模型性能，并搜索最佳阈值
    """
    model.eval()

    # 收集所有预测和标签
    all_labels, all_logits = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="收集预测结果"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            all_labels.append(labels.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_logits = np.vstack(all_logits)

    # 过滤有效类别
    valid_classes = np.sum(all_labels, axis=0) > 0
    filtered_labels = all_labels[:, valid_classes]
    filtered_logits = all_logits[:, valid_classes]

    # 测试不同阈值
    threshold_results = {}
    best_f1 = -1
    best_threshold = None
    best_results = None

    # 收集绘图数据
    thresholds_list, precisions_list, recalls_list = [], [], []

    print("\n测试不同阈值的性能:") if verbose else None
    for threshold in tqdm(thresholds, desc="测试阈值"):
        predictions = (all_logits > threshold).astype(float)
        filtered_predictions = (filtered_logits > threshold).astype(float)

        # 计算评估指标
        results = {
            "threshold": threshold,
            "accuracy": accuracy_score(all_labels, predictions) * 100,
            "hamming_loss": hamming_loss(all_labels, predictions),
            "precision_macro": precision_score(filtered_labels, filtered_predictions,
                                               average='macro', zero_division=0),
            "recall_macro": recall_score(filtered_labels, filtered_predictions,
                                         average='macro', zero_division=0),
            "f1_macro": f1_score(filtered_labels, filtered_predictions,
                                 average='macro', zero_division=0),
            "pr_auc_macro": average_precision_score(filtered_labels, filtered_logits,
                                                    average='macro')
        }

        # 收集绘图数据
        thresholds_list.append(threshold)
        precisions_list.append(results["precision_macro"])
        recalls_list.append(results["recall_macro"])

        threshold_results[threshold] = results

        # 更新最佳F1分数
        if results["f1_macro"] > best_f1:
            best_f1 = results["f1_macro"]
            best_threshold = threshold
            best_results = results

        if verbose:
            print(f"\nThreshold = {threshold:.2f}:")
            print(f"Accuracy: {results['accuracy']:.2f}%")
            print(f"Macro - P: {results['precision_macro']:.4f}, "
                  f"R: {results['recall_macro']:.4f}, "
                  f"F1: {results['f1_macro']:.4f}")

    # 记录到wandb
    try:
        wandb.log({
            "Dataset Threshold Evaluation": wandb.Table(
                data=[[f"{th:.2f}"] + [f"{v:.4f}" for k, v in res.items() if k != 'threshold']
                      for th, res in threshold_results.items()],
                columns=["Threshold"] + [k for k in best_results.keys() if k != 'threshold']
            ),
            "Dataset Best Threshold": best_threshold,
            "Dataset Best F1 Score": best_f1,
            "Precision-Recall vs Threshold": wandb.plot.line_series(
                xs=thresholds_list,
                ys=[precisions_list, recalls_list],
                keys=["Macro Precision", "Macro Recall"],
                title="Precision-Recall vs Threshold",
                xname="Threshold"
            )
        })
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {str(e)}")

    return best_results, threshold_results


def analyze_spatial_precision(model, test_loader, device, 
                            threshold=0.55,
                            save_path='spatial_precision_scatter.png'):
    """
    在指定的工作范围内绘制precision散点图, 并显示采样密度
    """
    model.eval()
    
    # 收集数据
    positions = []
    precisions = []
    scaler = test_loader.dataset.dataset.position_scaler

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="收集预测结果"):
            features = features.to(device)
            labels = labels.to(device)
            
            # 获取位置信息 (x,y)
            pos_normalized = features[:, :2].cpu().numpy()
            # 创建完整的特征向量（填充z轴为0）以匹配scaler的维度
            pos_full = np.zeros((pos_normalized.shape[0], 3))
            pos_full[:, :2] = pos_normalized
            # 转换回原始坐标
            pos = scaler.inverse_transform(pos_full)[:, :2]
            
            # 获取模型预测
            outputs = torch.sigmoid(model(features))
            predictions = (outputs > threshold).float()
            
            # 对每个样本计算precision
            for i in range(len(pos)):
                # 获取当前样本的预测和标签
                sample_pred = predictions[i:i+1]
                sample_label = labels[i:i+1]
                
                # 找出有效类别（有正样本的类别）
                valid_classes = (sample_label.sum(dim=0) > 0)
                
                if valid_classes.any():
                    # 过滤有效类别
                    filtered_pred = sample_pred[:, valid_classes]
                    filtered_label = sample_label[:, valid_classes]
                    
                    # 使用sklearn计算macro precision
                    precision = precision_score(
                        filtered_label.cpu().numpy(),
                        filtered_pred.cpu().numpy(),
                        average='macro',
                        zero_division=0
                    )
                    
                    positions.append(pos[i])
                    precisions.append(precision)
    
    positions = np.array(positions)
    precisions = np.array(precisions)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # 1. 散点图：显示precision分布
    scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                         c=precisions, 
                         cmap='YlOrRd',
                         s=50,
                         alpha=0.6,
                         vmin=0, vmax=1)
    
    # ax1.set_xlim(x_range)
    # ax1.set_ylim(y_range)
    ax1.set_xlabel('X Position (m)', fontsize=16)
    ax1.set_ylabel('Y Position (m)', fontsize=16)
    ax1.set_title('Spatial Precision Distribution', fontsize=18, pad=15)
    ax1.tick_params(axis='both', labelsize=18)  # 设置x和y轴刻度标签大小
    ax1.grid(True, linestyle='--', alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1, label='Precision')
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Precision', fontsize=16)
    
    # 2. 密度图：显示采样点密度
    from scipy.stats import gaussian_kde
    xy = np.vstack([positions[:, 0], positions[:, 1]])
    z = gaussian_kde(xy)(xy)
    
    density = ax2.scatter(positions[:, 0], positions[:, 1],
                         c=z,
                         cmap='viridis',
                         s=50,
                         alpha=0.6)
    
    # ax2.set_xlim(x_range)
    # ax2.set_ylim(y_range)
    ax2.set_xlabel('X Position (m)', fontsize=16)
    ax2.set_ylabel('Y Position (m)', fontsize=16)
    ax2.set_title('Sampling Density Distribution', fontsize=18, pad=15)
    ax2.tick_params(axis='both', labelsize=18)  # 设置x和y轴刻度标签大小
    ax2.grid(True, linestyle='--', alpha=0.3)
    cbar2 = plt.colorbar(density, ax=ax2, label='Density')
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Density', fontsize=16)
    
    # 打印统计信息
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    print(f"\n空间分布统计:")
    print(f"样本数量: {len(precisions)}")
    print(f"平均precision: {mean_precision:.3f} ± {std_precision:.3f}")
    print(f"最大precision: {np.max(precisions):.3f}")
    print(f"最小precision: {np.min(precisions):.3f}")
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return positions, precisions


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, 
                      default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\grasp_random_position_bottle_robot_table_57.pickle',
                      help='训练数据路径')
    parser.add_argument('--model_save_path', type=str,
                      default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\feasible_grasp_robot_table_bottle_57.pth',
                      help='模型保存路径')
    
    # 训练相关参数
    parser.add_argument('--random_seed', type=int, default=22, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--train_split', type=float, default=0.75, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=6, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=57, help='输出维度')
    parser.add_argument('--max_weight', type=float, default=10, help='BCE损失最大权重')
    
    # 训练控制参数
    parser.add_argument('--early_stop_patience', type=int, default=300, help='早停耐心值')
    parser.add_argument('--scheduler_patience', type=int, default=30, help='学习率调度器耐心值')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='学习率调度器衰减因子')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='最小学习率')


    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='regrasp', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, 
                      default='grasp_random_position_robot_resnet_with_self_attention_deeper',
                      help='wandb运行名称')
    
    # 添加新的参数
    parser.add_argument('--target_precision', type=float, default=0.8, 
                       help='目标精确率')
    parser.add_argument('--initial_threshold', type=float, default=0.6,
                       help='初始预测阈值')
    parser.add_argument('--min_precision', type=float, default=0.8, 
                       help='阈值选择时的最小precision要求')
    parser.add_argument('--min_recall', type=float, default=0.7, 
                       help='阈值选择时的最小recall要求')
    
    # 添加网络架构选择参数
    parser.add_argument('--network_type', type=str, default='mlp',
                      choices=['mlp', 'resnet', 'resnet_attention'],
                      help='选择网络架构类型')
    parser.add_argument('--use_stable_label', action='store_true',
                        help='是否使用stable label')
    
    return parser.parse_args()


def get_model(args):    
    """根据参数选择网络架构"""
    if args.network_type == 'mlp':
        from wrs.HuGroup_Qin.Shared_grasp_project.network.MlpBlock import GraspingNetwork
    elif args.network_type == 'resnet':
        from wrs.HuGroup_Qin.Shared_grasp_project.network.ResidualBlock import GraspingNetwork
    elif args.network_type == 'resnet_attention':
        from wrs.HuGroup_Qin.Shared_grasp_project.network.ResidualBlockWithSelfAttention import GraspingNetwork
    
    return GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)


if __name__ == '__main__':
    # 解析参数
    args = parse_args()
    set_seed(args.random_seed)

    # 加载数据集
    full_dataset = GraspingDataset(args.output_dim, args.use_stable_label, args.data_path)

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

    # model.load_state_dict(torch.load(args.model_save_path))
    # model.to(device)

    # 设置损失函数和优化器
    # criterion = lambda pred, target: (
    #     0.7 * BalancedBCELoss()(pred, target) +
    #     0.3 * FocalBCELoss()(pred, target)
    # )

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
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)  # 将所有参数记录到wandb
        )
    except Exception as e:
        print(f"Failed to initialize Weights and Biases: {e}")

    # 训练模型
    train_losses, val_precisions = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, args.model_save_path, args.initial_threshold,
        args.num_epochs, args.early_stop_patience,
        min_precision=args.min_precision,
        min_recall=args.min_recall
    )

    # # 已有数据集测试
    # model = get_model(args)
    # model.load_state_dict(torch.load(args.model_save_path))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    #
    # # 使用数据集测试模型
    # print("\n使用数据集进行测试...")
    # # dataset_results = test_model_with_dataset(model, test_loader, device='cpu')
    # dataset_results = evaluate_model(model, test_loader, device)

    # xy平面precision分布可视化
    # precision_matrix, sample_count = analyze_spatial_precision(
    # model, 
    # train_loader, # 用训练集数据可视化
    # device,
    # save_path=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\feasible_grasp\robot_spatial_precision_analysis.png')





