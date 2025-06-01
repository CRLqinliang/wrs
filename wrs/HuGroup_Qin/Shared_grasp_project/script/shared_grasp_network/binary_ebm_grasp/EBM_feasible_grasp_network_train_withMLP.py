import os
import sys
from os import error
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
from EBM_feasible_grasp_network_train import GraspEnergyNetwork
from ..shared_grasp.shared_grasp_network_train import SharedGraspEnergyDataset


class GraspEnergyNetworkWithMLP(nn.Module):
    def __init__(self, input_dim, pretrained_model_path, hidden_dims=[128, 128], dropout_rate=0.1):
        super().__init__()
        # 加载预训练的GraspEnergyNetwork
        self.energy_network = GraspEnergyNetwork(input_dim)
        checkpoint = torch.load(pretrained_model_path)
        self.energy_network.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结GraspEnergyNetwork的参数
        for param in self.energy_network.parameters():
            param.requires_grad = False
        self.energy_network.eval()
        
        # 构建MLP分类器
        mlp_layers = []
        prev_dim = 15  # 输入维度（Tinit + Tgoal + energy value）
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 添加输出层
        mlp_layers.append(nn.Linear(prev_dim, 1))
        mlp_layers.append(nn.Sigmoid())  # 使用Sigmoid激活函数得到0-1之间的输出
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x):

        # 获取前14维作为MLP的输入
        mlp_input = x[:, :14]
        
        # 通过能量网络获取能量值（不需要梯度）
        with torch.no_grad():
            energy_output = self.energy_network(x)

        # 将能量值与MLP的输入拼接
        mlp_input = torch.cat((mlp_input, energy_output), dim=1)
        
        # 通过MLP获取分类结果
        mlp_output = self.mlp(mlp_input)
        
        return mlp_output
    
    def get_energy_network(self):
        """
        获取能量网络实例
        
        Returns:
            GraspEnergyNetwork: 预训练的能量网络
        """
        return self.energy_network
    

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, save_path, early_stop_patience=50):
    """
    训练MLP分类器
    """
    model.to(device)
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        train_preds = []
        train_labels = []
        batch_count = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_loop:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # 收集预测结果
            train_preds.extend((outputs >= 0.5).cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())
            
            # 清理GPU内存
            del outputs, loss, inputs, labels
            
            train_loop.set_postfix({'loss': current_loss})
        
        # 计算训练指标
        train_metrics = calculate_classification_metrics(
            np.array(train_preds), 
            np.array(train_labels)
        )
        avg_train_loss = epoch_loss / batch_count

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(inputs)
                current_val_loss = criterion(outputs, labels).item()
                val_loss += current_val_loss
                
                # 收集预测结果
                val_preds.extend((outputs >= 0.5).cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
                
                # 清理GPU内存
                del outputs, inputs, labels
                
                val_loop.set_postfix({'loss': current_val_loss})
        
        # 计算验证指标
        val_metrics = calculate_classification_metrics(
            np.array(val_preds), 
            np.array(val_labels)
        )
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        
        # 记录详细指标到wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics['precision'],
            "train/recall": train_metrics['recall'],
            "train/f1": train_metrics['f1'],
            
            "val/loss": avg_val_loss,
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "val/f1": val_metrics['f1'],
            
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, F1: {val_metrics['f1']:.4f}")

        # 验证阶段后的早停检查
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # 保存最佳模型
            save_info = {
                'model_state_dict': model.state_dict(),
                'best_val_f1': best_val_f1,
                'epoch': epoch,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }
            print(f"\n保存最佳模型 - 验证集F1: {best_val_f1:.4f}")
            torch.save(save_info, save_path)
            
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\n早停: 验证集F1分数 {early_stop_patience} 个epoch没有改善")
            break

        # 清理内存
        del train_preds, train_labels, val_preds, val_labels
        gc.collect()
        torch.cuda.empty_cache()
    
    # 训练结束后，加载最佳模型并返回
    best_model_info = torch.load(save_path)
    model.load_state_dict(best_model_info['model_state_dict'])
    print(f"\n训练完成 - 加载最佳模型(Epoch {best_model_info['epoch']})")
    print(f"最佳验证集F1: {best_model_info['best_val_f1']:.4f}")
    
    return model


def evaluate_model(model, test_loader, device, args):
    """评估模型性能"""
    # 加载保存的模型信息
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"加载模型 - 最佳F1: {checkpoint['best_val_f1']:.4f}, Epoch: {checkpoint['epoch']}")
    
    model = model.to(device)
    model.eval()
    test_preds = []
    test_labels = []
    test_loss = 0.0
    criterion = nn.BCELoss()
    
    # 收集所有预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 收集预测结果
            test_preds.extend((outputs >= 0.5).cpu().numpy().flatten())
            test_labels.extend(labels.cpu().numpy().flatten())
    
    # 计算平均损失
    avg_test_loss = test_loss / len(test_loader)
    
    # 计算分类指标
    test_metrics = calculate_classification_metrics(
        np.array(test_preds),
        np.array(test_labels)
    )
    
    print("\n=== 测试集评估结果 ===")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    return test_metrics


def calculate_classification_metrics(predictions, labels):
    """
    计算分类指标
    """
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


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
            grasp_candidiate_num = get_id_regex(data_path_list[index-1])
            
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
            if ratio is not None:
                start_idx = int(len(raw_data) * ratio)
                raw_data = raw_data[start_idx:]
            
            # 使用numpy进行批量处理
            if grasp_candidiate_num > 0:
                for item in raw_data:
                    if isinstance(item[1], (list, np.ndarray)):
                        # 转换为numpy数组进行批量操作
                        indices = np.array(item[1])
                        indices = indices + grasp_candidiate_num
                        item[1] = indices.tolist()
                    elif isinstance(item[1], int):
                        item[1] = [item[1] + grasp_candidiate_num]
            
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
    """创建数据集"""
    # 创建训练集
    train_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_data_paths,  # 现在是逗号分隔的字符串
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label
    )

    # 创建验证集
    val_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_data_paths,
        use_quaternion=args.use_quaternion,
        use_stable_label=args.use_stable_label
    )
    
    # 创建测试集
    test_dataset = SharedGraspEnergyDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_data_paths,
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
        return pose_dim + stable_label_dim + grasp_pose_dim
    else:
        return pose_dim + grasp_pose_dim


def setup_training(args):
    """设置模型、损失函数、优化器等训练组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 自动计算输入维度
    input_dim = calculate_input_dim(args)
    print(f"计算得到的输入维度: {input_dim}")
    
    model = GraspEnergyNetworkWithMLP(
        input_dim=input_dim,
        pretrained_model_path=args.pretrained_model_path
    )
    
    # 初始化损失函数
    criterion = nn.BCELoss()
    
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
    
    # 数据相关参数 - 修改为列表形式
    parser.add_argument('--data_paths', nargs='+', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_57.pickle",
                       help='数据文件路径列表')
    parser.add_argument('--grasp_data_paths', nargs='+', type=str, default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_57.pickle",
                       help='抓取数据文件路径列表')
    parser.add_argument('--pretrained_model_path', type=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_57_h3_b2048_lr0.001_t0.5_r0.3_s0.7_q1_sl1_grobot_table_stinit.pth",
                       help='预训练模型路径')
    parser.add_argument('--model_save_path', type=str,
                       default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_MLP_best_model\best_model_grasp_ebm_SharedGraspNetworkWithMLP_bottle_experiment_data_57_h3_b1024_lr0.001_t0.5_r0.3_s0.7_q1_sl1.pth')
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.15)
    

    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--data_range', type=int, default=1e5, help='数据采样个数')
    parser.add_argument('--data_ratio', type=float, default=0.9, help='数据采样个数')
    

    # 模型结构参数 - 更新为简化的MLP参数
    parser.add_argument('--input_dim', type=int, default=19)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512])
    parser.add_argument('--num_layers', type=int, default=3)  # 替换num_res_blocks
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--use_quaternion', type=int, default=1, help='使用四元数表示 (1) 或简化表示 (0)')
    parser.add_argument('--use_stable_label', type=int, default=1, help='使用稳定性标签 (1) 或不使用 (0)')
    
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
    
    # 数据集参数
    parser.add_argument('--grasp_type', type=str, default='robot_table',
                       choices=['robot_table', 'table', 'robot'],
                       help='使用哪种抓取类型(robot_table, table, robot)')
    parser.add_argument('--state_type', type=str, default='init',
                       choices=['init', 'goal', 'both'],
                       help='处理哪种状态(init, goal, both)')
    args = parser.parse_args()

    # 将整数转换为布尔值
    args.use_quaternion = bool(args.use_quaternion)
    args.use_stable_label = bool(args.use_stable_label)
    
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 初始化wandb
    # wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    
    # 加载并合并多个数据集
    raw_data = load_raw_data(args.data_paths, args.data_ratio, args.data_range)
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
