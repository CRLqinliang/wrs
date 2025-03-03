import os
import sys
sys.path.append("E:/Qin/wrs")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import pickle
import numpy as np
import wandb
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc, random, time
import wrs.basis.robot_math as rm
from sklearn.preprocessing import StandardScaler


class VoxelBetaVAE_Conv3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, beta=4.0):
        super(VoxelBetaVAE_Conv3D, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(128*8*8*8, latent_dim)  # 假设输入尺寸64x64x64，经过3次下采样得到8x8x8
        self.fc_var = nn.Linear(128*8*8*8, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128*8*8*8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose3d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 8, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta * KLD, BCE, KLD


class VoxelBetaVAE_MLP(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, beta=4.0, input_size=64):
        super(VoxelBetaVAE_MLP, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_size = input_size
        
        # Calculate flattened input size
        self.flattened_size = in_channels * (input_size ** 3)
        
        # Encoder (MLP with SELU)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.SELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 128),
            nn.SELU(),
            nn.Dropout(0.1),
        )

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # Decoder (MLP with SELU)
        self.decoder_input = nn.Linear(latent_dim, 128)
        
        self.decoder = nn.Sequential(
            nn.SELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 512),
            nn.SELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.SELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, self.flattened_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        # Reshape back to 3D volume
        x = x.view(-1, 1, self.input_size, self.input_size, self.input_size)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta * KLD, BCE, KLD


# 条件能量模型
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
    

# 条件能量模型损失函数
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


# 体素抓取数据集
class VoxelGraspDataset(Dataset):
    def __init__(self, data, grasp_info_path=None, transform=None):
        """
        加载从ObstacleGraspNetwork_data_collection.py收集的数据
        
        参数:
            data: 可以是.npz文件路径(字符串)或预处理好的数据列表(从load_raw_data返回)
            grasp_info_path: 包含抓取姿态详细信息的文件路径
            transform: 对体素数据的转换操作
        """
        self.transform = transform
        
        print("正在加载数据集...")
        
        # 处理输入数据
        if isinstance(data, str):
            # 如果是文件路径，直接加载.npz文件
            npz_data = np.load(data, allow_pickle=True)
            self.voxel_data = npz_data['voxel_data']
            self.grasp_ids = npz_data['grasp_ids']
            self.labels_data = npz_data.get('labels', None)
        elif isinstance(data, list):
            # 如果是数据列表(从load_raw_data返回的格式)
            self.voxel_data = []
            self.grasp_ids = []
            self.labels_data = []
            
            # 收集所有样本的数据
            for sample in data:
                self.voxel_data.append(sample['voxel_data'])
                self.grasp_ids.append(sample['grasp_ids'])
                if 'labels' in sample:
                    self.labels_data.append(sample['labels'])
            
            # 如果没有收集到标签，设为None
            if len(self.labels_data) == 0:
                self.labels_data = None
        else:
            raise ValueError("data参数必须是文件路径(字符串)或预处理好的数据列表")
        
        print(f"加载了 {len(self.voxel_data)} 个场景")
        
        # 加载抓取姿态信息
        self.grasp_info = None
        if grasp_info_path:
            print("正在加载抓取姿态信息...")
            with open(grasp_info_path, 'rb') as f:
                self.grasp_info = pickle.load(f)
                
            # 预处理所有抓取信息，转换为统一格式的 7 维向量 (位置+四元数)
            self.processed_grasps = {}
            for gid, grasp in enumerate(self.grasp_info._grasp_list):
                # 获取姿态并转换为7维向量(位置+四元数)
                jaw_pos = grasp.ac_pos  # 位置 (3维)
                jaw_rotmat = grasp.ac_rotmat  # 旋转矩阵 (3x3)
                # 将旋转矩阵转换为四元数
                quat = rm.rotmat_to_quaternion(jaw_rotmat)  # 四元数 (4维)
                self.processed_grasps[gid] = np.concatenate([jaw_pos, quat])
                
            print(f"处理了 {len(self.processed_grasps)} 个抓取姿态")
        
        # 预处理数据，生成(样本, 标签)对
        print("正在准备数据...")
        self._prepare_data()
        print(f"数据准备完成: {len(self.features)} 个样本")
        
        # 统计正负样本数量
        pos_count = sum(1 for label in self.labels if label == 1)
        neg_count = sum(1 for label in self.labels if label == 0)
        print(f"正样本数量: {pos_count}, 负样本数量: {neg_count}")
        
        # 为BalancedBatchSampler添加标签属性
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)

    def _prepare_data(self):
        """预处理数据，为每个体素场景的所有抓取姿态生成样本对"""
        # 计算所有抓取候选项的数量
        total_grasps = len(self.processed_grasps) if self.processed_grasps else 0
        if total_grasps == 0:
            raise ValueError("没有找到有效的抓取姿态数据")
        
        print(f"总共有 {total_grasps} 个抓取候选项")
        
        # 预分配特征和标签列表
        self.features = []
        self.labels = []
        
        # 批量处理数据
        for idx in tqdm(range(len(self.voxel_data)), desc="处理样本"):
            voxel = self.voxel_data[idx]
            available_gids = self.grasp_ids[idx] if self.grasp_ids[idx] is not None else []
            
            # 将available_gids转换为集合以提高查找效率
            available_gids_set = set(available_gids)
            
            # 处理体素数据
            voxel_tensor = torch.FloatTensor(voxel.astype(np.float32))
            voxel_tensor = voxel_tensor.unsqueeze(0)  # 添加通道维度
            
            # 为该场景中的所有抓取姿态生成样本
            for gid, grasp_pose in self.processed_grasps.items():
                # 添加特征
                self.features.append((voxel_tensor, torch.FloatTensor(grasp_pose)))
                
                # 设置标签 - 如果gid在available_gids中则为1，否则为0
                if self.labels_data is not None and idx < len(self.labels_data):
                    # 如果有明确的标签数据，使用它
                    label = self.labels_data[idx].get(str(gid), 0)  # 默认为0如果找不到
                else:
                    # 否则，使用available_gids判断
                    label = 1 if gid in available_gids_set else 0
                
                self.labels.append(label)
            
            # 清理临时变量
            del voxel_tensor
        
        # 转换标签为numpy数组
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # 清理中间变量
        gc.collect()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        voxel, grasp_pose = self.features[idx]
        label = self.labels[idx]
        
        # 确保数据已经是CPU上的torch张量
        if not isinstance(voxel, torch.Tensor):
            voxel = torch.FloatTensor(voxel)
        if not isinstance(grasp_pose, torch.Tensor):
            grasp_pose = torch.FloatTensor(grasp_pose)
        
        # 预先应用变换
        if self.transform:
            voxel = self.transform(voxel)
        
        return voxel, grasp_pose, torch.tensor(label, dtype=torch.float32)


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 获取所有标签
        self.labels = dataset.labels_tensor
        
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
    
    
# 计算评估指标
def calculate_metrics(energies, labels, threshold=None):
    # 将NumPy数组转换为一维数组
    energies = energies.flatten()
    labels = labels.flatten()
    
    # 计算ROC曲线和最佳阈值
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, -energies)  # 注意能量越低越好，所以取负
        roc_auc = auc(fpr, tpr)
        
        # 找到最佳阈值
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = threshold
        fpr, tpr, _ = roc_curve(labels, -energies)
        roc_auc = auc(fpr, tpr)
    
    # 使用最佳阈值计算预测值
    predictions = (energies <= optimal_threshold).astype(int)
    
    # 计算 accuracy, precision, recall, f1
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算正负样本的能量统计
    pos_energies = energies[labels == 1]
    neg_energies = energies[labels == 0]
    
    pos_energy_mean = np.mean(pos_energies) if len(pos_energies) > 0 else 0
    pos_energy_std = np.std(pos_energies) if len(pos_energies) > 0 else 0
    neg_energy_mean = np.mean(neg_energies) if len(neg_energies) > 0 else 0
    neg_energy_std = np.std(neg_energies) if len(neg_energies) > 0 else 0
    energy_gap = neg_energy_mean - pos_energy_mean
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'pos_energy_mean': pos_energy_mean,
        'pos_energy_std': pos_energy_std,
        'neg_energy_mean': neg_energy_mean,
        'neg_energy_std': neg_energy_std,
        'energy_gap': energy_gap
    }


# 训练函数
def train_model(vae, energy_model, train_loader, val_loader, criterion_vae, criterion_energy, optimizer_vae, 
                optimizer_energy, scheduler, device, num_epochs, save_path, 
                lambda_energy=1.0, early_stop_patience=10):
    
    vae.to(device)
    energy_model.to(device)
    
    best_val_metric = float('-inf')  # 能量间隔是越大越好
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        vae.train()
        energy_model.train()
        
        train_vae_loss = 0.0
        train_energy_loss = 0.0
        train_total_loss = 0.0
        train_energies_list = []
        train_labels_list = []
        samples_count = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for voxels, grasp_poses, labels in train_loop:
            voxels = voxels.to(device)
            grasp_poses = grasp_poses.to(device)
            labels = labels.to(device)
            batch_size = voxels.size(0)
            samples_count += batch_size
            
            # 清除梯度
            optimizer_vae.zero_grad()
            optimizer_energy.zero_grad()
            
            # VAE前向传播
            recon_voxels, mu, log_var, z = vae(voxels)
            
            # VAE损失
            loss_vae, bce, kld = vae.loss_function(recon_voxels, voxels, mu, log_var)
            
            # 将潜在向量和抓取姿态连接起来作为能量模型的输入
            energy_input = torch.cat([z, grasp_poses], dim=1)
            
            # 能量模型前向传播
            energies = energy_model(energy_input)
            loss_energy = criterion_energy(energies, labels)
            
            # 总损失
            loss = loss_vae + lambda_energy * loss_energy
            
            # 反向传播和优化
            loss.backward()
            optimizer_vae.step()
            optimizer_energy.step()
            
            # 记录损失和能量值
            train_vae_loss += loss_vae.item()
            train_energy_loss += loss_energy.item()
            train_total_loss += loss.item()
            
            # 收集训练数据时立即转为numpy并释放tensor
            train_energies_list.append(energies.detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())
            
            # 打印当前批次损失
            train_loop.set_postfix({
                'vae_loss': loss_vae.item() / batch_size,
                'energy_loss': loss_energy.item() / batch_size
            })
            
            # 清理内存
            del voxels, grasp_poses, labels, recon_voxels, mu, log_var, z, energies, loss_vae, loss_energy, loss
            torch.cuda.empty_cache()
            
        # 计算平均损失
        avg_train_vae_loss = train_vae_loss / samples_count
        avg_train_energy_loss = train_energy_loss / samples_count
        avg_train_total_loss = train_total_loss / samples_count
        
        # 计算训练集能量指标
        train_energies = np.concatenate(train_energies_list)
        train_labels = np.concatenate(train_labels_list)
        train_metrics = calculate_metrics(train_energies, train_labels)
        
        # 清理训练数据列表
        del train_energies_list, train_labels_list
        gc.collect()
        
        # 验证阶段
        vae.eval()
        energy_model.eval()
        
        val_vae_loss = 0.0
        val_energy_loss = 0.0
        val_total_loss = 0.0
        val_energies_list = []
        val_labels_list = []
        val_samples_count = 0
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for val_voxels, val_grasp_poses, val_labels in val_loop:
                val_voxels = val_voxels.to(device)
                val_grasp_poses = val_grasp_poses.to(device)  
                val_labels = val_labels.to(device)
                val_batch_size = val_voxels.size(0)
                val_samples_count += val_batch_size
                
                # VAE前向传播
                val_recon_voxels, val_mu, val_log_var, val_z = vae(val_voxels)
                
                # VAE损失
                val_loss_vae, _, _ = vae.loss_function(val_recon_voxels, val_voxels, val_mu, val_log_var)
                
                # 将潜在向量和抓取姿态连接起来作为能量模型的输入
                val_energy_input = torch.cat([val_z, val_grasp_poses], dim=1)
                
                # 能量模型前向传播
                val_energies = energy_model(val_energy_input)
                val_loss_energy = criterion_energy(val_energies, val_labels)
                
                # 总损失
                val_loss = val_loss_vae + lambda_energy * val_loss_energy
                
                # 记录损失和能量值
                val_vae_loss += val_loss_vae.item()
                val_energy_loss += val_loss_energy.item()
                val_total_loss += val_loss.item()
                
                # 收集验证数据时立即转为numpy并释放tensor
                val_energies_list.append(val_energies.cpu().numpy())
                val_labels_list.append(val_labels.cpu().numpy())
                
                # 清理内存
                del val_voxels, val_grasp_poses, val_labels, val_recon_voxels, val_mu, val_log_var, val_z, val_energies
                del val_loss_vae, val_loss_energy, val_loss
                torch.cuda.empty_cache()
        
        # 计算平均损失
        avg_val_vae_loss = val_vae_loss / val_samples_count
        avg_val_energy_loss = val_energy_loss / val_samples_count
        avg_val_total_loss = val_total_loss / val_samples_count
        
        # 计算验证指标
        val_energies = np.concatenate(val_energies_list)
        val_labels = np.concatenate(val_labels_list)
        val_metrics = calculate_metrics(val_energies, val_labels)
        
        # 清理验证数据列表
        del val_energies_list, val_labels_list
        gc.collect()
        
        # 更新学习率调度器
        scheduler.step(avg_val_total_loss)
        
        # 记录到wandb
        wandb.log({
            "epoch": epoch,
            "train/vae_loss": avg_train_vae_loss,
            "train/energy_loss": avg_train_energy_loss,
            "train/total_loss": avg_train_total_loss,
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics['precision'],
            "train/recall": train_metrics['recall'],
            "train/f1": train_metrics['f1'],
            "train/auc": train_metrics['auc'],
            "train/pos_energy_mean": train_metrics['pos_energy_mean'],
            "train/neg_energy_mean": train_metrics['neg_energy_mean'],
            "train/energy_gap": train_metrics['energy_gap'],
            "train/optimal_threshold": train_metrics['optimal_threshold'],
            
            "val/vae_loss": avg_val_vae_loss,
            "val/energy_loss": avg_val_energy_loss,
            "val/total_loss": avg_val_total_loss,
            "val/accuracy": val_metrics['accuracy'],
            "val/precision": val_metrics['precision'],
            "val/recall": val_metrics['recall'],
            "val/f1": val_metrics['f1'],
            "val/auc": val_metrics['auc'],
            "val/pos_energy_mean": val_metrics['pos_energy_mean'],
            "val/neg_energy_mean": val_metrics['neg_energy_mean'],
            "val/energy_gap": val_metrics['energy_gap'],
            "val/optimal_threshold": val_metrics['optimal_threshold'],
            
            "learning_rate": optimizer_vae.param_groups[0]['lr']
        })
        
        # 打印当前epoch的主要指标
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - VAE Loss: {avg_train_vae_loss:.4f}, Energy Loss: {avg_train_energy_loss:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, Energy Gap: {train_metrics['energy_gap']:.4f}")
        print(f"Val - VAE Loss: {avg_val_vae_loss:.4f}, Energy Loss: {avg_val_energy_loss:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Energy Gap: {val_metrics['energy_gap']:.4f}")
        
        # 早停与模型保存
        current_metric = val_metrics['f1']
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'energy_model_state_dict': energy_model.state_dict(),
                'epoch': epoch,
                'best_metric': best_val_metric,
                'val_metrics': val_metrics,
                'optimal_threshold': val_metrics['optimal_threshold']
            }, os.path.join(save_path, 'best_model.pth'))
            
            print(f"最佳模型已保存 - 能量间隔: {best_val_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停 - {early_stop_patience} 个epoch没有改善")
                break
    
    # 保存最终模型
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'energy_model_state_dict': energy_model.state_dict(),
        'epoch': epoch,
        'final_metric': current_metric,
        'optimal_threshold': val_metrics['optimal_threshold']
    }, os.path.join(save_path, 'final_model.pth'))
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    vae.load_state_dict(checkpoint['vae_state_dict'])
    energy_model.load_state_dict(checkpoint['energy_model_state_dict'])
    
    return vae, energy_model, checkpoint['best_metric']


def evaluate_model(vae, energy_model, test_loader, device, save_path=None):
    vae.to(device)
    energy_model.to(device)
    vae.eval()
    energy_model.eval()
    
    test_energies_list = []
    test_labels_list = []
    reconstructions = []
    originals = []
    latent_vectors = []
    grasp_poses_list = []
    
    with torch.no_grad():
        for voxels, grasp_poses, labels in tqdm(test_loader, desc="Evaluating"):
            voxels = voxels.to(device)
            grasp_poses = grasp_poses.to(device)
            labels = labels.to(device)
            
            # VAE 前向传播
            recon_voxels, mu, log_var, z = vae(voxels)
            
            # 将潜在向量和抓取姿态连接起来作为能量模型的输入
            energy_input = torch.cat([z, grasp_poses], dim=1)
            
            # 能量模型前向传播
            energies = energy_model(energy_input)
            
            # 收集数据用于指标计算
            test_energies_list.append(energies.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())
            
            # 收集数据用于可视化 (仅保存少量样本)
            if len(reconstructions) < 10:
                reconstructions.append(recon_voxels.cpu().numpy())
                originals.append(voxels.cpu().numpy())
                latent_vectors.append(z.cpu().numpy())
                grasp_poses_list.append(grasp_poses.cpu().numpy())
            
            # 清理内存
            del voxels, grasp_poses, labels, recon_voxels, mu, log_var, z, energies, energy_input
            torch.cuda.empty_cache()
    
    # 计算指标
    test_energies = np.concatenate(test_energies_list)
    test_labels = np.concatenate(test_labels_list)
    metrics = calculate_metrics(test_energies, test_labels)
    
    # 清理内存
    del test_energies_list, test_labels_list
    gc.collect()
    
    # 输出指标
    print("\n===== 测试集评估结果 =====")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确度: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"能量间隔: {metrics['energy_gap']:.4f}")
    print(f"正样本平均能量: {metrics['pos_energy_mean']:.4f} ± {metrics['pos_energy_std']:.4f}")
    print(f"负样本平均能量: {metrics['neg_energy_mean']:.4f} ± {metrics['neg_energy_std']:.4f}")
    print(f"最佳阈值: {metrics['optimal_threshold']:.4f}")
    
    # 保存评估结果
    if save_path:
        results = {
            'metrics': metrics,
            'energy_threshold': metrics['optimal_threshold'],
            'visualizations': {
                'reconstructions': reconstructions,
                'originals': originals,
                'latent_vectors': latent_vectors,
                'grasp_poses': grasp_poses_list
            }
        }
        torch.save(results, os.path.join(save_path, 'evaluation_results.pth'))
    
    return metrics


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_data(data_path, ratio=0.5):
    """加载原始数据
    Args:
        data_path: 数据文件路径
        ratio: 使用数据的比例，默认0.5表示使用后半部分数据
    """
    # 使用numpy加载.npz文件而非pickle
    raw_data = np.load(data_path, allow_pickle=True)
    
    # 获取voxel_data和grasp_ids，这是你需要的主要数据
    voxel_data = raw_data['voxel_data']
    grasp_ids = raw_data['grasp_ids']
    
    # 如果有labels数据，也加载它
    labels = raw_data.get('labels', None)
    
    # 根据ratio截取数据
    total_size = len(voxel_data)
    start_idx = int(total_size * ratio)
    
    # 创建截取后的数据字典列表
    dataset = []
    for i in range(start_idx, total_size):
        sample = {
            'voxel_data': voxel_data[i],
            'grasp_ids': grasp_ids[i]
        }
        if labels is not None:
            sample['labels'] = labels[i]
        dataset.append(sample)
    
    print(f"加载数据: 总量 {len(dataset)}, 起始比例 {1 - ratio:.1%}")
    return dataset


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
    train_dataset = VoxelGraspDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_info_path
    )

    # 使用训练集的标准化器创建验证集和测试集
    val_dataset = VoxelGraspDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_info_path
    )
    
    test_dataset = VoxelGraspDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_info_path
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
   # val_sampler = BalancedBatchSampler(val_dataset, args.batch_size)
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
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="体素条件能量模型训练")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                       default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\obstacle_grasp_data.npz",
                       help='体素数据的npz文件路径')
    parser.add_argument('--grasp_info_path', type=str, 
                       default=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle",
                       help='抓取信息的pickle文件路径')
    parser.add_argument('--train_split', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_split', type=float, default=0.15, help='验证集比例')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='conv3d', choices=['conv3d', 'mlp'], 
                        help='VAE模型类型: conv3d或mlp')
    parser.add_argument('--latent_dim', type=int, default=32, help='潜在空间维度')
    parser.add_argument('--beta', type=float, default=1.0, help='beta-VAE的beta参数')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512], help='能量模型的隐藏层维度')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout率')
    parser.add_argument('--input_size', type=int, default=64, help='输入体素的尺寸')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lambda_energy', type=float, default=0.5, help='能量损失权重')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--data_ratio', type=float, default=0.5, help='数据比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 保存和日志
    parser.add_argument('--save_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\vae_ebm_model', help='模型保存路径')
    parser.add_argument('--wandb_project', type=str, default='voxel-grasp-energy', help='Wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb运行名称')
    
    # 运行模式
    parser.add_argument('--train', type=bool, default=True, help='训练模型')
    parser.add_argument('--eval', type=bool, default=True, help='评估模型')

    # 添加新参数控制数据预取和处理
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的worker数量')
    parser.add_argument('--prefetch_factor', type=int, default=3, help='数据预取因子')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化wandb
    run_name = args.wandb_name or f"vae-ebm-{args.model_type}-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # 加载原始数据
    raw_data = load_raw_data(args.data_path, args.data_ratio)
    indices = split_data_indices(len(raw_data), args.train_split, args.val_split)
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_datasets(raw_data, indices, args)
    
    # 及时清理原始数据
    del raw_data, indices
    gc.collect()

    print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
    
    # 使用BalancedBatchSampler创建数据加载器
    train_sampler = BalancedBatchSampler(train_dataset, args.batch_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 创建模型 - 根据model_type选择不同的VAE实现
    if args.model_type == 'conv3d':
        print("使用3D卷积VAE模型")
        vae = VoxelBetaVAE_Conv3D(in_channels=1, latent_dim=args.latent_dim, beta=args.beta).to(device)
    else:  # 'mlp'
        print("使用MLP VAE模型")
        vae = VoxelBetaVAE_MLP(in_channels=1, latent_dim=args.latent_dim, beta=args.beta, 
                              input_size=args.input_size).to(device)
    
    energy_model = GraspEnergyNetwork(
        input_dim=args.latent_dim + 7,  # 潜在空间维度 + 7维抓取姿态 (位置+四元数)
        hidden_dims=args.hidden_dims,
        num_layers=len(args.hidden_dims),
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 创建优化器
    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr)
    optimizer_energy = optim.Adam(energy_model.parameters(), lr=args.lr)
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer_vae, 'min', factor=0.5, patience=5, verbose=True
    )
    
    # 创建损失函数
    criterion_vae = None  # VAE已经有内置的损失函数
    criterion_energy = EnergyBasedLoss()
    
    # 训练模型
    if args.train:
        print("开始训练模型...")
        vae, energy_model, best_metric = train_model(
            vae=vae,
            energy_model=energy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_vae=criterion_vae,
            criterion_energy=criterion_energy,
            optimizer_vae=optimizer_vae,
            optimizer_energy=optimizer_energy,
            scheduler=scheduler,
            device=device,
            num_epochs=args.num_epochs,
            save_path=args.save_path,
            lambda_energy=args.lambda_energy,
            early_stop_patience=args.early_stop_patience
        )
        print(f"训练完成，最佳指标: {best_metric:.4f}")
    
    # 评估模型
    if args.eval:
        print("开始评估模型...")
        metrics = evaluate_model(
            vae=vae,
            energy_model=energy_model,
            test_loader=test_loader,
            device=device,
            save_path=args.save_path
        )
        print("评估完成!")
    
    # 关闭wandb
    wandb.finish()


if __name__ == "__main__":
    main()
