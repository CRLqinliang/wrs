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
import optuna
import json


class VoxelBetaVAE_Conv3D(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128, beta=4.0, input_size=[25, 25, 30]):
        super(VoxelBetaVAE_Conv3D, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_size = input_size
        
        # 计算编码器输出尺寸（经过3次下采样）
        h_out = input_size[0] // 8  # 25 -> 3
        w_out = input_size[1] // 8  # 25 -> 3
        d_out = input_size[2] // 8  # 30 -> 4
        self.encoded_shape = (h_out, w_out, d_out)
        self.flattened_size = 128 * h_out * w_out * d_out  # 128 * 3 * 3 * 4 = 4608
        
        # Encoder (保持不变)
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

        # 修改全连接层大小
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        
        # Decoder (保持不变)
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
    
    def decode(self, z):
        x = self.decoder_input(z)
        # 重塑为正确的形状
        x = x.view(-1, 128, self.encoded_shape[0], self.encoded_shape[1], self.encoded_shape[2])
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return BCE + self.beta * KLD, BCE, KLD


class VoxelBetaVAE_MLP(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64, beta=4.0, dropout=0.1, input_size=[25, 25, 30]):
        super(VoxelBetaVAE_MLP, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.input_shape = input_size

        # 计算展平后的输入大小
        self.flattened_size = in_channels * input_size[0] * input_size[1] * input_size[2]  # 1 * 25 * 25 * 30 = 18750

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 编码器输出大小调整
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # 解码器输入层
        self.decoder_input = nn.Linear(latent_dim, 128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.flattened_size),
            nn.Sigmoid()
        )

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        # 重塑为正确的3D体素形状
        x = x.view(-1, 1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return x

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
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
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
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


class StreamingVoxelGraspDataset(Dataset):
    def __init__(self, data, grasp_info_path=None, transform=None):
        """
        流式加载抓取数据集，避免一次性加载所有样本到内存
        
        参数:
            data: 可以是.npz文件路径(字符串)或预处理好的数据列表
            grasp_info_path: 包含抓取姿态详细信息的文件路径
            transform: 对体素数据的转换操作
        """
        self.transform = transform
        
        # 处理输入数据
        if isinstance(data, str):
            self.data_path = data
            with np.load(data, allow_pickle=True, mmap_mode='r') as npz_data:
                self.voxel_data_shape = npz_data['voxel_data'].shape
                self.num_scenes = self.voxel_data_shape[0]
            self.data_mode = 'file'
        elif isinstance(data, list):
            self.data_list = data
            self.num_scenes = len(data)
            self.data_mode = 'list'
        else:
            raise ValueError("data参数必须是文件路径(字符串)或预处理好的数据列表")
        
        # 加载抓取姿态信息
        if grasp_info_path:
            import sys
            sys.path.append("E:/Qin/wrs")
            import wrs.basis.robot_math as rm
            
            with open(grasp_info_path, 'rb') as f:
                self.grasp_info = pickle.load(f)
            
            # 预处理所有抓取信息，转换为统一格式的 7 维向量 (位置+四元数)
            self.processed_grasps = {}
            grasp_list = self.grasp_info._grasp_list
            
            for gid, grasp in enumerate(grasp_list):
                jaw_pos = grasp.ac_pos
                jaw_rotmat = grasp.ac_rotmat
                quat = rm.rotmat_to_quaternion(jaw_rotmat)
                self.processed_grasps[gid] = np.concatenate([jaw_pos, quat])
            
            self.num_grasps = len(self.processed_grasps)
        
        # 构建样本索引映射
        self._build_sample_indices()
        
        # 构建标签索引，用于平衡采样
        self._build_label_indices()
    
    def _load_scene(self, idx):
        """加载指定索引的场景体素数据"""
        if self.data_mode == 'file':
            return np.load(self.data_path, allow_pickle=True, mmap_mode='r')['voxel_data'][idx]
        else:  # self.data_mode == 'list'
            return self.data_list[idx]['voxel_data']
    
    def _load_grasp_ids(self, idx):
        """加载指定索引的抓取ID列表"""
        if self.data_mode == 'file':
            return np.load(self.data_path, allow_pickle=True, mmap_mode='r')['grasp_ids'][idx]
        else:  # self.data_mode == 'list'
            return self.data_list[idx]['grasp_ids']
    
    def _load_labels(self, idx):
        """加载指定索引的标签（如果有）"""
        if self.data_mode == 'file':
            return np.load(self.data_path, allow_pickle=True, mmap_mode='r').get('labels', None)
        else:  # self.data_mode == 'list'
            return self.data_list[idx]['labels'] if 'labels' in self.data_list[idx] else None
        
    def _build_sample_indices(self):
        """构建(场景idx, 抓取idx)样本索引，不加载实际数据"""
        self.sample_indices = []
        
        # 预先计算总样本数并分配索引
        for scene_idx in range(self.num_scenes):
            for grasp_idx in self.processed_grasps.keys():
                self.sample_indices.append((scene_idx, grasp_idx))
                
        self.num_samples = len(self.sample_indices)
        print(f"构建了 {self.num_samples} 个样本索引")
    
    def _build_label_indices(self):
        """构建标签索引，用于BalancedBatchSampler"""
        # 初始化标签数组，但不填充
        self.labels = np.zeros(self.num_samples, dtype=np.float32)

        # 依次处理每个场景的标签，但不加载体素数据
        for idx, (scene_idx, grasp_idx) in enumerate(tqdm(self.sample_indices)):
            # 延迟加载grasp_ids
            available_gids = self._load_grasp_ids(scene_idx)
            available_gids = available_gids if available_gids is not None else []
            available_gids_set = set(available_gids)
            
            # 确定标签
            label = 1 if grasp_idx in available_gids_set else 0
            self.labels[idx] = label
        
        # 构建索引张量，用于采样器
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
        
        # 统计正负样本数量
        pos_count = np.sum(self.labels == 1)
        neg_count = np.sum(self.labels == 0)
        print(f"正样本数量: {pos_count}, 负样本数量: {neg_count}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取样本索引
        scene_idx, grasp_idx = self.sample_indices[idx]
        
        # 按需加载体素数据
        voxel = self._load_scene(scene_idx)
        
        # 处理体素数据
        voxel_tensor = torch.FloatTensor(voxel.astype(np.float32))
        voxel_tensor = voxel_tensor.unsqueeze(0)  # 添加通道维度
        
        # 获取对应的抓取姿态
        grasp_pose = torch.FloatTensor(self.processed_grasps[grasp_idx])
        
        # 获取标签
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
        
        return voxel_tensor, grasp_pose, torch.tensor(label, dtype=torch.float32)


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
    
    # 确保所有返回值都是Python原生类型
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'pos_energy_mean': float(pos_energy_mean),
        'pos_energy_std': float(pos_energy_std),
        'neg_energy_mean': float(neg_energy_mean),
        'neg_energy_std': float(neg_energy_std),
        'energy_gap': float(energy_gap)
    }


# 训练函数
def train_model(vae, energy_model, train_loader, val_loader, criterion_vae, criterion_energy, optimizer_vae, 
                optimizer_energy, scheduler, device, num_epochs, save_path, 
                lambda_energy=1.0, early_stop_patience=10, trial=None):
    
    vae.to(device)
    energy_model.to(device)
    
    best_val_metric = float('-inf')
    patience_counter = 0
    peak_memory_usage = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        vae.train()
        energy_model.train()
        
        train_vae_loss = 0.0
        train_energy_loss = 0.0
        train_total_loss = 0.0

        train_bce_loss = 0.0
        train_kld_loss = 0.0
        train_energies_list = []
        train_labels_list = []
        samples_count = 0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for voxels, grasp_poses, labels in train_loop:
            voxels = voxels.to(device)
            grasp_poses = grasp_poses.to(device)
            labels = labels.to(device)
            batch_size = voxels.size(0)
            samples_count += batch_size
            
            optimizer_vae.zero_grad()
            optimizer_energy.zero_grad()
            
            recon_voxels, mu, log_var, z = vae(voxels)
            loss_vae, bce, kld = vae.loss_function(recon_voxels, voxels, mu, log_var)
            
            energy_input = torch.cat([z, grasp_poses], dim=1)
            energies = energy_model(energy_input)
            loss_energy = criterion_energy(energies, labels)
            
            loss = loss_vae + lambda_energy * loss_energy
            
            loss.backward()
            optimizer_vae.step()
            optimizer_energy.step()
            
            train_vae_loss += loss_vae.item()
            train_energy_loss += loss_energy.item()
            train_total_loss += loss.item()
            
            train_bce_loss += bce.item()
            train_kld_loss += kld.item()

            train_energies_list.append(energies.detach().cpu().numpy())
            train_labels_list.append(labels.cpu().numpy())
            
            train_loop.set_postfix({
                'vae_loss': loss_vae.item() / batch_size,
                'energy_loss': loss_energy.item() / batch_size
            })
            
            del voxels, grasp_poses, labels, recon_voxels, mu, log_var, z, energies, loss_vae, loss_energy, loss
            torch.cuda.empty_cache()
            
        avg_train_vae_loss = train_vae_loss / samples_count
        avg_train_energy_loss = train_energy_loss / samples_count
        avg_train_total_loss = train_total_loss / samples_count
        avg_train_bce_loss = train_bce_loss / samples_count
        avg_train_kld_loss = train_kld_loss / samples_count

        train_energies = np.concatenate(train_energies_list)
        train_labels = np.concatenate(train_labels_list)
        train_metrics = calculate_metrics(train_energies, train_labels)

        del train_energies_list, train_labels_list
        gc.collect()
        
        # 验证阶段
        vae.eval()
        energy_model.eval()
        
        val_vae_loss = 0.0
        val_energy_loss = 0.0
        val_total_loss = 0.0
        val_bce_loss = 0.0
        val_kld_loss = 0.0
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
                
                val_recon_voxels, val_mu, val_log_var, val_z = vae(val_voxels)
                val_loss_vae, val_bce, val_kld = vae.loss_function(val_recon_voxels, val_voxels, val_mu, val_log_var)
                
                val_energy_input = torch.cat([val_z, val_grasp_poses], dim=1)
                val_energies = energy_model(val_energy_input)
                val_loss_energy = criterion_energy(val_energies, val_labels)

                val_loss = val_loss_vae + lambda_energy * val_loss_energy
                
                val_vae_loss += val_loss_vae.item()
                val_energy_loss += val_loss_energy.item()
                val_total_loss += val_loss.item()

                val_bce_loss += val_bce.item()
                val_kld_loss += val_kld.item()
                
                val_energies_list.append(val_energies.cpu().numpy())
                val_labels_list.append(val_labels.cpu().numpy())
                
                del val_voxels, val_grasp_poses, val_labels, val_recon_voxels, val_mu, val_log_var, val_z, val_energies
                del val_loss_vae, val_loss_energy, val_loss
                torch.cuda.empty_cache()
        
        avg_val_vae_loss = val_vae_loss / val_samples_count
        avg_val_energy_loss = val_energy_loss / val_samples_count
        avg_val_total_loss = val_total_loss / val_samples_count
        avg_val_bce_loss = val_bce_loss / val_samples_count  
        avg_val_kld_loss = val_kld_loss / val_samples_count

        val_energies = np.concatenate(val_energies_list)
        val_labels = np.concatenate(val_labels_list)
        val_metrics = calculate_metrics(val_energies, val_labels)
        
        del val_energies_list, val_labels_list
        gc.collect()
        
        scheduler.step(avg_val_total_loss)
        
        wandb.log({
            "epoch": epoch,
            "train/vae_loss": avg_train_vae_loss,
            "train/energy_loss": avg_train_energy_loss,
            "train/total_loss": avg_train_total_loss,
            "train/vae_bce_loss": avg_train_bce_loss,
            "train/vae_kld_loss": avg_train_kld_loss,
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
            "val/vae_bce_loss": avg_val_bce_loss,
            "val/vae_kld_loss": avg_val_kld_loss,
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
            
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'energy_model_state_dict': energy_model.state_dict(),
                'epoch': epoch,
                'best_metric': best_val_metric,
                'val_metrics': val_metrics,
                'optimal_threshold': val_metrics['optimal_threshold']
            }, os.path.join(save_path, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"早停 - {early_stop_patience} 个epoch没有改善")
                break
        
        # 如果提供了trial对象，则报告指标并检查是否应该提前终止
        if trial is not None:
            # 报告当前epoch的F1分数
            trial.report(val_metrics['f1'], epoch)
            
            # 检查是否应该提前终止
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'energy_model_state_dict': energy_model.state_dict(),
        'epoch': epoch,
        'final_metric': current_metric,
        'optimal_threshold': val_metrics['optimal_threshold']
    }, os.path.join(save_path, 'final_model.pth'))
    
    checkpoint = torch.load(os.path.join(save_path, 'best_model.pth'))
    vae.load_state_dict(checkpoint['vae_state_dict'])
    energy_model.load_state_dict(checkpoint['energy_model_state_dict'])
    
    return vae, energy_model, checkpoint['best_metric']


def evaluate_model(vae, energy_model, model_path, test_loader, device):
    # 加载模型和最佳阈值
    checkpoint = torch.load(model_path)
    
    # 获取模型状态字典
    vae_state_dict = checkpoint['vae_state_dict']
    energy_model_state_dict = checkpoint['energy_model_state_dict']
    
    # 获取保存的最佳阈值
    optimal_threshold = checkpoint.get('optimal_threshold', None)
    
    # 创建模型实例并加载权重
    # 注意：这里假设您已经有了vae和energy_model的实例，如果没有，需要先创建
    vae.load_state_dict(vae_state_dict)
    energy_model.load_state_dict(energy_model_state_dict)
    
    vae.to(device)
    energy_model.to(device)
    vae.eval()
    energy_model.eval()
    
    print(f"已加载模型，使用阈值: {optimal_threshold}")
    
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
            
            recon_voxels, mu, log_var, z = vae(voxels)
            energy_input = torch.cat([z, grasp_poses], dim=1)
            energies = energy_model(energy_input)
            
            test_energies_list.append(energies.cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())
            
            if len(reconstructions) < 10:
                reconstructions.append(recon_voxels.cpu().numpy())
                originals.append(voxels.cpu().numpy())
                latent_vectors.append(z.cpu().numpy())
                grasp_poses_list.append(grasp_poses.cpu().numpy())
            
            del voxels, grasp_poses, labels, recon_voxels, mu, log_var, z, energies, energy_input
            torch.cuda.empty_cache()
    
    test_energies = np.concatenate(test_energies_list)
    test_labels = np.concatenate(test_labels_list)
    
    # 使用模型自带的阈值计算指标
    metrics = calculate_metrics(test_energies, test_labels, threshold=optimal_threshold)
    
    del test_energies_list, test_labels_list
    gc.collect()
    
    print("\n===== 测试集评估结果 =====")
    print(f"使用模型阈值: {optimal_threshold:.4f}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确度: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"能量间隔: {metrics['energy_gap']:.4f}")
    print(f"正样本平均能量: {metrics['pos_energy_mean']:.4f} ± {metrics['pos_energy_std']:.4f}")
    print(f"负样本平均能量: {metrics['neg_energy_mean']:.4f} ± {metrics['neg_energy_std']:.4f}")
    
    # 保存评估结果
    results = {
        'metrics': metrics,
        'threshold_used': optimal_threshold,
        'visualizations': {
            'reconstructions': reconstructions[:10],
            'originals': originals[:10],
            'latent_vectors': latent_vectors[:10],
            'grasp_poses': grasp_poses_list[:10]
        }
    }
    
    # 如果model_path是目录，则在该目录下保存结果
    if os.path.isdir(model_path):
        result_path = os.path.join(model_path, 'test_results.pth')
    else:
        # 如果model_path是文件，则在同目录下保存结果
        result_path = os.path.join(os.path.dirname(model_path), 'test_results.pth')
    

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
    
    print(f"加载数据: 总量 {len(dataset)}, 数据占比 {1 - ratio:.1%}")
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
    train_dataset = StreamingVoxelGraspDataset(
        [raw_data[i] for i in indices['train']],
        args.grasp_info_path
    )

    # 使用训练集的标准化器创建验证集和测试集
    val_dataset = StreamingVoxelGraspDataset(
        [raw_data[i] for i in indices['val']],
        args.grasp_info_path
    )
    
    test_dataset = StreamingVoxelGraspDataset(
        [raw_data[i] for i in indices['test']],
        args.grasp_info_path
    )
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(dataset, indices, args):
    """创建数据加载器，使用索引子集采样"""
    # 创建子集采样器，根据索引从同一个数据集中选择不同的样本
    class SubsetRandomSampler(torch.utils.data.Sampler):
        def __init__(self, indices, balanced=False, dataset=None):
            self.indices = indices
            self.balanced = balanced
            self.dataset = dataset

            if balanced and dataset is not None:
                # 构建标签映射
                self.pos_indices = []
                self.neg_indices = []

                for idx in self.indices:
                    label = dataset.labels[idx]
                    if label == 1:
                        self.pos_indices.append(idx)
                    else:
                        self.neg_indices.append(idx)

                print(f"子集中 - 正样本数: {len(self.pos_indices)}, 负样本数: {len(self.neg_indices)}")

        def __iter__(self):
            if self.balanced and self.dataset is not None:
                # 平衡采样
                num_samples = min(len(self.pos_indices), len(self.neg_indices)) * 2
                indices = []

                # 随机打乱正负样本
                pos_indices = self.pos_indices.copy()
                neg_indices = self.neg_indices.copy()
                random.shuffle(pos_indices)
                random.shuffle(neg_indices)

                # 依次添加正负样本
                for i in range(min(len(pos_indices), len(neg_indices))):
                    indices.append(pos_indices[i])
                    indices.append(neg_indices[i])

                return iter(indices)
            else:
                # 普通随机采样
                indices = self.indices.copy()
                random.shuffle(indices)
                return iter(indices)

        def __len__(self):
            if self.balanced and self.dataset is not None:
                return min(len(self.pos_indices), len(self.neg_indices)) * 2
            else:
                return len(self.indices)

    # 使用平衡采样器创建训练集数据加载器
    train_sampler = SubsetRandomSampler(indices['train'], balanced=True, dataset=dataset)

    # 配置训练集数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 8),
        pin_memory=True,
        persistent_workers=False if args.disable_multiprocessing else True,
        prefetch_factor=2 if not args.disable_multiprocessing else None
    )
    
    # 验证和测试加载器使用普通批处理
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 4),
        pin_memory=True,
        persistent_workers=False if args.disable_multiprocessing else True,
        prefetch_factor=2 if not args.disable_multiprocessing else None
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 4),
        pin_memory=True,
        persistent_workers=False if args.disable_multiprocessing else True,
        prefetch_factor=2 if not args.disable_multiprocessing else None
    )
    
    return train_loader, val_loader, test_loader


def optuna_objective(trial, args=None):
    """Optuna优化的目标函数：训练并评估模型，返回评估指标
    
    Args:
        trial: Optuna trial对象
        args: 命令行参数，如果为None则自动解析
        
    Returns:
        float: 需要最小化的目标值（通常是-f1分数）
    """
    # 如果没有提供参数，则解析命令行参数
    if args is None:
        args = parse_args()
    
    # 从trial中采样超参数
    args.lr = trial.suggest_float('lr', 1e-4, 1e-2)
    args.beta = trial.suggest_float('beta', 0.1, 10.0)
    args.latent_dim = trial.suggest_int('latent_dim', 4, 32)
    args.vae_dropout_rate = trial.suggest_float('vae_dropout_rate', 0.1, 0.5)
    args.ebm_dropout_rate = trial.suggest_float('ebm_dropout_rate', 0.1, 0.3)
    
    # 为当前trial创建唯一的运行名称和保存路径
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    trial_name = f"trial_{trial.number}_{timestamp}"
    trial_save_path = os.path.join(args.save_path, trial_name)
    os.makedirs(trial_save_path, exist_ok=True)
    args.save_path = trial_save_path
    
    # 设置wandb运行名称
    args.wandb_name = f"optuna_{trial_name}"
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 初始化wandb
    run_name = args.wandb_name or f"vae-ebm-{args.model_type}-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # 创建数据集
    print("创建数据集...")

    # 提供两种数据加载方案
    if args.data_ratio < 1.0:
        # 方案1: 使用load_raw_data加载部分数据，控制数据量
        print(f"使用load_raw_data加载部分数据，比例: {args.data_ratio:.2f}")
        raw_data = load_raw_data(args.data_path, ratio=args.data_ratio)
        total_size = len(raw_data)
        indices = split_data_indices(total_size, args.train_split, args.val_split)

        # 创建数据集，使用load_raw_data加载的数据
        print("使用预加载数据创建数据集...")
        train_dataset = StreamingVoxelGraspDataset(
            [raw_data[i] for i in indices['train']],
            args.grasp_info_path
        )

        val_dataset = StreamingVoxelGraspDataset(
            [raw_data[i] for i in indices['val']],
            args.grasp_info_path
        )

        test_dataset = StreamingVoxelGraspDataset(
            [raw_data[i] for i in indices['test']],
            args.grasp_info_path
        )

        print(f"数据集大小 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 8),
            pin_memory=True,
            persistent_workers=False if args.disable_multiprocessing else True,
            prefetch_factor=2 if not args.disable_multiprocessing else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 4),
            pin_memory=True,
            persistent_workers=False if args.disable_multiprocessing else True,
            prefetch_factor=2 if not args.disable_multiprocessing else None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0 if args.disable_multiprocessing else min(os.cpu_count(), 4),
            pin_memory=True,
            persistent_workers=False if args.disable_multiprocessing else True,
            prefetch_factor=2 if not args.disable_multiprocessing else None
        )
    else:
        # 方案2: 使用流式加载，处理全部数据
        print("使用流式加载处理全部数据")
        # 创建索引
        total_size = np.load(args.data_path, allow_pickle=True, mmap_mode='r')['voxel_data'].shape[0]
        indices = split_data_indices(total_size, args.train_split, args.val_split)

        dataset = StreamingVoxelGraspDataset(
            args.data_path,
            args.grasp_info_path
        )

        print(f"数据集大小: {len(dataset)}")

        # 创建数据加载器 - 传递单一数据集和索引
        train_loader, val_loader, test_loader = create_data_loaders(dataset, indices, args)

    # 创建模型
    if args.model_type == 'conv3d':
        vae = VoxelBetaVAE_Conv3D(in_channels=1, latent_dim=args.latent_dim, beta=args.beta,
                                 input_size=args.input_size).to(device)
    else:  # 'mlp'
        vae = VoxelBetaVAE_MLP(in_channels=1, latent_dim=args.latent_dim, beta=args.beta,
                               dropout=args.vae_dropout_rate,
                               input_size=args.input_size).to(device)

    energy_model = GraspEnergyNetwork(
        input_dim=args.latent_dim + 7,
        hidden_dims=args.hidden_dims,
        num_layers=len(args.hidden_dims),
        dropout_rate=args.ebm_dropout_rate
    ).to(device)

    # 创建优化器和学习率调度器
    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr)
    optimizer_energy = optim.Adam(energy_model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer_vae, 'min', factor=0.5, patience=5, verbose=True)

    # 创建损失函数
    criterion_vae = None
    criterion_energy = EnergyBasedLoss()

    # 训练模型，传递trial参数
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
        early_stop_patience=args.early_stop_patience,
        trial=trial  # 传递trial参数
    )

    # 每个epoch结束后向Optuna报告验证集性能指标
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
    val_metrics = checkpoint.get('val_metrics', {})

    f1_score = val_metrics.get('f1', 0.0)
    auc_score = val_metrics.get('auc', 0.0)
    energy_gap = val_metrics.get('energy_gap', 0.0)

    # 将numpy类型转换为Python原生类型
    if isinstance(f1_score, (np.float32, np.float64)):
        f1_score = float(f1_score)
    if isinstance(auc_score, (np.float32, np.float64)):
        auc_score = float(auc_score)
    if isinstance(energy_gap, (np.float32, np.float64)):
        energy_gap = float(energy_gap)

    # 记录额外指标到trial
    trial.set_user_attr('auc', auc_score)
    trial.set_user_attr('energy_gap', energy_gap)

    # 返回需要最小化的指标（取负号，因为我们想要最大化F1分数）
    return -f1_score


def run_optuna_optimization(args=None):
    """运行Optuna超参数优化
    
    Args:
        args: 命令行参数，如果为None则自动解析
    """
    if args is None:
        args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 添加Optuna相关参数
    args.use_wandb = True  # 控制是否使用wandb
    args.n_trials = 30 if not hasattr(args, 'n_trials') else args.n_trials  # Optuna trials数量
    args.study_name = f"vae_ebm_study_{time.strftime('%Y%m%d_%H%M%S')}" if not hasattr(args, 'study_name') else args.study_name

    # 创建保存目录
    optuna_save_dir = os.path.join(args.optuna_save_path, 'optuna_vae_ebm_studies')
    os.makedirs(optuna_save_dir, exist_ok=True)

    # 设置数据库存储
    storage_path = os.path.join(optuna_save_dir, f"{args.study_name}.db")
    storage_url = f"sqlite:///{storage_path}"
    
    # 创建Optuna研究
    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',  # 我们返回-f1_score，所以是最小化
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=storage_url,
        load_if_exists=True
    )
    
    print(f"开始Optuna超参数优化: {args.study_name}")
    print(f"计划进行 {args.n_trials} 次试验")
    print(f"结果将保存在: {optuna_save_dir}")
    
    # 运行优化
    study.optimize(lambda trial: optuna_objective(trial, args), n_trials=args.n_trials)
    
    # 显示结果
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    
    print("研究统计:")
    print(f"  完成的试验数量: {len(study.trials)}")
    print(f"  被剪枝的试验数量: {len(pruned_trials)}")
    print(f"  成功完成的试验数量: {len(complete_trials)}")
    
    if study.best_trial:
        print("\n最佳试验:")
        trial = study.best_trial
        print(f"  试验编号: {trial.number}")
        print(f"  F1分数: {-trial.value:.4f}")
        print(f"  AUC: {trial.user_attrs.get('auc', 'N/A')}")
        print(f"  能量差: {trial.user_attrs.get('energy_gap', 'N/A')}")
        print("\n最佳参数:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
    
    # 保存结果可视化
    visualizations_dir = os.path.join(optuna_save_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    try:
        # 优化历史
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(visualizations_dir, 'optimization_history.png'))
        
        # 参数重要性
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(visualizations_dir, 'param_importances.png'))
        
        # 参数关系
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(visualizations_dir, 'parallel_coordinate.png'))
        
        print(f"可视化结果已保存到: {visualizations_dir}")
    except Exception as e:
        print(f"保存可视化时出错: {e}")
    
    # 保存最佳配置
    best_config = {
        'best_params': study.best_params,
        'best_value': -study.best_value if study.best_value else None,
        'best_trial': study.best_trial.number if study.best_trial else None
    }
    
    with open(os.path.join(optuna_save_dir, 'best_config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)
    
    return study


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
    parser.add_argument('--model_type', type=str, default='mlp', choices=['conv3d', 'mlp'],
                        help='VAE模型类型: conv3d或mlp')
    parser.add_argument('--latent_dim', type=int, default=32, help='潜在空间维度')
    parser.add_argument('--beta', type=float, default=1.0, help='beta-VAE的beta参数')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512], help='能量模型的隐藏层维度')
    parser.add_argument('--ebm_dropout_rate', type=float, default=0.1,
                        help='dropout率')
    parser.add_argument('--vae_dropout_rate', type=float, default=0.3,
                        help='dropout率')
    parser.add_argument('--input_size', type=int, nargs='+', default=[25, 25, 30], help='输入体素的结构')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--lambda_energy', type=float, default=1e3, help='能量损失权重')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--data_ratio', type=float, default=0.1, help='数据比例')
    parser.add_argument('--seed', type=int, default=23, help='随机种子')

    
    # 保存和日志
    parser.add_argument('--save_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\vae_ebm_model', help='模型保存路径')
    parser.add_argument('--optuna_save_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\script\obstacle_network\optuna_output', help='Optuna保存路径')
    parser.add_argument('--wandb_project', type=str, default='voxel-grasp-energy', help='Wandb项目名称')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb运行名称')
    
    # 运行模式
    parser.add_argument('--train', type=bool, default=True, help='训练模型')
    parser.add_argument('--eval', type=bool, default=True, help='评估模型')

    # 添加新参数控制数据预取和处理
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载的worker数量')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='数据预取因子')
    parser.add_argument('--disable_multiprocessing', action='store_true', 
                        help='禁用多进程数据加载，解决序列化问题')

    # 添加Optuna相关参数
    parser.add_argument('--optuna', type=bool, default=True, help='使用Optuna进行超参数搜索')
    parser.add_argument('--n_trials', type=int, default=30, help='Optuna试验次数')
    parser.add_argument('--study_name', type=str, default="VAE_EBM_study", help='Optuna研究名称')
    
    args = parser.parse_args()
    
    # 确保input_size是列表
    if args.input_size is None:
        args.input_size = [25, 25, 30]
    
    # 打印参数以便调试
    print(f"解析的input_size参数: {args.input_size}, 类型: {type(args.input_size)}")
    
    return args


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    args.input_size = [25, 25, 30]
    
    # 检查是否使用Optuna
    if args.optuna:
        print("启动Optuna超参数搜索模式")
        study = run_optuna_optimization(args)
        return
    
    # 原有的训练流程
    # ... existing code ...

if __name__ == "__main__":
    main()
