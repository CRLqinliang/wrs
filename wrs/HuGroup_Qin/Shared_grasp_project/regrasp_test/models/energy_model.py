import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridEnergyModel(nn.Module):
    def __init__(self, num_discrete=4, dim_cont=2, hidden_dims=[64, 64, 32]):
        super().__init__()
        
        self.num_discrete = num_discrete
        self.dim_cont = dim_cont
        
        # 增强离散变量处理
        self.discrete_net = nn.Sequential(
            nn.Linear(num_discrete, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        
        # 连续变量处理
        self.continuous_net = nn.Sequential(
            nn.Linear(dim_cont, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # 增强融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        # 添加离散变量映射层
        self.discrete_proj = nn.Linear(num_discrete, hidden_dims[1])
        
    def forward(self, x):
        # 分离离散和连续部分
        x_d = x[:, :self.num_discrete]
        x_c = x[:, self.num_discrete:]
        
        # 分别处理
        h_d = self.discrete_net(x_d)
        h_c = self.continuous_net(x_c)
        
        # 使用映射层和残差连接增强离散变量影响
        x_d_proj = self.discrete_proj(x_d)
        h = torch.cat([h_d + x_d_proj, h_c], dim=1)
        
        # 融合并输出能量
        energy = self.fusion_net(h)
        return energy.squeeze()