import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 修改 shortcut 的逻辑，确保维度变换一致
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.fc1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += shortcut  # 残差连接
        return self.relu(out)  # 非线性激活

class GraspingNetwork(nn.Module):
    def __init__(self, input_dim=7, output_dim=109):
        super().__init__()
        self.network = nn.Sequential(
            # 第一层：特征提取和升维
            ResidualBlock(input_dim, 256),
            # 第二层：特征增强
            ResidualBlock(256, 512),
            # 第三层：高级特征提取
            ResidualBlock(512, 1024),
            # 第四层：继续特征提取
            ResidualBlock(1024, 1024),
            # 第五层：特征处理
            ResidualBlock(1024, 512),
            # 第六层：特征压缩
            ResidualBlock(512, 256),
            # 第七层：特征整合
            ResidualBlock(256, 128),
            # 第八层：输出层
            nn.Linear(128, output_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    # 简单测试代码
    import torch
    model = GraspingNetwork(input_dim=7, output_dim=109)
    test_input = torch.randn(8, 7)  # Batch size = 8, input_dim = 7
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
