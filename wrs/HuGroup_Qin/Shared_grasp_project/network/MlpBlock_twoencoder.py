import torch  # 在文件开头缺少这个必要的导入
import torch.nn as nn

class GraspingNetwork(nn.Module):
    def __init__(self, input_dim=8, output_dim=109, dropout_rate=0.3):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even")
        self.input_dim = input_dim

        # 第一个编码器
        dims_encoder1 = [self.input_dim//2, 128, 256, 512]
        encoder1 = []
        for i in range(len(dims_encoder1)-1):
            encoder1.extend([
                nn.Linear(dims_encoder1[i], dims_encoder1[i+1]),
                nn.BatchNorm1d(dims_encoder1[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        self.encoder_network1 = nn.Sequential(*encoder1)

        # 第二个编码器
        dims_encoder2 = [self.input_dim//2, 128, 256, 512]
        encoder2 = []
        for i in range(len(dims_encoder2)-1):
            encoder2.extend([
                nn.Linear(dims_encoder2[i], dims_encoder2[i+1]),
                nn.BatchNorm1d(dims_encoder2[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        self.encoder_network2 = nn.Sequential(*encoder2)

        dims = [1024, 512, 256, output_dim]
        layers = []
        
        for i in range(len(dims)-2):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        
        # 添加最后一层（输出层）
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder1 = self.encoder_network1(x[:, :self.input_dim//2])
        encoder2 = self.encoder_network2(x[:, self.input_dim//2:])
        x = torch.cat([encoder1, encoder2], dim=1)
        return self.network(x)


if __name__ == '__main__':
    # 简单测试代码
    import torch
    model = GraspingNetwork(input_dim=7, output_dim=109)
    test_input = torch.randn(8, 7)  # Batch size = 8, input_dim = 7
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
