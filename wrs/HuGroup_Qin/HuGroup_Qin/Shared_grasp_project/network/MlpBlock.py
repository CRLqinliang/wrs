import torch.nn as nn

class GraspingNetwork(nn.Module):
    def __init__(self, input_dim=7, output_dim=109):
        super().__init__()
        dims = [input_dim, 256, 512, 1024, 1024, 512, 256, 128, output_dim]
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
        return self.network(x)


if __name__ == '__main__':

    # 简单测试代码
    import torch
    model = GraspingNetwork(input_dim=7, output_dim=109)
    test_input = torch.randn(8, 7)  # Batch size = 8, input_dim = 7
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
