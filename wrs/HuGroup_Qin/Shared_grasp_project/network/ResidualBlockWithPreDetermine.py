"""

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241210 Osaka Univ.

"""
import torch
import torch.nn as nn

class ResidualBlockWithPreDetermine(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(ResidualBlockWithPreDetermine, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.fc1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += shortcut
        return self.relu(out)

class GraspingNetwork(nn.Module):
    def __init__(self, input_dim=17, output_dim=109):
        super(GraspingNetwork, self).__init__()
        self.net_encode = nn.Sequential(
            ResidualBlockWithPreDetermine(input_dim, 128, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(128, 256, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(256, 512, dropout_rate=0.2)
        )

        self.net_grasppre1 = nn.Sequential(
            ResidualBlockWithPreDetermine(512, 512, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(512, 512, dropout_rate=0.2)
        )
        
        self.net_grasppre2 = nn.Sequential(
            ResidualBlockWithPreDetermine(512, 512, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(512, 512, dropout_rate=0.2)
        )

        self.net_feasiblepre = nn.Sequential(
            ResidualBlockWithPreDetermine(512, 256, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(256, 128, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(128, 32, dropout_rate=0.2)
        )
        
        self.net_grasppre3 = nn.Sequential(
            ResidualBlockWithPreDetermine(544, 256, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(256, 128, dropout_rate=0.2),
            ResidualBlockWithPreDetermine(128, 128, dropout_rate=0.2)
        )
        
        self.Linear = nn.Linear(128, output_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encode = self.net_encode(x)
        grasppre1 = self.net_grasppre1(encode)
        grasppre2 = self.net_grasppre2(grasppre1)
        feasiblepre = self.net_feasiblepre(encode)
        grasppre3 = self.net_grasppre3(torch.cat([grasppre2, feasiblepre], dim=1))
        pre_id = self.Linear(grasppre3)
        return pre_id, feasiblepre

if __name__ == '__main__':

    import torch
    model = GraspingNetwork(input_dim=17, output_dim=40)
    test_input = torch.randn(8, 17)  # Batch size = 8, input_dim = 17
    test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
