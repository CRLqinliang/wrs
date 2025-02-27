""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241209 Osaka Univ.

"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        out = torch.matmul(attention, value)
        return out + x  # 残差连接


class ResidualBlockWithSelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(ResidualBlockWithSelfAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.attention = SelfAttention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 投影映射：当输入和输出维度不同时对齐
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = self.attention(out)
        out += shortcut  # 残差连接
        return self.relu(out)


class GraspingNetwork(nn.Module):
    def __init__(self, input_dim=7, output_dim=936):
        super(GraspingNetwork, self).__init__()
        self.network = nn.Sequential(
            ResidualBlockWithSelfAttention(input_dim, 256),
            ResidualBlockWithSelfAttention(256, 512, dropout_rate=0.3),
            ResidualBlockWithSelfAttention(512, 1024, dropout_rate=0.3),
            ResidualBlockWithSelfAttention(1024, 1024, dropout_rate=0.3),
            ResidualBlockWithSelfAttention(1024, 512, dropout_rate=0.3),
            ResidualBlockWithSelfAttention(512, 256, dropout_rate=0.3),
            ResidualBlockWithSelfAttention(256, 128, dropout_rate=0.3),
            nn.Linear(128, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    pass
