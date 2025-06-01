# 混合能量模型实现（Hybrid Energy Model）

这个项目实现了一个混合能量模型，用于同时处理离散和连续变量，可以应用在机器人抓取姿态规划与重规划任务中。

## 项目结构

```
regrasp_test/
├── __init__.py           # 包初始化文件
├── config/               # 配置相关模块
│   ├── __init__.py
│   └── configuration.py  # 配置管理
├── data/                 # 数据相关模块
│   ├── __init__.py
│   └── data_generation.py  # 数据生成函数
├── models/               # 模型定义模块
│   ├── __init__.py
│   └── energy_model.py   # 混合能量模型定义
├── utils/                # 工具函数模块
│   ├── __init__.py
│   ├── losses.py         # 损失函数
│   ├── sampling.py       # 采样方法
│   └── visualization.py  # 可视化工具
├── main.py               # 命令行主入口
├── run.py                # 简单运行脚本
├── train.py              # 模型训练
├── sample.py             # 模型采样
└── README.md             # 项目说明
```

## 安装依赖

本项目需要以下依赖：

```bash
pip install torch matplotlib numpy wandb tqdm pillow
```

## 使用方法

### 快速开始（使用演示脚本）

```bash
python -m regrasp_test.run
```

这会使用默认配置运行训练和采样，并在wandb上记录结果。

### 使用命令行接口

训练模型：

```bash
python -m regrasp_test.main --mode train --dist_type multi_gaussian --epochs 500
```

采样：

```bash
python -m regrasp_test.main --mode sample --model_path models/hybrid_energy_model.pth
```

训练并采样：

```bash
python -m regrasp_test.main --mode train_and_sample --dist_type spiral --epochs 300
```

### 命令行参数

- `--mode`: 运行模式，可选 'train', 'sample', 'train_and_sample'
- `--use_wandb`: 是否使用wandb记录实验，默认True
- `--model_path`: 模型保存/加载路径
- `--dist_type`: 数据分布类型，可选 'multi_gaussian', 'circle', 'spiral', 'checkerboard'
- `--num_discrete`: 离散变量维度
- `--epochs`: 训练轮数
- `--batch_size`: 批量大小
- `--lr`: 学习率
- `--n_samples`: 每个类别的采样点数
- `--steps`: Langevin采样步数

## 特性

1. **多种复杂数据分布支持**：
   - 多高斯混合分布
   - 环形分布
   - 螺旋分布
   - 棋盘格分布

2. **灵活的能量模型**：
   - 支持同时建模离散和连续变量
   - 使用对比学习进行训练

3. **高级采样技术**：
   - 兰格万动力学采样
   - 支持模拟退火策略
   - 可视化采样轨迹

4. **实验跟踪与可视化**：
   - 使用Weights & Biases记录实验
   - 能量场可视化
   - 数据分布可视化
   - 训练曲线跟踪

## 扩展

要扩展这个项目，可以：

1. 添加新的数据分布类型
2. 实现更复杂的能量模型架构
3. 开发新的采样算法
4. 将模型应用于实际的机器人抓取任务

## 贡献

欢迎提交Issue和Pull Request！ 