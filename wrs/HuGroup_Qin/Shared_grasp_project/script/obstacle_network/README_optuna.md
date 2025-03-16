# VAE-EBM 模型超参数优化指南

本指南介绍如何使用 Optuna 框架对 VAE-EBM 障碍物网络模型的超参数进行优化。

## 简介

我们创建了一个基于 Optuna 的超参数优化框架，用于自动搜索以下超参数的最优组合：

- 学习率 (lr)
- Beta-VAE 系数 (beta)
- 潜在空间维度 (latent_dim)
- VAE Dropout 率 (vae_dropout_rate)

这些参数对模型性能有显著影响，通过优化它们可以获得更好的障碍物识别效果。

## 依赖安装

在运行之前，请确保安装了 Optuna 库：

```bash
pip install optuna
```

## 使用方法

### 基本用法

最简单的方式是使用我们提供的包装脚本 `run_optuna_search.py`：

```bash
python run_optuna_search.py --data_path <你的数据路径> --save_path <结果保存路径>
```

### 必需参数

- `--data_path`：训练数据的路径
- `--save_path`：结果保存路径，Optuna 的数据库文件和结果 JSON 将保存在这里

### 重要可选参数

- `--n_trials`：Optuna 试验次数，默认为 20，增加此值可能找到更好的参数组合，但会增加运行时间
- `--train_best`：添加此标志将在参数搜索完成后，使用最佳参数组合重新训练一个完整模型
- `--model_type`：模型类型，可选 'conv3d' 或 'mlp'，默认为 'mlp'
- `--num_epochs`：每次试验的训练轮数，默认为 30。建议设置较小的值以加快超参数搜索
- `--data_ratio`：使用的数据比例，默认为 0.3。降低此值可加快搜索速度

### 完整示例

```bash
python run_optuna_search.py \
    --data_path /path/to/your/data \
    --grasp_info_path /path/to/grasp/info \
    --save_path ./optuna_results \
    --n_trials 50 \
    --model_type mlp \
    --num_epochs 20 \
    --batch_size 32 \
    --early_stop_patience 5 \
    --data_ratio 0.3 \
    --train_best
```

## 参数搜索范围

当前设置的参数搜索范围为：

- 学习率 (lr)：1e-5 到 1e-3，对数均匀分布
- Beta-VAE 系数 (beta)：0.1 到 5.0，均匀分布
- 潜在空间维度 (latent_dim)：16 到 128，离散均匀分布
- VAE Dropout 率 (vae_dropout_rate)：0.1 到 0.5，均匀分布

如需修改搜索范围，请编辑 `run_obstacle_network_optuna.py` 文件中的 `objective` 函数。

## 结果解释

优化完成后，结果将保存在指定的 `save_path` 目录中：

1. SQLite 数据库文件 (`.db`)：包含所有试验的详细信息
2. JSON 结果文件：包含最佳参数和所有试验的摘要信息

如果使用了 `--train_best` 选项，最佳模型将保存在 `save_path/best_model_TIMESTAMP` 目录中。

## 自定义优化

如果需要优化其他参数或修改优化目标，可以直接编辑 `run_obstacle_network_optuna.py` 文件：

- 在 `objective` 函数中添加或修改参数搜索范围
- 修改优化目标（当前使用验证集 F1 分数作为优化目标）

## 注意事项

1. 超参数优化是计算密集型任务，建议在有 GPU 的环境下运行
2. 对于初始搜索，可以使用较小的 `data_ratio` 和 `num_epochs` 来加快速度
3. 找到初步最优参数后，可以缩小搜索范围并增加 `n_trials` 进行精细搜索
4. Optuna 支持并行试验，但当前实现是串行的。如需并行，请参考 Optuna 文档进行修改 