""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241205 Osaka Univ.
"""

from typing import List, Tuple, Optional, Union
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
sys.path.append("H:/Qin/wrs")

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import argparse, tqdm, wandb
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
from wrs.HuGroup_Qin.data.network.MlpBlock import GraspingNetwork
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, average_precision_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# 常量定义
OBJECT_MESH_PATH = Path(r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
# base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
# mgm.gen_frame().attach_to(base)


def env_setup():
    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.unshow_cdprim()
    return robot

def collect_model_predictions(model_1, model_2, test_loader, device, shared_testdataset=False):
    """收集两个模型的预测结果"""
    model_1.eval()
    model_2.eval()
    all_labels, all_logits_1, all_logits_2 = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc="收集预测结果"):
            inputs, labels = inputs.to(device), labels.to(device)
            if not shared_testdataset:
                outputs_1 = torch.sigmoid(model_1(inputs))
                outputs_2 = torch.sigmoid(model_2(inputs))
            else:
                outputs_1 = torch.sigmoid(model_1(inputs[:, :6]))
                outputs_2 = torch.sigmoid(model_2(inputs[:, 6:12]))
            all_labels.append(labels.cpu().numpy())
            all_logits_1.append(outputs_1.cpu().numpy())
            all_logits_2.append(outputs_2.cpu().numpy())

    return (np.vstack(all_labels), 
            np.vstack(all_logits_1), 
            np.vstack(all_logits_2))

def evaluate_threshold_combination(all_labels, all_logits_1, all_logits_2, th1, th2):
    # 对两个模型的预测结果进行AND操作
    predictions_1 = (all_logits_1 > th1).astype(float)
    predictions_2 = (all_logits_2 > th2).astype(float)
    predictions = np.logical_and(predictions_1, predictions_2).astype(float)

    # 过滤有效类别, 只考虑有正样本的数据
    valid_classes = np.sum(all_labels, axis=0) > 0
    filtered_labels = all_labels[:, valid_classes]
    filtered_predictions = predictions[:, valid_classes]

    # 计算评估指标
    return {
        "threshold_1": th1,
        "threshold_2": th2,
        "accuracy": accuracy_score(all_labels, predictions) * 100,
        "hamming_loss": hamming_loss(all_labels, predictions),
        "precision_macro": precision_score(filtered_labels, filtered_predictions,
                                        average='macro', zero_division=0),
        "recall_macro": recall_score(filtered_labels, filtered_predictions,
                                    average='macro', zero_division=0),
        "f1_macro": f1_score(filtered_labels, filtered_predictions,
                            average='macro', zero_division=0)
    }

def plot_heatmap(matrix, x_labels, y_labels, title, filename):
    """绘制并保存热力图"""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels, 
                     annot=False, fmt=".1f",
                     cmap="viridis",
                     annot_kws={"color": "black", "fontsize": 10})
    plt.title(title, fontsize=14)
    plt.xlabel("Model robot-table Threshold", fontsize=14)
    plt.ylabel("Model robot-table Threshold", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 手动设置颜色条的字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    # 调整子图的布局，确保所有元素都能显示
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def log_results_to_wandb(results_data, precision_matrix, recall_matrix, 
                        thresholds_1, thresholds_2, best_thresholds, best_f1):
    """记录结果到wandb并在本地绘制热力图"""
    try:
        # 创建热力图数据表
        wandb.log({
            "Dataset Threshold Evaluation": wandb.Table(
                data=results_data,
                columns=["Threshold_1", "Threshold_2", "Accuracy", "Hamming_Loss", 
                        "Precision_Macro", "Recall_Macro", "F1_Macro"]
            ),
            "Dataset Best Thresholds": best_thresholds,
            "Dataset Best F1 Score": best_f1
        })

        # 在本地绘制热力图
        plot_heatmap(precision_matrix, [f"{x:.2f}" for x in thresholds_1], [f"{x:.2f}" for x in thresholds_2],
                     "Precision Heatmap", "precision_heatmap.png")
        plot_heatmap(recall_matrix, [f"{x:.2f}" for x in thresholds_1], [f"{x:.2f}" for x in thresholds_2],
                     "Recall Heatmap", "recall_heatmap.png")

    except Exception as e:
        print(f"Warning: Failed to log to wandb: {str(e)}")

def test_AND_model_with_dataset(args, model_1, model_2, test_loader, device,
                            thresholds_1=np.arange(0.3, 0.9, 0.05),
                            thresholds_2=np.arange(0.3, 0.9, 0.05),
                            verbose=True):
    # 收集模型预测
    all_labels, all_logits_1, all_logits_2 = collect_model_predictions(
        model_1, model_2, test_loader, device, args.shared_testdataset
    )

    # 初始化结果存储
    threshold_results = {}
    best_f1 = -1
    best_thresholds = None
    best_results = None
    results_data = []
    
    # 创建性能矩阵
    precision_matrix = np.zeros((len(thresholds_1), len(thresholds_2)))
    recall_matrix = np.zeros((len(thresholds_1), len(thresholds_2)))

    # 测试不同阈值组合
    print("\n测试不同阈值组合的性能:") if verbose else None
    for i, th1 in enumerate(tqdm.tqdm(thresholds_1, desc="测试模型1阈值")):
        for j, th2 in enumerate(thresholds_2):
            # 评估当前阈值组合
            results = evaluate_threshold_combination(
                all_labels, all_logits_1, all_logits_2, th1, th2
            )
            
            # 更新结果
            results_data.append([th1, th2] + [results[k] for k in results.keys() 
                                            if k not in ['threshold_1', 'threshold_2']])
            threshold_results[(th1, th2)] = results
            precision_matrix[i, j] = results["precision_macro"]
            recall_matrix[i, j] = results["recall_macro"]

            # 更新最佳结果
            if results["f1_macro"] > best_f1:
                best_f1 = results["f1_macro"]
                best_thresholds = (th1, th2)
                best_results = results

            # 输出当前结果
            if verbose:
                print(f"\nThreshold_1 = {th1:.2f}, Threshold_2 = {th2:.2f}:")
                print(f"Accuracy: {results['accuracy']:.2f}%")
                print(f"Macro - P: {results['precision_macro']:.4f}, "
                      f"R: {results['recall_macro']:.4f}, "
                      f"F1: {results['f1_macro']:.4f}")

    # 记录结果到wandb
    log_results_to_wandb(results_data, precision_matrix, recall_matrix,
                        thresholds_1, thresholds_2, best_thresholds, best_f1)

    return best_results, threshold_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    parser.add_argument('--model_path_1', type=str, default='model/feasible_grasp_robot_table_bottle_57.pth',
                        help='模型保存路径')
    parser.add_argument('--model_path_2', type=str, default='model/feasible_grasp_robot_table_bottle_57.pth',
                        help='模型保存路径')
    parser.add_argument('--test_dataset', type=str, default="H:\Qin\wrs\wrs\HuGroup_Qin\data\grasps\Bottle\common_grasp_random_position_bottle_57.pickle",
                        help='测试数据集路径')
    parser.add_argument('--shared_testdataset', type=bool, default=True, help='是否使用共享测试数据集')
    parser.add_argument('--input_dim', type=int, default=6, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=57, help='输出维度')
    parser.add_argument('--batch_size', type=int, default=256, help='批量大小')

    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='feasible_grasp', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str,
                        default='shared_grasp_AND_evaluation_bottle_57',
                        help='wandb运行名称')

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    if args.shared_testdataset:
        print("使用共享测试数据集...")
        from wrs.HuGroup_Qin.data.common_grasp_network_train import GraspingDataset
    else:
        from wrs.HuGroup_Qin.data.grasp_network_train import GraspingDataset

    # 已有数据集测试
    model_1 = GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)
    model_2 = GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)

    model_1.load_state_dict(torch.load(args.model_path_1))
    model_2.load_state_dict(torch.load(args.model_path_2))

    # 加载测试数据集
    test_dataset = GraspingDataset(args.output_dim, args.test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # 可以为两个模型设置不同的阈值范围
    thresholds_1 = np.arange(0.3, 0.9, 0.05)
    thresholds_2 = np.arange(0.3, 0.9, 0.05)

    # wandb
    wandb.init(project=args.wandb_project, name=args.wandb_name)

    # 使用数据集测试模型
    print("\n使用数据集进行测试...")
    dataset_results = test_AND_model_with_dataset(
        args, model_1, model_2, test_loader,
        device='cpu',
        thresholds_1=thresholds_1,
        thresholds_2=thresholds_2
    )
