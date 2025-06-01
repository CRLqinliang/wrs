import os
import numpy as np
import pickle
import sys
import random  # 添加Python的random模块
sys.path.append("E:/Qin/wrs")
import wrs.basis.robot_math as rm
import wrs.basis.constant as ct
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.grasping.reasoner import GraspReasoner
from wrs.manipulation.placement.flatsurface import FSReferencePoses
from tqdm import tqdm
from scipy.stats import qmc  # 添加到文件开头的import部分
import argparse
import torch  
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import precision_score, recall_score, f1_score

base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)


BASE_PATH = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle"
GRASP_DATA_PREFIX = "bottle_grasp"
SAVE_PREFIX = "SharedGraspNetwork_bottle_experiment_data"

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
        layers.append(nn.SELU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.SELU())
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


def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def robot_env_setup():
    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([-90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    # robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    return robot


def obj_setup(name, pos, rotmat, rgb=None, alpha=None):
    # we only consider SE(2) DOF for the object
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha,
                initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_cmodel.pos = pos
    obj_cmodel.rotmat = rotmat
    # obj_cmodel.show_local_frame()
    # obj_cmodel.attach_to(base)
    return obj_cmodel


def get_file_paths(grasp_id):
    """根据grasp_id生成对应的文件路径"""
    grasp_data_path = os.path.join(BASE_PATH, f"{GRASP_DATA_PREFIX}_{grasp_id}.pickle")
    save_path = os.path.join(BASE_PATH, f"{SAVE_PREFIX}_{grasp_id}.pickle")
    return grasp_data_path, save_path

# Function to find common IDs and remove duplicates
def find_common_id(ref_graspid, given_graspid):
    if not given_graspid or not ref_graspid:
        return None
    common_id = set(ref_graspid) & set(given_graspid)
    return list(common_id) if common_id else None


def parse_args():
    parser = argparse.ArgumentParser(description='共享抓取数据收集')
    parser.add_argument('--grasp_ids', type=list, default=[922],
                        help='要处理的抓取ID列表')
    parser.add_argument('--total_iterations', type=int, default=int(1000),
                        help='总迭代次数')
    parser.add_argument('--pre_num', type=int, default=int(100),
                        help='预采样次数')
    parser.add_argument('--save_batch_size', type=int, default=1000,
                        help='每批保存的数据大小')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    
    parser.add_argument('--input_dim', type=int, default=21,
                        help='输入维度')
    parser.add_argument('--hidden_dims', type=list, default=[512, 512, 512],
                        help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='层数')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout率')
    parser.add_argument('--model_init_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_57_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl1_grobot_table_stinit.pth',
                        help='初始状态模型路径')
    parser.add_argument('--model_goal_path', type=str, default=r'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\feasible_best_model\best_model_grasp_ebm_SharedGraspNetwork_bottle_experiment_data_57_h3_b2048_lr0.001_t0.5_r75000_s0.7_q1_sl1_grobot_table_stinit.pth',
                        help='目标状态模型路径')

    parser.add_argument('--f', help=argparse.SUPPRESS) 
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 设置所有随机种子以确保可重复性
    SEED = args.seed
    random.seed(SEED)  # Python的random模块
    np.random.seed(SEED)  # NumPy

    # 如果使用PyTorch，也设置其随机种子
    if torch.cuda.is_available():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 更新全局配置
    GRASP_IDS = args.grasp_ids
    TOTAL_ITERATIONS = args.total_iterations
    SAVE_BATCH_SIZE = args.save_batch_size
    
    # 使用LHS采样器
    init_sampler = qmc.LatinHypercube(d=3, seed=SEED)  # 添加种子
    init_samples = init_sampler.random(n=TOTAL_ITERATIONS)

    goal_sampler = qmc.LatinHypercube(d=3, seed=SEED+1)  # 使用不同的种子
    goal_samples = goal_sampler.random(n=TOTAL_ITERATIONS)

    ranges = np.array([
        [-0.45, .45],         # pos_x
        [0.1, .6],            # pos_y
        [0, 2 * np.pi],       # theta
    ])

    init_scaled_samples = qmc.scale(init_samples, ranges[:, 0], ranges[:, 1])
    goal_scaled_samples = qmc.scale(goal_samples, ranges[:, 0], ranges[:, 1])
    
    init_pos_x = np.round(init_scaled_samples[:, 0], decimals=3)
    init_pos_y = np.round(init_scaled_samples[:, 1], decimals=3)
    init_theta = np.round(init_scaled_samples[:, 2], decimals=2)

    goal_pos_x = np.round(goal_scaled_samples[:, 0], decimals=3)
    goal_pos_y = np.round(goal_scaled_samples[:, 1], decimals=3)
    goal_theta = np.round(goal_scaled_samples[:, 2], decimals=2)

    # world configuration
    robot = robot_env_setup()
    gripper = wrs_gripper_v3.WRSGripper3()
    collect_data = []

    # object configuration
    obj_cmodel = mcm.CollisionModel(initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_init_pos = np.array([-0.424  ,0.156  ,0.   ])
    obj_init_rotmat = rm.rotmat_from_euler(0, 0, 0)
    obj_goal_pos = obj_init_pos.copy()
    obj_goal_rotmat = obj_init_rotmat.copy()
    obj_cmodel_copy = obj_cmodel.copy()

    # 生成所有可能的组合
    fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
    all_combinations = np.array(np.meshgrid(
        np.arange(len(fs_reference_poses)),  # init姿态的所有可能（0-4）
        np.arange(len(fs_reference_poses))   # goal姿态的所有可能（0-4）
    )).T.reshape(-1, 2)  # 形状为 (25, 2)
    repeats = TOTAL_ITERATIONS // len(all_combinations) + 1
    repeated_combinations = np.tile(all_combinations, (repeats, 1))
    np.random.shuffle(repeated_combinations)
    init_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 0]
    goal_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 1]

    # onehot编码
    obj_encoder = OneHotEncoder(sparse_output=False)
    obj_encoder.fit(np.array([0, 1, 2, 3, 4]).reshape(-1, 1))
    

    # 标准模式：使用两个模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    thresholds_dict = {}
    models['init'] = GraspEnergyNetwork(input_dim=args.input_dim, hidden_dims=args.hidden_dims,
                                        num_layers=args.num_layers, dropout_rate=args.dropout_rate)
    models['goal'] = GraspEnergyNetwork(input_dim=args.input_dim, hidden_dims=args.hidden_dims,
                                        num_layers=args.num_layers, dropout_rate=args.dropout_rate)
    checkpoint_init = torch.load(args.model_init_path, map_location=device)
    models['init'].load_state_dict(checkpoint_init['model_state_dict'])
    models['init'] = models['init'].to(device).float()
    checkpoint_goal = torch.load(args.model_goal_path, map_location=device)
    models['goal'].load_state_dict(checkpoint_goal['model_state_dict'])
    models['goal'] = models['goal'].to(device).float()
    thresholds_dict['init'] = checkpoint_init.get('optimal_threshold', -5.0)
    thresholds_dict['goal'] = checkpoint_goal.get('optimal_threshold', -5.0)
    print(f"\n使用标准模式（两个模型）进行评估")
    print(f"初始状态模型: {os.path.basename(args.model_init_path)}, 阈值: {thresholds_dict['init']:.3f}")
    print(f"目标状态模型: {os.path.basename(args.model_goal_path)}, 阈值: {thresholds_dict['goal']:.3f}")

    # 初始化存储执行时间的字典，每个grasp_id对应一个列表
    nn_execution_times = {grasp_id: [] for grasp_id in GRASP_IDS}
    analysis_execution_times = {grasp_id: [] for grasp_id in GRASP_IDS}
    
    # 初始化存储预测结果和真实标签的字典
    all_predictions = {grasp_id: [] for grasp_id in GRASP_IDS}
    all_pred_sum = {grasp_id: [] for grasp_id in GRASP_IDS}
    all_ground_truths = {grasp_id: [] for grasp_id in GRASP_IDS}
    all_energies = {grasp_id: [] for grasp_id in GRASP_IDS}  # 存储能量值

    for grasp_id in GRASP_IDS:
        
        grasp_data_path, common_id_save_path = get_file_paths(str(grasp_id))
        object_feasible_grasps = grasp_load(grasp_data_path)

        # 首先收集所有ee_values用于计算最大最小值
        ee_values = np.array([grasp.ee_values for grasp in object_feasible_grasps])
        ee_min = np.min(ee_values)
        ee_max = np.max(ee_values)

        RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)
        with tqdm(total=TOTAL_ITERATIONS, desc=f"Processing grasp_id {grasp_id}") as pbar:
            for sample_idx in range(TOTAL_ITERATIONS):
                # 保存原始采样位置
                original_init_pos = np.array([init_pos_x[sample_idx], init_pos_y[sample_idx], 0])
                original_goal_pos = np.array([goal_pos_x[sample_idx], goal_pos_y[sample_idx], 0])
                
                # 设置初始位置和目标位置
                obj_init_pos = fs_reference_poses._poses[init_stable_random_indices[sample_idx]][0].copy()
                obj_init_rotmat = fs_reference_poses._poses[init_stable_random_indices[sample_idx]][1].copy()
                obj_init_pos[0:2] = original_init_pos[0:2]  # 只覆盖x和y坐标
                obj_init_rotmat = rm.rotmat_from_euler(0, 0, init_theta[sample_idx]) @ obj_init_rotmat

                obj_goal_pos = fs_reference_poses._poses[goal_stable_random_indices[sample_idx]][0].copy()
                obj_goal_rotmat = fs_reference_poses._poses[goal_stable_random_indices[sample_idx]][1].copy()
                obj_goal_pos[0:2] = original_goal_pos[0:2]  # 只覆盖x和y坐标
                obj_goal_rotmat = rm.rotmat_from_euler(0, 0, goal_theta[sample_idx]) @ obj_goal_rotmat

                # 保存用于计算可行抓取的位置深拷贝
                init_pos_for_feasible = obj_init_pos
                init_rotmat_for_feasible = obj_init_rotmat
                goal_pos_for_feasible = obj_goal_pos
                goal_rotmat_for_feasible = obj_goal_rotmat

                # 在init_pos_for_feasible, init_rotmat_for_feasible处计算可行抓取
                # 计算可行抓取 - robot_table_obj
                init_obj = obj_setup(name="init_obj", pos=init_pos_for_feasible, rotmat=init_rotmat_for_feasible, alpha=0.3)

                # 获取真实可行抓取
                start_time = time.time()
                analysis_init_available_gids_robot_table, _, _ = RegraspReasoner.find_feasible_gids(
                    goal_pose=[init_pos_for_feasible, init_rotmat_for_feasible],
                    obstacle_list=[init_obj],
                    toggle_dbg=False
                )

                analysis_goal_available_gids_robot_table, _, _ = RegraspReasoner.find_feasible_gids(
                    goal_pose=[goal_pos_for_feasible, goal_rotmat_for_feasible],
                    obstacle_list=[init_obj],
                    toggle_dbg=False
                )

                common_id = find_common_id(analysis_init_available_gids_robot_table, analysis_goal_available_gids_robot_table)
                end_time = time.time()
                execution_time = end_time - start_time
                analysis_execution_times[grasp_id].append(execution_time)

                # 创建真实标签
                gt_feasible_grasp_ids = np.zeros(len(object_feasible_grasps))
                if common_id is not None:
                    gt_feasible_grasp_ids[list(set(common_id))] = 1
                
                # 存储真实标签
                all_ground_truths[grasp_id].append(gt_feasible_grasp_ids)

                # 开始测量神经网络预测时间
                start_time = time.time()
                
                # 神经网络预测部分
                # init_features = []
                # goal_features = []
                # for index, grasp_pose in enumerate(object_feasible_grasps):
                #     init_features.append(np.concatenate([init_pos_for_feasible, R.from_matrix(init_rotmat_for_feasible).as_quat(),
                #                                          obj_encoder.transform([[init_stable_random_indices[sample_idx]]]).copy()[0],
                #                                          grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                #                                          ]))
                #
                #     goal_features.append(np.concatenate([goal_pos_for_feasible, R.from_matrix(goal_rotmat_for_feasible).as_quat(),
                #                                          obj_encoder.transform([[goal_stable_random_indices[sample_idx]]]).copy()[0],
                #                                          grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                #                                          ]))
                #
                # # 将特征转换为PyTorch张量
                # init_features = torch.from_numpy(np.array(init_features).astype(np.float32)).to(device)
                # goal_features = torch.from_numpy(np.array(goal_features).astype(np.float32)).to(device)
                # input_features = torch.cat([init_features, goal_features], dim=0)  # 形状为 (2N, 15)
                #
                # # 模型推理
                # with torch.no_grad():  # 不需要梯度计算
                #     energies = models['init'](input_features).cpu().numpy() # 形状为 (2N, 1)
                # init_energies = energies[:len(init_features)]
                # goal_energies = energies[len(init_features):]
                #
                # init_binary = (init_energies < thresholds_dict['init']).astype(np.int32)
                # goal_binary = (goal_energies < thresholds_dict['goal']).astype(np.int32)
                # pred = np.minimum(init_binary, goal_binary)
                # pred = pred.squeeze()

                input_features = []
                for index, grasp_pose in enumerate(object_feasible_grasps):
                    input_features.append(np.concatenate([init_pos_for_feasible, R.from_matrix(init_rotmat_for_feasible).as_quat(),
                                                         goal_pos_for_feasible, R.from_matrix(goal_rotmat_for_feasible).as_quat(),
                                                         grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                         ]))

                # 将特征转换为PyTorch张量
                input_features = torch.from_numpy(np.array(input_features).astype(np.float32)).to(device)

                # 模型推理
                with torch.no_grad():  # 不需要梯度计算
                    energies = models['init'](input_features).cpu().numpy() # 形状为 (2N, 1)

                shared_binary = (energies < thresholds_dict['init']).astype(np.int32)
                pred = shared_binary.squeeze()

 
                # 测量执行时间结束
                end_time = time.time()
                execution_time = end_time - start_time
                nn_execution_times[grasp_id].append(execution_time)

                # 存储预测结果和能量
                all_predictions[grasp_id].append(pred)
                init_obj.detach()
                del init_obj

                # 更新进度条
                pbar.update(1)
        
        del RegraspReasoner

    # 在运行完所有迭代后统一计算精度
    precision_rates = {}
    recall_rates = {}
    f1_scores = {}

    for grasp_id in GRASP_IDS:
        # 合并所有迭代的预测和真实标签
        all_preds = np.concatenate(all_predictions[grasp_id])
        all_truths = np.concatenate(all_ground_truths[grasp_id])
        
        # 计算总体精度
        precision = precision_score(all_truths, all_preds, average='binary')
        recall = recall_score(all_truths, all_preds, average='binary')
        f1 = f1_score(all_truths, all_preds, average='binary')
 
        precision_rates[grasp_id] = precision
        recall_rates[grasp_id] = recall
        f1_scores[grasp_id] = f1
        
    # 创建结果字典
    results = {
        'grasp_id': list(GRASP_IDS),
        'precision': [precision_rates[gid] for gid in GRASP_IDS],
        'recall': [recall_rates[gid] for gid in GRASP_IDS],
        'f1_score': [f1_scores[gid] for gid in GRASP_IDS],
        'model_init': [os.path.basename(args.model_init_path)] * len(GRASP_IDS),
        'model_goal': [os.path.basename(args.model_goal_path)] * len(GRASP_IDS),
        'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")] * len(GRASP_IDS)
    }
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 定义CSV文件路径
    csv_path = os.path.join(os.path.dirname(args.model_init_path), 'planning_efficiency_results.csv')
    
    # 如果文件不存在，创建新文件并写入表头
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        # 如果文件存在，追加数据
        df.to_csv(csv_path, mode='a', header=False, index=False)
    
    # 仍然打印结果以便实时查看
    print("\n结果已保存到:", csv_path)
    print("precision_rates: ", precision_rates)
    print("recall_rates: ", recall_rates)
    print("f1_scores: ", f1_scores)
    
        


