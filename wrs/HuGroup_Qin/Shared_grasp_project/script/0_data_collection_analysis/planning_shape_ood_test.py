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
from sklearn.metrics import precision_recall_curve
base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)


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


def obj_setup(name, pos, rotmat, rgb=None, alpha=None, initor=None):
    # we only consider SE(2) DOF for the object
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha, 
                  initor=initor)
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
    parser.add_argument('--grasp_ids', type=int, default=None,
                        help='要处理的抓取ID列表')
    parser.add_argument('--total_iterations', type=int, default=int(30000),
                        help='总迭代次数')
    parser.add_argument('--pre_num', type=int, default=int(100),
                        help='预采样次数')
    parser.add_argument('--save_batch_size', type=int, default=1000,
                        help='每批保存的数据大小')
    parser.add_argument('--seed', type=int, default=24,
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
    parser.add_argument('--target_env_type', type=str, default=None,
                        help='物体路径')
    parser.add_argument('--grasp_data_path', type=str, default=None,
                        help='抓取数据路径')
    parser.add_argument('--objects_str', type=str, default=None,
                        help='物体字符串')
    parser.add_argument('--target_objects_str', type=str, default=None,
                        help='目标物体字符串')
    
    parser.add_argument('--shared_flag', type=str, default='J',
                        help='是否使用共享模型')
    parser.add_argument('--validate_grasp_data_path', type=str, default=None,
                        help='验证抓取数据路径')
    parser.add_argument('--validate_env_data_path', type=str, default=None,
                        help='验证物体路径')

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
    GRASP_IDS = [args.grasp_ids]
    TOTAL_ITERATIONS = args.total_iterations
    SAVE_BATCH_SIZE = args.save_batch_size
    OBJ_PATH = args.target_env_type

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

    # 标准模式：使用两个模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    thresholds_dict = {}
    if args.shared_flag == 'L' or args.shared_flag == 'J':
        args.input_dim = 14
    elif args.shared_flag == 'D':
        args.input_dim = 21
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

        if args.shared_flag == 'J':
            # 寻找最佳阈值 300个正负样本对； 然后利用不同数量的正负样本对，计算不同数量的阈值；
            pos_sample_num = 0
            obj_init_pos_list = []
            obj_init_rotmat_list = []
            obj_goal_pos_list = []
            obj_goal_rotmat_list = []
            common_id_list = []
            # 从完整路径中提取物体名称并去掉.stl扩展名
            if os.path.exists(rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{args.objects_str}_calibarate_data.pkl'):
                with open(rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{args.objects_str}_calibarate_data.pkl', 'rb') as f:
                    data_list = pickle.load(f)
                obj_init_pos_list = data_list['init_pos']
                init_rotmat_quat_list = data_list['init_quat']
                obj_goal_pos_list = data_list['goal_pos']
                goal_rotmat_quat_list = data_list['goal_quat']
                common_id_list = data_list['common_id']
                
                # 将四元数转换回旋转矩阵
                obj_init_rotmat_list = [R.from_quat(quat).as_matrix() for quat in init_rotmat_quat_list]
                obj_goal_rotmat_list = [R.from_quat(quat).as_matrix() for quat in goal_rotmat_quat_list]
            else:
                with tqdm(total=TOTAL_ITERATIONS, desc=f"Processing grasp_id {grasp_id}") as pbar:

                    obj_grasp_data_path = args.validate_grasp_data_path.split(',')
                    obj_env_data_path = args.validate_env_data_path.split(',')

                    # TODO： 随机选择obj然后收集数据集；
                    for grasp_path, env_path in zip(obj_grasp_data_path, obj_env_data_path):
                        grasp_data_path = grasp_path
                        object_feasible_grasps = grasp_load(grasp_data_path)
                        RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)

                        # object configuration
                        obj_cmodel = mcm.CollisionModel(initor=env_path)
                        fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
                        all_combinations = np.array(np.meshgrid(
                            np.arange(len(fs_reference_poses)),  # init姿态的所有可能
                            np.arange(len(fs_reference_poses))   # goal姿态的所有可能
                        )).T.reshape(-1, 2)  # 形状为 (25, 2)
                        repeats = TOTAL_ITERATIONS // len(all_combinations) + 1
                        repeated_combinations = np.tile(all_combinations, (repeats, 1))
                        np.random.shuffle(repeated_combinations)
                        init_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 0]
                        goal_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 1]

                        for sample_idx in range(TOTAL_ITERATIONS):
                            # 保存原始采样位置
                            sample_idx += 5000
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

                            # 计算可行抓取 - robot_table_obj
                            init_obj = obj_setup(name="init_obj", pos=init_pos_for_feasible, 
                                            rotmat=init_rotmat_for_feasible, alpha=0.3, initor=env_path)

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
                            pbar.update(1)
                            init_obj.detach()
                            del init_obj

                            if common_id is not None:
                                filtered_common_id = [idx for idx in common_id if idx < 180] # 只考虑前180个抓取
                                if len(filtered_common_id) > 0:
                                    obj_init_pos_list.append(init_pos_for_feasible)
                                    obj_init_rotmat_list.append(init_rotmat_for_feasible)
                                    obj_goal_pos_list.append(goal_pos_for_feasible)
                                    obj_goal_rotmat_list.append(goal_rotmat_for_feasible)
                                    temp_id = np.zeros(180) # 180个抓取
                                    temp_id[filtered_common_id] = 1
                                    common_id_list.append(temp_id)
                                    pos_sample_num += np.count_nonzero(filtered_common_id)
                                    if pos_sample_num >= 1000: # TODO
                                        pos_sample_num=0
                                        print(f"收集到1000个正样本")
                                        break
                            else:
                                continue
                        
                    # 在保存之前，将旋转矩阵转换为四元数
                    init_rotmat_quat_list = [R.from_matrix(rotmat).as_quat() for rotmat in obj_init_rotmat_list]
                    goal_rotmat_quat_list = [R.from_matrix(rotmat).as_quat() for rotmat in obj_goal_rotmat_list]
                    
                    # 重新组织data_list
                    data_list = {
                        'init_pos': np.array(obj_init_pos_list),
                        'init_quat': np.array(init_rotmat_quat_list),
                        'goal_pos': np.array(obj_goal_pos_list),
                        'goal_quat': np.array(goal_rotmat_quat_list),
                        'common_id': np.array(common_id_list)
                    }
                    # 使用pickle保存数据
                    save_path = rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{args.objects_str}_calibarate_data.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(data_list, f)

            obj_init_pos_list = np.array(obj_init_pos_list, dtype=object)
            obj_init_rotmat_list = np.array(obj_init_rotmat_list, dtype=object)
            obj_goal_pos_list = np.array(obj_goal_pos_list, dtype=object)
            obj_goal_rotmat_list = np.array(obj_goal_rotmat_list, dtype=object)
            common_id_list = np.array(common_id_list)

            # 
            grasp_data_path = args.grasp_data_path
            object_feasible_grasps = grasp_load(grasp_data_path)
            RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)

            # object configuration
            obj_cmodel = mcm.CollisionModel(initor=OBJ_PATH)
            fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
            all_combinations = np.array(np.meshgrid(
                np.arange(len(fs_reference_poses)),  # init姿态的所有可能
                np.arange(len(fs_reference_poses))   # goal姿态的所有可能
            )).T.reshape(-1, 2)  # 形状为 (25, 2)
            repeats = TOTAL_ITERATIONS // len(all_combinations) + 1
            repeated_combinations = np.tile(all_combinations, (repeats, 1))
            np.random.shuffle(repeated_combinations)
            init_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 0]
            goal_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 1]


            # 利用precision_recall_curve计算不同数量的阈值
            init_features = []
            goal_features = []

            # 收集所有批次的能量值
            all_init_energies = []
            all_goal_energies = []
            for i in range(len(obj_init_pos_list)):
                init_pos_for_feasible = obj_init_pos_list[i]
                init_rotmat_for_feasible = obj_init_rotmat_list[i]
                goal_pos_for_feasible = obj_goal_pos_list[i]
                goal_rotmat_for_feasible = obj_goal_rotmat_list[i]

                for index, grasp_pose in enumerate(object_feasible_grasps._grasp_list[0:180]):
                    init_features.append(np.concatenate([init_pos_for_feasible, R.from_matrix(init_rotmat_for_feasible).as_quat(),
                                                        grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                        ]))
                
                    goal_features.append(np.concatenate([goal_pos_for_feasible, R.from_matrix(goal_rotmat_for_feasible).as_quat(),
                                                        grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                        ]))
                
            # 将特征转换为PyTorch张量
            init_features = torch.from_numpy(np.array(init_features).astype(np.float32)).to(device)
            goal_features = torch.from_numpy(np.array(goal_features).astype(np.float32)).to(device)
            input_features = torch.cat([init_features, goal_features], dim=0)  # 形状为 (2N, 14)
            
            # 模型推理
            with torch.no_grad():  # 不需要梯度计算
                energies = models['init'](input_features).cpu().numpy() # 形状为 (2N, 1)
            init_energies = energies[:len(init_features)]
            goal_energies = energies[len(init_features):]
            all_init_energies.append(init_energies)
            all_goal_energies.append(goal_energies)

            all_init_energies = np.concatenate(all_init_energies)
            all_goal_energies = np.concatenate(all_goal_energies)
            all_combined_energies = all_init_energies + all_goal_energies

            # 将多维标签数组展平成一维
            common_id_list = np.array(common_id_list)
            flattened_labels = common_id_list.flatten()

            # 计算precision_recall_curve
            precision, recall, thresholds = precision_recall_curve(flattened_labels, -all_combined_energies)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            if best_idx < len(thresholds):
                best_threshold = -thresholds[best_idx]  # 需要转回原始能量的符号
            else:
                best_threshold = np.min(all_combined_energies) - 1.0
            print(f"最佳阈值: {best_threshold:.3f}")

            

        # 测试数据收集
        obj_init_pos_list = []
        obj_init_rotmat_list = []
        obj_goal_pos_list = []
        obj_goal_rotmat_list = []
        common_id_list = []

        grasp_data_path = args.grasp_data_path
        object_feasible_grasps = grasp_load(grasp_data_path)
        RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)

        # object configuration
        obj_cmodel = mcm.CollisionModel(initor=OBJ_PATH)
        fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
        all_combinations = np.array(np.meshgrid(
            np.arange(len(fs_reference_poses)),  # init姿态的所有可能
            np.arange(len(fs_reference_poses))   # goal姿态的所有可能
        )).T.reshape(-1, 2)  # 形状为 (25, 2)
        repeats = TOTAL_ITERATIONS // len(all_combinations) + 1
        repeated_combinations = np.tile(all_combinations, (repeats, 1))
        np.random.shuffle(repeated_combinations)
        init_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 0]
        goal_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 1]
    
        target_obj_name = os.path.basename(args.target_objects_str).replace('.stl', '')
        if os.path.exists(rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{target_obj_name}_test_data.pkl'):
            with open(rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{target_obj_name}_test_data.pkl', 'rb') as f:
                data_list = pickle.load(f)
            obj_init_pos_list = data_list['init_pos']
            init_rotmat_quat_list = data_list['init_quat']
            obj_goal_pos_list = data_list['goal_pos']
            goal_rotmat_quat_list = data_list['goal_quat']
            common_id_list = data_list['common_id']
            all_ground_truths[grasp_id] = data_list['common_id']
            obj_init_rotmat_list = [R.from_quat(quat).as_matrix() for quat in init_rotmat_quat_list]
            obj_goal_rotmat_list = [R.from_quat(quat).as_matrix() for quat in goal_rotmat_quat_list]
        else:
            # 采集够300个正样本
            pos_sample_num = 0
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
                    init_obj = obj_setup(name="init_obj", pos=init_pos_for_feasible, 
                                    rotmat=init_rotmat_for_feasible, alpha=0.3, initor=OBJ_PATH)

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

                    # 更新进度条
                    pbar.update(1)
                    init_obj.detach()
                    del init_obj

                    # 创建前180个Grasp pose的真实标签
                    gt_feasible_grasp_ids = np.zeros(180)
                    if common_id is not None:
                        filtered_common_id = [idx for idx in common_id if idx < 180]
                        if len(filtered_common_id) > 0:
                            gt_feasible_grasp_ids[filtered_common_id] = 1
                            # 存储真实标签
                            all_ground_truths[grasp_id].append(gt_feasible_grasp_ids)

                            # 保存位置和旋转矩阵信息
                            obj_init_pos_list.append(init_pos_for_feasible)
                            obj_init_rotmat_list.append(init_rotmat_for_feasible)
                            obj_goal_pos_list.append(goal_pos_for_feasible)
                            obj_goal_rotmat_list.append(goal_rotmat_for_feasible)
                            common_id_list.append(gt_feasible_grasp_ids)
            
                            pos_sample_num += np.count_nonzero(filtered_common_id)
                            if pos_sample_num >= 1000:
                                print(f"收集到1000个正样本")
                                break

                # 在循环结束后保存数据
                # 将旋转矩阵转换为四元数用于保存
                init_rotmat_quat_list = [R.from_matrix(rotmat).as_quat() for rotmat in obj_init_rotmat_list]
                goal_rotmat_quat_list = [R.from_matrix(rotmat).as_quat() for rotmat in obj_goal_rotmat_list]
                
                # 重新组织data_list
                data_list = {
                    'init_pos': np.array(obj_init_pos_list),
                    'init_quat': np.array(init_rotmat_quat_list),
                    'goal_pos': np.array(obj_goal_pos_list),
                    'goal_quat': np.array(goal_rotmat_quat_list),
                    'common_id': np.array(common_id_list)
                }
                # 使用pickle保存数据
                save_path = rf'E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\threshold_calibarate_data\{target_obj_name}_test_data.pkl'
                with open(save_path, 'wb') as f:
                    pickle.dump(data_list, f)
                print(f"测试数据已保存到: {save_path}")

        # 开始测量神经网络预测时间
        start_time = time.time()
        
        # 神经网络预测部分 L 方法
        if args.shared_flag == 'L' or args.shared_flag == 'J': 
            predictions = []
            energies_list = []
            
            # 对每个位置进行预测
            for idx in range(len(obj_init_pos_list)):
                init_features = []
                goal_features = []
                
                init_pos = obj_init_pos_list[idx]
                init_rotmat = obj_init_rotmat_list[idx]
                goal_pos = obj_goal_pos_list[idx]
                goal_rotmat = obj_goal_rotmat_list[idx]
                
                for grasp_pose in object_feasible_grasps._grasp_list[0:180]:
                    init_features.append(np.concatenate([init_pos, R.from_matrix(init_rotmat).as_quat(),
                                                      grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                      ]))
                
                    goal_features.append(np.concatenate([goal_pos, R.from_matrix(goal_rotmat).as_quat(),
                                                      grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                      ]))
                
                # 将特征转换为PyTorch张量
                init_features = torch.from_numpy(np.array(init_features).astype(np.float32)).to(device)
                goal_features = torch.from_numpy(np.array(goal_features).astype(np.float32)).to(device)
                input_features = torch.cat([init_features, goal_features], dim=0)  # 形状为 (2N, 14)
                
                # 模型推理
                with torch.no_grad():  # 不需要梯度计算
                    energies = models['init'](input_features).cpu().numpy() # 形状为 (2N, 1)
                init_energies = energies[:len(init_features)]
                goal_energies = energies[len(init_features):]

                if args.shared_flag == 'L':
                    init_binary = (init_energies < thresholds_dict['init']).astype(np.int32)
                    goal_binary = (goal_energies < thresholds_dict['goal']).astype(np.int32)
                    pred = np.minimum(init_binary, goal_binary)
                    pred = pred.squeeze()

                elif args.shared_flag == 'J':
                    final_binary = ((init_energies + goal_energies) < best_threshold).astype(np.int32)
                    pred = final_binary.squeeze()
                
                predictions.append(pred)
                energies_list.append(init_energies + goal_energies)

        elif args.shared_flag == 'D':
            # D 方法
            predictions = []
            energies_list = []
            
            # 对每个位置进行预测
            for idx in range(len(obj_init_pos_list)):
                input_features = []
                
                init_pos = obj_init_pos_list[idx]
                init_rotmat = obj_init_rotmat_list[idx]
                goal_pos = obj_goal_pos_list[idx]
                goal_rotmat = obj_goal_rotmat_list[idx]
                
                for grasp_pose in object_feasible_grasps._grasp_list[0:180]:
                    input_features.append(np.concatenate([init_pos, R.from_matrix(init_rotmat).as_quat(),
                                                        goal_pos, R.from_matrix(goal_rotmat).as_quat(),
                                                        grasp_pose.ac_pos, R.from_matrix(grasp_pose.ac_rotmat).as_quat()
                                                        ]))
                # 将特征转换为PyTorch张量
                input_features = torch.from_numpy(np.array(input_features).astype(np.float32)).to(device)
                
                # 模型推理
                with torch.no_grad():  # 不需要梯度计算
                    energies = models['init'](input_features).cpu().numpy() # 形状为 (N, 1)

                if args.shared_flag == 'D':
                    shared_binary = (energies < thresholds_dict['init']).astype(np.int32)
                    pred = shared_binary.squeeze()
                    
                predictions.append(pred)
                energies_list.append(energies)

        # 测量执行时间结束
        end_time = time.time()
        execution_time = end_time - start_time
        nn_execution_times[grasp_id].append(execution_time)

        # 存储预测结果和能量
        all_predictions[grasp_id] = predictions
        all_energies[grasp_id] = energies_list

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
        'env_name': str(args.target_env_type),
        'shared_flag': args.shared_flag,
        'precision': [precision_rates[gid] for gid in GRASP_IDS],
        'recall': [recall_rates[gid] for gid in GRASP_IDS],
        'f1_score': [f1_scores[gid] for gid in GRASP_IDS],
        'model_name': [os.path.basename(args.model_init_path)] * len(GRASP_IDS),
        'timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")] * len(GRASP_IDS),
        'objects_str': [args.objects_str] * len(GRASP_IDS),
        'target_objects_str': [args.target_objects_str] * len(GRASP_IDS)
    }
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 定义CSV文件路径
    csv_path = os.path.join(os.path.dirname(args.model_init_path), 'planning_shape_ood_results.csv')
    
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
    
        