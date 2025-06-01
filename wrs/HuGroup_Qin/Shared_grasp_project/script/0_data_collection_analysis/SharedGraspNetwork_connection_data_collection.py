""" 
Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241202 Osaka Univ.

"""

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
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env_without_table_collision import nova2_gripper_v3_without_table
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import SharedGraspAnimeData
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import PickPlacePlannerFromModel
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import trajectory_feasibility_identification
from wrs.grasping.reasoner import GraspReasoner
from wrs.manipulation.placement.flatsurface import FSReferencePoses
from tqdm import tqdm
from scipy.stats import qmc  # 添加到文件开头的import部分
import gc
import argparse
import torch  # 添加PyTorch，以防将来使用


# world configuration
base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)
BASE_PATH = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle"
GRASP_DATA_PREFIX = "bottle_grasp"
SAVE_PREFIX = "SharedGraspNetwork_bottle_experiment_data_final_connection"


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


def robot_env_setup_without_table():
    # robot configuration
    robot = nova2_gripper_v3_without_table(enable_cc=True)
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


# Function to find common IDs and remove duplicates
def find_common_id(ref_graspid, given_graspid):
    if not given_graspid or not ref_graspid:
        return None
    common_id = set(ref_graspid) & set(given_graspid)
    return list(common_id) if common_id else None


def append_save(data, path):
    """使用追加模式写入数据，避免读取整个文件"""
    try:
        # 如果文件不存在，直接写入
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return
        
        # 如果文件存在，使用临时文件
        temp_path = path + '.temp'
        
        # 复制原文件到临时文件
        import shutil
        shutil.copy2(path, temp_path)
        
        # 读取原有数据并追加新数据
        with open(temp_path, 'rb') as f:
            existing_data = pickle.load(f)
        
        existing_data.extend(data)
        
        # 写入合并后的数据到临时文件
        with open(temp_path, 'wb') as f:
            pickle.dump(existing_data, f)
        
        # 替换原文件
        os.replace(temp_path, path)
        
        # 清理内存
        del existing_data
        gc.collect()
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if 'existing_data' in locals():
            del existing_data
        gc.collect()


def show_common_grasp(obj_init_pos, obj_init_rotmat, obj_goal_pos, obj_goal_rotmat,
                      obstacle, common_id, object_feasible_grasps, gripper):
    robot = robot_env_setup()
    # robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    obj_init_pos = obj_init_pos.copy()
    obj_goal_pos = obj_goal_pos.copy()
    obj_init_rotmat = obj_init_rotmat.copy()
    obj_goal_rotmat = obj_goal_rotmat.copy()

    init_obj = obj_setup(name="init_obj", pos=obj_init_pos, rotmat=obj_init_rotmat, alpha=1)
    goal_obj = obj_setup(name="goal_obj", pos=obj_goal_pos, rotmat=obj_goal_rotmat, alpha=1)


    if common_id is None:
        return None
        
    # show common grasp
    for index in range(len(common_id)):
        init_grasp_pos = obj_init_rotmat @ object_feasible_grasps._grasp_list[common_id[index]].ac_pos + obj_init_pos
        init_grasp_rotmat = obj_init_rotmat @ object_feasible_grasps._grasp_list[common_id[index]].ac_rotmat
        
        goal_grasp_pos = obj_goal_rotmat @ object_feasible_grasps._grasp_list[common_id[index]].ac_pos + obj_goal_pos
        goal_grasp_rotmat = obj_goal_rotmat @ object_feasible_grasps._grasp_list[common_id[index]].ac_rotmat
        
        gripper.grip_at_by_pose(
            jaw_center_pos=init_grasp_pos,
            jaw_center_rotmat=init_grasp_rotmat,
            jaw_width=object_feasible_grasps._grasp_list[index].ee_values
        )
        gripper.gen_meshmodel(rgb=rm.const.hug_blue, alpha=1).attach_to(base)
        jnv_values = robot.ik(init_grasp_pos, init_grasp_rotmat, option='multiple')
        for jnv_value in jnv_values:
            robot.goto_given_conf(jnt_values=jnv_value)
            if not robot.is_collided(obstacle_list = [init_obj]):
                robot.gen_meshmodel(alpha=0.5).attach_to(base)

        gripper.grip_at_by_pose(
            jaw_center_pos=goal_grasp_pos,
            jaw_center_rotmat=goal_grasp_rotmat,
            jaw_width=object_feasible_grasps._grasp_list[index].ee_values
        )
        gripper.gen_meshmodel(rgb=rm.const.orange, alpha=1).attach_to(base)
        jnv_values = robot.ik(goal_grasp_pos, goal_grasp_rotmat, option='multiple')
        for jnv_value in jnv_values:
            robot.goto_given_conf(jnt_values=jnv_value)
            if not robot.is_collided(obstacle_list = [goal_obj]):
             robot.gen_meshmodel(alpha=0.5).attach_to(base)

    init_obj.attach_to(base)
    goal_obj.attach_to(base)
    base.run()


def show_grasp(robot_env, obj_cmodel, obj_pos, obj_rotmat, grasp_id, object_feasible_grasps, gripper):
    # robot_env.gen_meshmodel(alpha=1, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    # robot_env.show_cdprim()
    obj_cmodel.attach_to(base)

    if grasp_id is None:
        return None

    for index in grasp_id:
        grasp_pos = obj_rotmat @ object_feasible_grasps._grasp_list[index].ac_pos + obj_pos
        grasp_rotmat = obj_rotmat @ object_feasible_grasps._grasp_list[index].ac_rotmat

        gripper.grip_at_by_pose(
            jaw_center_pos=grasp_pos,
            jaw_center_rotmat=grasp_rotmat,
            jaw_width=object_feasible_grasps._grasp_list[index].ee_values
        )
        gripper.gen_meshmodel(toggle_cdprim=True, alpha=1).attach_to(base)
        jnv_values = robot_env.ik(grasp_pos, grasp_rotmat, option='multiple')
        if jnv_values is not None:
                jnv_value = jnv_values[-1]
                robot_env.goto_given_conf(jnt_values=jnv_value)
                robot_env.gen_meshmodel(alpha=1).attach_to(base)
        robot_env.show_cdprim()
    base.run()


def get_file_paths(grasp_id):
    """根据grasp_id生成对应的文件路径"""
    grasp_data_path = os.path.join(BASE_PATH, f"{GRASP_DATA_PREFIX}_{grasp_id}.pickle")
    save_path = os.path.join(BASE_PATH, f"{SAVE_PREFIX}_{grasp_id}.pickle")
    return grasp_data_path, save_path


def process_batch(batch_data, save_path):
    """单独处理每个批次的数据"""
    try:
        append_save(batch_data, save_path)
    finally:
        # 更彻底的清理
        for item in batch_data:
            for subitem in item:
                if isinstance(subitem, list):
                    subitem.clear()
        batch_data.clear()
        gc.collect()
    return []


def are_lists_identical(list1, list2):
    # 如果两个都是None，返回True
    if list1 is None and list2 is None:
        return True

    # 如果只有一个是None，返回False
    if list1 is None or list2 is None:
        return False

    # 两个都不是None，比较它们的元素
    return set(list1) == set(list2)




def parse_args():
    parser = argparse.ArgumentParser(description='共享抓取数据收集')
    parser.add_argument('--grasp_ids', type=int, nargs='+', default=[57],
                        help='要处理的抓取ID列表')
    parser.add_argument('--total_iterations', type=int, default=int(1e4),
                        help='总迭代次数')
    parser.add_argument('--save_batch_size', type=int, default=1000,
                        help='每批保存的数据大小')
    parser.add_argument('--seed', type=int, default=231,
                        help='随机种子')
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

    # 添加障碍物采样器
    obst_pos_sampler = qmc.LatinHypercube(d=2, seed=SEED+2)  # 使用不同的种子
    obst_samples = obst_pos_sampler.random(n=TOTAL_ITERATIONS)
    ranges = np.array([
        [-0.45, .45],         # pos_x
        [0.1, .6],            # pos_y
        [0, 2 * np.pi],       # theta
    ])
    # ranges = np.array([
    #     [-0.19, .19],         # pos_x
    #     [0.25, .7],            # pos_y
    #     [0, 2 * np.pi],       # theta
    # ])

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
    robot_without_table = robot_env_setup_without_table()
    gripper = wrs_gripper_v3.WRSGripper3()
    collect_data = []

    # object configuration
    obj_cmodel = mcm.CollisionModel(initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_init_pos = np.array([-0.424  ,0.156  ,0.   ])
    obj_init_rotmat = rm.rotmat_from_euler(0, 0, 0)

    obj_goal_pos = obj_init_pos.copy()
    obj_goal_rotmat = obj_init_rotmat.copy()
    obj_cmodel_copy = obj_cmodel.copy()
    

    #stable placement generation
    fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)

    # 生成所有可能的组合
    all_combinations = np.array(np.meshgrid(
        np.arange(len(fs_reference_poses)),  # init姿态的所有可能（0-4）
        np.arange(len(fs_reference_poses))   # goal姿态的所有可能（0-4）
    )).T.reshape(-1, 2)  # 形状为 (25, 2)

    # 计算每种组合需要重复的次数
    repeats = TOTAL_ITERATIONS // len(all_combinations) + 1

    # 重复组合以达到所需的迭代次数
    repeated_combinations = np.tile(all_combinations, (repeats, 1))

    # 随机打乱顺序
    np.random.shuffle(repeated_combinations)

    # 截取需要的数量
    init_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 0]
    goal_stable_random_indices = repeated_combinations[:TOTAL_ITERATIONS, 1]

    # 创建轨迹规划器
    planner = PickPlacePlannerFromModel(robot)

    # 为每个grasp_id处理数据
    for grasp_id in GRASP_IDS:
        print(f"正在处理 grasp_id: {grasp_id}")
        grasp_data_path, common_id_save_path = get_file_paths(str(grasp_id))
        
        if not os.path.exists(grasp_data_path):
            print(f"错误: 找不到文件 {grasp_data_path}")
            continue

        # grasp planning
        object_feasible_grasps = grasp_load(grasp_data_path)
        RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)
        RegraspReasoner_without_table = GraspReasoner(robot_without_table, object_feasible_grasps)

        with tqdm(total=TOTAL_ITERATIONS, desc=f"Processing grasp_id {grasp_id}") as pbar:
            collect_data = []
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
                init_obj = obj_setup(name="init_obj", pos=init_pos_for_feasible, rotmat=init_rotmat_for_feasible, alpha=0.3)
                init_available_gids_robot_table, _ , _ = RegraspReasoner.find_feasible_gids(
                    goal_pose=[init_pos_for_feasible, init_rotmat_for_feasible],
                    obstacle_list=[init_obj],
                    toggle_dbg=False
                )
                goal_obj = obj_setup(name="goal_obj", pos=goal_pos_for_feasible, rotmat=goal_rotmat_for_feasible, alpha=0.3)
                goal_available_gids_robot_table, _, _ = RegraspReasoner.find_feasible_gids(
                    goal_pose=[goal_pos_for_feasible, goal_rotmat_for_feasible],
                    obstacle_list=[goal_obj],
                    toggle_dbg=False
                )

                # 找到共同的抓取ID
                common_id = find_common_id(init_available_gids_robot_table, goal_available_gids_robot_table)

                if common_id:
                    # 将rotmat转换为欧拉角
                    init_euler = rm.rotmat_to_euler(init_rotmat_for_feasible)
                    goal_euler = rm.rotmat_to_euler(goal_rotmat_for_feasible)
                    init_pose = [init_pos_for_feasible[0], init_pos_for_feasible[1], init_pos_for_feasible[2],
                                 init_euler[0], init_euler[1], init_euler[2]]
                    goal_pose = [goal_pos_for_feasible[0], goal_pos_for_feasible[1], goal_pos_for_feasible[2],
                                 goal_euler[0], goal_euler[1], goal_euler[2]]

                    # 创建共享抓取动画数据
                    shared_grasp_data = SharedGraspAnimeData(
                        [init_pose], [goal_pose],
                        [], [common_id], [],
                        robot.end_effector, object_feasible_grasps
                    )

                    # 生成轨迹数据
                    # print("开始判断轨迹可行性")
                    trajectory_feasible_grasp_ids = trajectory_feasibility_identification(planner, shared_grasp_data, init_obj.copy(),
                                                                 robot, object_feasible_grasps)
                else:
                    trajectory_feasible_grasp_ids = None

                collect_data.append([[obj_init_pos, obj_init_rotmat],
                                     list(set(init_available_gids_robot_table)) if init_available_gids_robot_table is not None else None,
                                     init_stable_random_indices[sample_idx].copy(),

                                    [obj_goal_pos, obj_goal_rotmat],
                                    list(set(goal_available_gids_robot_table)) if goal_available_gids_robot_table is not None else None,
                                    goal_stable_random_indices[sample_idx].copy(),

                                    common_id if common_id is not None else None,
                                    trajectory_feasible_grasp_ids if trajectory_feasible_grasp_ids is not None else None])
                del init_obj
                del goal_obj

                if (sample_idx + 1) % SAVE_BATCH_SIZE == 0:
                    process_batch(collect_data, common_id_save_path)
                    collect_data.clear()
                    collect_data = []  # 明确重置列表
                    gc.collect()

                pbar.update(1)

            # 处理最后一个不完整的批次
            if collect_data:
                process_batch(collect_data, common_id_save_path)
                gc.collect()



