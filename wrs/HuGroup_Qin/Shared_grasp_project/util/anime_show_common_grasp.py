""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241229 Osaka Univ.

"""
import sys
sys.path.append("H:/Qin/wrs")  
import time
import pickle
import numpy as np
from wrs import rm, wd, mcm, mgm
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
import wrs.motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)
def env_setup():
    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    # robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    return robot

class AnimeData(object):
    def __init__(self, robot, init_poses, goal_poses, common_ids, gripper, object_feasible_grasps,
                 initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl"):
        self.init_obj_cmodel = mcm.CollisionModel(
            name="init_obj",
            initor=initor,
            alpha=0.9
        )
        self.goal_obj_cmodel = mcm.CollisionModel(
            name="goal_obj",
            initor=initor,
            alpha=0.5
        )
        self.robot = robot
        self.init_poses = init_poses
        self.goal_poses = goal_poses
        self.gripper = gripper
        self.object_feasible_grasps = object_feasible_grasps
        self.common_ids = common_ids
        self.counter = 0
        self.gripper_models = []  # 存储夹爪模型的列表
        self.robot_cmodels = []  # 新增：用于存储所有机器人模型的列表

def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def update(anime_data, task):
    if anime_data.counter >= len(anime_data.common_ids):
        anime_data.counter = 0

    if base.inputmgr.keymap["space"] is True:
        # 清除之前的模型
        anime_data.init_obj_cmodel.detach()
        anime_data.goal_obj_cmodel.detach()
        for model in anime_data.gripper_models:
            model.detach()
        for model in anime_data.robot_cmodels:
            model.detach()
        anime_data.gripper_models.clear()  # 确保清空列表
        anime_data.robot_cmodels.clear()

        # 获取当前的common_ids并打印调试信息
        current_common_ids = anime_data.common_ids[anime_data.counter]
        print(f"当前counter: {anime_data.counter}")
        print(f"当前common_ids: {current_common_ids}")

        # 检查并显示抓取
        if current_common_ids is not None and len(current_common_ids) > 0:
            print(f"开始处理 {len(current_common_ids)} 个抓取点")
            # 获取姿态
            init_pos = anime_data.init_poses[anime_data.counter][:3]
            init_rotmat = rm.rotmat_from_euler(*anime_data.init_poses[anime_data.counter][3:])
            goal_pos = anime_data.goal_poses[anime_data.counter][:3]
            goal_rotmat = rm.rotmat_from_euler(*anime_data.goal_poses[anime_data.counter][3:])

            # 显示物体
            anime_data.init_obj_cmodel.pos = init_pos
            anime_data.init_obj_cmodel.rotmat = init_rotmat
            anime_data.init_obj_cmodel.show_local_frame()
            anime_data.init_obj_cmodel.attach_to(base)

            anime_data.goal_obj_cmodel.pos = goal_pos
            anime_data.goal_obj_cmodel.rotmat = goal_rotmat
            anime_data.goal_obj_cmodel.show_local_frame()
            anime_data.goal_obj_cmodel.attach_to(base)

            for grasp_id in current_common_ids:
                grasp = anime_data.object_feasible_grasps._grasp_list[grasp_id]
                print(f"处理抓取ID: {grasp_id}")

                # 初始位置的抓取
                anime_data.gripper.grip_at_by_pose(
                    jaw_center_pos=init_rotmat @ grasp.ac_pos + init_pos,
                    jaw_center_rotmat=init_rotmat @ grasp.ac_rotmat,
                    jaw_width=grasp.ee_values
                )
                gripper_model = anime_data.gripper.gen_meshmodel(alpha=1, rgb=rm.const.steel_blue)
                gripper_model.attach_to(base)
                anime_data.gripper_models.append(gripper_model)

                ik_jnv = anime_data.robot.ik(init_rotmat @ grasp.ac_pos + init_pos,
                                            init_rotmat @ grasp.ac_rotmat,
                                            option='single')
                if ik_jnv is not None:
                    anime_data.robot.goto_given_conf(ik_jnv)
                    robot_model = anime_data.robot.gen_meshmodel(toggle_tcp_frame=False,
                                                               toggle_jnt_frames=False)
                    robot_model.attach_to(base)
                    anime_data.robot_cmodels.append(robot_model)

                # 目标位置的抓取
                anime_data.gripper.grip_at_by_pose(
                    jaw_center_pos=goal_rotmat @ grasp.ac_pos + goal_pos,
                    jaw_center_rotmat=goal_rotmat @ grasp.ac_rotmat,
                    jaw_width=grasp.ee_values
                )
                gripper_model = anime_data.gripper.gen_meshmodel(alpha=0.5, rgb=rm.const.steel_blue)
                gripper_model.attach_to(base)
                anime_data.gripper_models.append(gripper_model)

                ik_jnv = anime_data.robot.ik(goal_rotmat @ grasp.ac_pos + goal_pos,
                                            goal_rotmat @ grasp.ac_rotmat,
                                            option='single')
                if ik_jnv is not None:
                    anime_data.robot.goto_given_conf(ik_jnv)
                    robot_model = anime_data.robot.gen_meshmodel(alpha=0.5,
                                                               toggle_tcp_frame=False,
                                                               toggle_jnt_frames=False)
                    robot_model.attach_to(base)
                    anime_data.robot_cmodels.append(robot_model)

                print(f"成功显示抓取ID: {grasp_id}")
                break
            else:
                print(f"处理抓取ID {grasp_id} 时出错")
        else:
            print("当前没有可行抓取点")

        anime_data.counter += 1
        time.sleep(0.1)

    return task.cont

def anime_show_common_grasp(robot, object_feasible_grasps, gripper, common_grasp_data_path):
    # 读取数据
    common_grasp_data = grasp_load(common_grasp_data_path)
    init_poses = []
    goal_poses = []
    common_ids = []

    # 处理数据
    for item in common_grasp_data:
        init_pos = np.array(item[0][0]).flatten()
        init_rot = rm.rotmat_to_euler(item[0][1])  # 转换为欧拉角
        init_poses.append(np.concatenate([init_pos, init_rot]))

        goal_pos = np.array(item[5][0]).flatten()
        goal_rot = rm.rotmat_to_euler(item[5][1])  # 转换为欧拉角
        goal_poses.append(np.concatenate([goal_pos, goal_rot]))

        common_ids.append(item[-1])

    # 创建动画数据对象
    anime_data = AnimeData(robot, init_poses, goal_poses, common_ids, gripper, object_feasible_grasps)

    # 设置更新任务
    taskMgr.doMethodLater(0.01, update, "update", extraArgs=[anime_data], appendTask=True)
    base.run()


if __name__ == "__main__":
    robot = env_setup()
    # 调用示例
    object_feasible_grasps_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle"
    object_feasible_grasps = grasp_load(object_feasible_grasps_path)
    gripper = wrs_gripper_v3.WRSGripper3()
    common_grasp_data_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_109.pickle"
    anime_show_common_grasp(robot, object_feasible_grasps, gripper, common_grasp_data_path)