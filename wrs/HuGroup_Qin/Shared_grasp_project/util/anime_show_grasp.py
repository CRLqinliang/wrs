""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241229 Osaka Univ.

"""
import time
import pickle
import numpy as np
from wrs import rm, wd, mcm
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])

def env_setup():
    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    return robot

class AnimeData(object):
    def __init__(self, obj_poses, grasp_ids, gripper, object_feasible_grasps):
        self.obj_cmodel = mcm.CollisionModel(
            name="init_obj",
            initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl"
        )
        self.obj_poses = obj_poses
        self.gripper = gripper
        self.object_feasible_grasps = object_feasible_grasps
        self.grasp_ids = grasp_ids
        self.counter = 0
        self.gripper_models = []  # 存储夹爪模型的列表

def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def update(anime_data, task):
    if anime_data.counter >= len(anime_data.grasp_ids):
        anime_data.counter = 0

    if base.inputmgr.keymap["space"] is True:
        # 清除之前的模型
        anime_data.obj_cmodel.detach()
        for model in anime_data.gripper_models:
            model.detach()
        anime_data.gripper_models.clear()

        # 获取当前姿态
        obj_pos = anime_data.obj_poses[anime_data.counter][:3]
        obj_rotmat = rm.rotmat_from_euler(*anime_data.obj_poses[anime_data.counter][3:])


        # 显示物体
        anime_data.obj_cmodel.pos = obj_pos
        anime_data.obj_cmodel.rotmat = obj_rotmat
        anime_data.obj_cmodel.show_local_frame()
        anime_data.obj_cmodel.attach_to(base)


        # 显示抓取点
        grasp_ids = anime_data.grasp_ids[anime_data.counter]
        if grasp_ids is not None:  # 检查是否有可行抓取
            for grasp_id in grasp_ids:
                grasp = anime_data.object_feasible_grasps._grasp_list[grasp_id]

                anime_data.gripper.grip_at_by_pose(
                    jaw_center_pos=obj_rotmat @ grasp.ac_pos + obj_pos,
                    jaw_center_rotmat=obj_rotmat @ grasp.ac_rotmat,
                    jaw_width=grasp.ee_values
                )
                gripper_model = anime_data.gripper.gen_meshmodel(rgb=rm.const.hug_blue, alpha=.2)
                gripper_model.attach_to(base)
                anime_data.gripper_models.append(gripper_model)


        anime_data.counter += 1
        time.sleep(0.1)  # 添加延时以避免按键重复触发

    return task.cont

def anime_show_grasp(object_feasible_grasps, gripper, grasp_data_path):
    # 读取数据
    grasp_data = grasp_load(grasp_data_path)
    obj_poses = []
    grasp_ids = []

    # 处理数据
    for item in grasp_data:
        obj_pos = np.array(item[0][0]).flatten()
        obj_rot = rm.rotmat_to_euler(item[0][1])  # 转换为欧拉角
        obj_poses.append(np.concatenate([obj_pos, obj_rot]))

        grasp_ids.append(item[-1])

    # 创建动画数据对象
    anime_data = AnimeData(obj_poses, grasp_ids, gripper, object_feasible_grasps)

    # 设置更新任务
    taskMgr.doMethodLater(0.01, update, "update", extraArgs=[anime_data], appendTask=True)
    base.run()



if __name__ == "__main__":
    env_setup()
    # 调用示例
    object_feasible_grasps_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle"
    object_feasible_grasps = grasp_load(object_feasible_grasps_path)
    gripper = wrs_gripper_v3.WRSGripper3()
    grasp_data_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\SharedGraspNetwork_bottle_experiment_data_109.pickle"
    anime_show_grasp(object_feasible_grasps, gripper, grasp_data_path)