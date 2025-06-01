""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20250112 Osaka Univ.

"""
import numpy as np
import torch
import pickle
import sys
sys.path.append(r"H:\Qin\wrs")
from wrs import rm, wd, mcm, gg, mgm, rrtc, adp
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
from wrs.HuGroup_Qin.Shared_grasp_project.network.MlpBlock import GraspingNetwork
from torch.utils.data import DataLoader, Dataset, random_split
import time
import argparse
from sklearn.preprocessing import OneHotEncoder


class GraspingDataset(Dataset):
    def __init__(self, output_dim, pickle_file, normalize_data=False, stable_label=True):
        self.output_dim = output_dim
        self.normalize_data = normalize_data
        self.stable_label = stable_label

        if pickle_file is None:
            self.position_scaler = StandardScaler()
            return

        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            raise ValueError(f"Error loading file {pickle_file}: {e}")

        # 提取初始位姿和目标位姿
        init_poses = []
        goal_poses = []
        for item in data:
            init_pos = np.array(item[0][0]).flatten()
            init_rot = rm.rotmat_to_euler(item[0][1])
            init_poses.append(np.concatenate([init_pos, init_rot]))
            goal_pos = np.array(item[1][0]).flatten()
            goal_rot = rm.rotmat_to_euler(item[1][1])
            goal_poses.append(np.concatenate([goal_pos, goal_rot]))

        init_poses = np.array(init_poses, dtype=np.float32)
        goal_poses = np.array(goal_poses, dtype=np.float32)

        # 分离位置和角度数据
        init_positions = init_poses[:, :3]
        init_angles = init_poses[:, 3:]
        goal_positions = goal_poses[:, :3]
        goal_angles = goal_poses[:, 3:]

        # 根据normalize_data参数决定是否进行标准化
        if self.normalize_data:
            # 合并所有位置数据进行标准化
            all_positions = np.vstack([init_positions, goal_positions])
            self.position_scaler = StandardScaler()
            all_positions_scaled = self.position_scaler.fit_transform(all_positions)

            # 分别获取标准化后的初始位置和目标位置
            init_positions_scaled = all_positions_scaled[:len(init_positions)]
            goal_positions_scaled = all_positions_scaled[len(init_positions):]
        else:
            # 不进行标准化，直接使用原始数据
            init_positions_scaled = init_positions
            goal_positions_scaled = goal_positions
            self.position_scaler = None

        # 是否增加稳定标签到输入里面
        if self.stable_label:
            # 获取原始标签并重塑为二维数组
            init_raw_labels = np.array([item[2] for item in data], dtype=int).reshape(-1, 1)
            goal_raw_labels = np.array([item[3] for item in data], dtype=int).reshape(-1, 1)

            # goal_raw_labels = init_raw_labels # 在相同的stable placement下，init和goal的label是一样的

            # 创建OneHotEncoder并指定类别
            encoder = OneHotEncoder(categories=[np.array([0, 1, 2, 3, 4])], sparse=False)

            # 转换标签为one-hot编码
            init_stable_label_one_hot = encoder.fit_transform(init_raw_labels)
            goal_stable_label_one_hot = encoder.fit_transform(goal_raw_labels)

            # 连接所有输入特征
            self.inputs = np.concatenate([
                init_positions_scaled,
                init_angles,
                init_stable_label_one_hot,  # 现在会是一个5维的one-hot向量
                goal_positions_scaled,
                goal_angles,
                goal_stable_label_one_hot  # 现在会是一个5维的one-hot向量
            ], axis=1)
        else:
            # 组合所有特征
            self.inputs = np.concatenate([
                init_positions_scaled,
                init_angles,
                goal_positions_scaled,
                goal_angles
            ], axis=1).astype(np.float32)

        # 处理标签
        self.labels = np.array([self._create_target_vector(item[-1]) for item in data], dtype=np.float32)

    # def _rotmat_to_euler(self, rotmat):
    #     """将旋转矩阵转换为欧拉角(rx, ry, rz)"""
    #     r = R.from_matrix(rotmat)
    #     return r.as_euler('xyz', degrees=False)

    def _create_target_vector(self, label_ids):
        target_vector = np.zeros(self.output_dim, dtype=np.float32)
        if label_ids == None:
            return target_vector
        target_vector[label_ids] = 1
        return target_vector

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_vector = self.inputs[idx].copy()
        label = self.labels[idx]

        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def save_scaler(self, scaler_path):
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.position_scaler, f)

    @staticmethod
    def load_scaler(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)


class SharedGraspAnimeData(object):
    def __init__(self, init_poses, goal_poses, labels, common_ids, common_ids_logits,
                 gripper, object_feasible_grasps,
                 initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl"):

        self.init_obj_cmodel = mcm.CollisionModel(
            name="init_obj",
            initor=initor
        )
        self.goal_obj_cmodel = mcm.CollisionModel(
            name="goal_obj",
            initor=initor
        )
        self.init_poses = init_poses
        self.goal_poses = goal_poses
        self.gripper = gripper
        self.object_feasible_grasps = object_feasible_grasps
        self.predict_common_ids = common_ids
        self.predict_common_ids_logits = common_ids_logits
        self.labels = labels
        self.counter = 0  #
        self.gripper_models = []  # 存储夹爪模型的列表


class TrajectoryAnimeData(object):
    def __init__(self, mot_data_dict):
        self.counter = 0
        self.mot_data_dict = mot_data_dict

class SharedGraspTrajectoryAnimeData(object):
    def __init__(self, shared_grasp_data, trajectory_data):
        self.shared_grasp_data = shared_grasp_data
        self.trajectory_data = trajectory_data


class PickPlacePlannerFromModel(adp.ADPlanner):

    def __init__(self, robot):
        """
        :param object:
        :param robot: must be an instance of SglArmRobotInterface
        author: weiwei, hao
        date: 20191122, 20210113, 20240316
        """
        super().__init__(robot)
        self.robot = robot

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def reason_common_gids(self,
                           grasp_collection,
                           goal_pose_list,
                           obstacle_list=None,
                           toggle_dbg=False):
        """
        find the common collision free and IK feasible gids
        :param eef: an end effector instance
        :param grasp_collection: grasping.grasp.GraspCollection
        :param goal_pose_list: [[pos0, rotmat0]], [pos1, rotmat1], ...]
        :param obstacle_list
        :return: common grasp poses
        author: weiwei
        date: 20210113, 20210125
        """
        # start reasoning
        previous_available_gids = range(len(grasp_collection))
        intermediate_available_gids = []
        eef_collided_grasps_num = 0
        ik_failed_grasps_num = 0
        rbt_collided_grasps_num = 0
        for goal_id, goal_pose in enumerate(goal_pose_list):
            goal_pos = goal_pose[0]
            goal_rotmat = goal_pose[1]
            grasp_and_gid = zip(previous_available_gids,  # need .copy()?
                                [grasp_collection[i] for i in previous_available_gids])
            previous_available_gids = []
            for gid, grasp in grasp_and_gid:
                goal_jaw_center_pos = goal_pos + goal_rotmat.dot(grasp.ac_pos)
                goal_jaw_center_rotmat = goal_rotmat.dot(grasp.ac_rotmat)
                jnt_values = self.robot.ik(tgt_pos=goal_jaw_center_pos, tgt_rotmat=goal_jaw_center_rotmat)
                if jnt_values is not None:
                    self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=grasp.ee_values)
                    if not self.robot.is_collided(obstacle_list=obstacle_list):
                        if not self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
                            previous_available_gids.append(gid)
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.const.green, alpha=1).attach_to(base)
                                # self.robot.gen_meshmodel(rgb=rm.const.green, alpha=.3).attach_to(base)
                        else:  # ee collided
                            eef_collided_grasps_num += 1
                            if toggle_dbg:
                                self.robot.end_effector.gen_meshmodel(rgb=rm.const.yellow, alpha=.3).attach_to(base)
                                # self.robot.gen_meshmodel(rgb=rm.const.yellow, alpha=.3).attach_to(base)
                    else:  # robot collided
                        rbt_collided_grasps_num += 1
                        if toggle_dbg:
                            self.robot.end_effector.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                            # self.robot.gen_meshmodel(rgb=rm.const.orange, alpha=.3).attach_to(base)
                else:  # ik failure
                    ik_failed_grasps_num += 1
                    if toggle_dbg:
                        self.robot.end_effector.grip_at_by_pose(jaw_center_pos=goal_jaw_center_pos,
                                                                jaw_center_rotmat=goal_jaw_center_rotmat,
                                                                jaw_width=grasp.ee_values)
                        self.robot.end_effector.gen_meshmodel(rgb=rm.const.magenta, alpha=.3).attach_to(base)
            intermediate_available_gids.append(previous_available_gids.copy())
            if toggle_dbg:
                print('-----start-----')
                print(f"Number of available grasps at goal-{str(goal_id)}: ", len(previous_available_gids))
                print("Number of collided grasps at goal-{str(goal_id)}: ", eef_collided_grasps_num)
                print("Number of failed IK at goal-{str(goal_id)}: ", ik_failed_grasps_num)
                print("Number of collided robots at goal-{str(goal_id)}: ", rbt_collided_grasps_num)
                print("------end_type------")
        if toggle_dbg:
            base.run()
        return previous_available_gids

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_moveto(self,
                            obj_cmodel,
                            grasp,
                            moveto_pose_list,
                            moveto_approach_direction_list,
                            moveto_approach_distance_list,
                            moveto_depart_direction_list,
                            moveto_depart_distance_list,
                            start_jnt_values=None,
                            pick_approach_jaw_width=None,
                            pick_approach_direction=None,
                            pick_approach_distance=.07,
                            pick_depart_direction=None,
                            pick_depart_distance=.07,
                            linear_granularity=.02,
                            obstacle_list=None,
                            use_rrt=True,
                            toggle_dbg=False):
        """
        pick and move an object to multiple poses
        :param obj_cmodel:
        :param grasp:
        :param moveto_pose_list: [[pos, rotmat], [pos, rotmat], ...]
        :param moveto_approach_direction_list:
        :param moveto_approach_distance_list:
        :param moveto_depart_direction_list:
        :param moveto_depart_distance_list:
        :param start_jnt_values: None means starting from the linear end of the pick motion
        :param pick_approach_direction
        :param pick_approach_distance
        :param pick_depart_direction
        :param pick_depart_distance
        :param linear_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        # pick up object
        pick_tcp_pos = obj_cmodel.rotmat.dot(grasp.ac_pos) + obj_cmodel.pos
        pick_tcp_rotmat = obj_cmodel.rotmat.dot(grasp.ac_rotmat)
        if pick_approach_jaw_width is None:
            pick_approach_jaw_width = self.robot.end_effector.jaw_range[1]
        pick_motion = self.gen_approach(goal_tcp_pos=pick_tcp_pos,
                                        goal_tcp_rotmat=pick_tcp_rotmat,
                                        start_jnt_values=start_jnt_values,
                                        linear_direction=pick_approach_direction,
                                        linear_distance=pick_approach_distance,
                                        ee_values=pick_approach_jaw_width,
                                        linear_granularity=linear_granularity,
                                        obstacle_list=obstacle_list,
                                        object_list=[obj_cmodel],
                                        use_rrt=use_rrt,
                                        toggle_dbg=toggle_dbg)
        if pick_motion is None:
            print("PPPlanner: Error encountered when generating pick approach motion!")
            return None
        obj_cmodel_copy = obj_cmodel.copy()
        # for robot_mesh in pick_motion.mesh_list:
        #     obj_cmodel_copy.attach_to(robot_mesh)
        self.robot.goto_given_conf(pick_motion.jv_list[-1])
        self.robot.hold(obj_cmodel=obj_cmodel_copy, jaw_width=grasp.ee_values)
        pick_motion.extend([pick_motion.jv_list[-1]], toggle_mesh=False)
        pick_depart = self.gen_linear_depart_from_given_conf(start_jnt_values=pick_motion.jv_list[-1],
                                                             direction=pick_depart_direction,
                                                             distance=pick_depart_distance,
                                                             ee_values=None,
                                                             granularity=linear_granularity,
                                                             obstacle_list=obstacle_list,
                                                             toggle_dbg=toggle_dbg)
        if pick_depart is None:
            print("PPPlanner: Error encountered when generating pick depart motion!")
            return None
        else:
            moveto_motion = adp.mpi.motd.MotionData(robot=self.robot)
            # move to goals
            moveto_start_jnt_values = pick_depart.jv_list[-1]
            for i, goal_pose in enumerate(moveto_pose_list):
                goal_tcp_pos = goal_pose[1].dot(grasp.ac_pos) + goal_pose[0]
                goal_tcp_rotmat = goal_pose[1].dot(grasp.ac_rotmat)
                moveto_ap = self.gen_approach_depart(goal_tcp_pos=goal_tcp_pos,
                                                     goal_tcp_rotmat=goal_tcp_rotmat,
                                                     start_jnt_values=moveto_start_jnt_values,
                                                     approach_direction=moveto_approach_direction_list[i],
                                                     approach_distance=moveto_approach_distance_list[i],
                                                     approach_ee_values=None,
                                                     depart_direction=moveto_depart_direction_list[i],
                                                     depart_distance=moveto_depart_distance_list[i],
                                                     depart_ee_values=None,  # do not change jaw width
                                                     linear_granularity=linear_granularity,
                                                     obstacle_list=obstacle_list,
                                                     use_rrt=use_rrt,
                                                     toggle_dbg=toggle_dbg)
                if moveto_ap is None:
                    print(f"Error encountered when generating motion to the {i}th goal!")
                    return None
                else:
                    moveto_motion += moveto_ap
                    moveto_start_jnt_values = moveto_motion.jv_list[-1]
            return pick_motion + pick_depart + moveto_motion

    @adp.mpi.InterplatedMotion.keep_states_decorator
    def gen_pick_and_place(self,
                           obj_cmodel,
                           grasp_collection,
                           goal_pose_list,
                           start_jnt_values=None,
                           end_jnt_values=None,
                           pick_approach_jaw_width=None,
                           pick_approach_direction=None,  # handz
                           pick_approach_distance=None,
                           pick_depart_direction=None,  # handz
                           pick_depart_distance=None,
                           place_approach_direction_list=None,
                           place_approach_distance_list=None,
                           place_depart_direction_list=None,
                           place_depart_distance_list=None,
                           place_depart_jaw_width=None,
                           linear_granularity=.02,
                           use_rrt=True,
                           obstacle_list=None,
                           reason_grasps=True,
                           common_gid_list_ref=None,
                           toggle_dbg=False):
        """
        :param obj_cmodel:
        :param grasp_collection: grasping.grasp.GraspCollection
        :param goal_pose_list:
        :param start_jnt_values: start from the start of pick approach if None
        :param end_jnt_values: end at the end of place depart if None
        :param pick_approach_jaw_width: default value if None
        :param pick_approach_direction: handz if None
        :param pick_approach_distance:
        :param pick_depart_direction: handz if None
        :param pick_depart_distance:
        :param place_approach_direction_list:
        :param place_approach_distance_list:
        :param place_depart_direction_list:
        :param place_depart_distance_list:
        :param place_depart_jaw_width:
        :param linear_granularity:
        :param use_rrt:
        :param obstacle_list:
        :param reason_grasps: examine grasps sequentially in case of False
        :return:
        author: weiwei
        date: 20191122, 20200105, 20240317
        """
        ## picking parameters
        if pick_approach_jaw_width is None:
            pick_approach_jaw_width = self.robot.end_effector.jaw_range[1]
        if pick_approach_distance is None:
            pick_approach_distance = .07
        if pick_depart_distance is None:
            pick_depart_distance = .07
        ## approach depart parameters
        if place_depart_jaw_width is None:
            place_depart_jaw_width = self.robot.end_effector.jaw_range[1]
        if place_approach_direction_list is None:
            place_approach_direction_list = [-rm.const.z_ax] * len(goal_pose_list)
        if place_approach_distance_list is None:
            place_approach_distance_list = [.07] * len(goal_pose_list)
        if place_depart_direction_list is None:
            place_depart_direction_list = [rm.const.z_ax] * len(goal_pose_list)
        if place_depart_distance_list is None:
            place_depart_distance_list = [.07] * len(goal_pose_list)
        if reason_grasps:
            common_gid_list = self.reason_common_gids(grasp_collection=grasp_collection,
                                                      goal_pose_list=[obj_cmodel.pose] + goal_pose_list,
                                                      obstacle_list=obstacle_list,
                                                      toggle_dbg=False)
        elif common_gid_list_ref is not None:
            common_gid_list = common_gid_list_ref

        if len(common_gid_list) == 0:
            print("No common grasp id at the given goal poses!")
            return None

        for gid in common_gid_list:
            obj_cmodel_copy = obj_cmodel.copy()
            pm_mot = self.gen_pick_and_moveto(obj_cmodel=obj_cmodel_copy,
                                              grasp=grasp_collection[gid],
                                              moveto_pose_list=goal_pose_list,
                                              moveto_approach_direction_list=place_approach_direction_list,
                                              moveto_approach_distance_list=place_approach_distance_list,
                                              moveto_depart_direction_list=place_depart_direction_list,
                                              moveto_depart_distance_list=place_depart_distance_list[:-1] + [0],
                                              start_jnt_values=start_jnt_values,
                                              pick_approach_jaw_width=pick_approach_jaw_width,
                                              pick_approach_direction=pick_approach_direction,
                                              pick_approach_distance=pick_approach_distance,
                                              pick_depart_direction=pick_depart_direction,
                                              pick_depart_distance=pick_depart_distance,
                                              linear_granularity=linear_granularity,
                                              obstacle_list=obstacle_list,
                                              use_rrt=use_rrt,
                                              toggle_dbg=toggle_dbg)
            if pm_mot is None:
                print("Cannot generate the pick and moveto motion!")
                continue
            # place
            last_goal_pos = goal_pose_list[-1][0]
            last_goal_rotmat = goal_pose_list[-1][1]
            obj_cmodel_copy.pose = (last_goal_pos, last_goal_rotmat)
            dep_mot = self.gen_depart_from_given_conf(start_jnt_values=pm_mot.jv_list[-1],
                                                      end_jnt_values=end_jnt_values,
                                                      linear_direction=place_depart_direction_list[-1],
                                                      linear_distance=place_depart_distance_list[-1],
                                                      ee_values=place_depart_jaw_width,
                                                      linear_granularity=linear_granularity,
                                                      obstacle_list=obstacle_list,
                                                      object_list=[obj_cmodel_copy],
                                                      use_rrt=use_rrt,
                                                      toggle_dbg=toggle_dbg)
            if dep_mot is None:
                print("Cannot generate the release motion!")
                continue
            # for robot_mesh in dep_mot.mesh_list:
            #     obj_cmodel_copy.attach_to(robot_mesh)
            return pm_mot + dep_mot
        print("None of the reasoned common grasps are valid.")
        return None


def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def data_saved(anime_data, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(anime_data, f)


def shared_grasp_update(anime_data, task):
    if base.inputmgr.keymap['w']:  # 检测w键
        base.taskMgr.stop()  # 停止任务管理器
        return task.done

    if anime_data.counter >= len(anime_data.predict_common_ids):
        anime_data.counter = 0

    if base.inputmgr.keymap["space"] is True:
        # 清除之前的模型
        anime_data.init_obj_cmodel.detach()
        anime_data.goal_obj_cmodel.detach()
        for model in anime_data.gripper_models:
            model.detach()
        anime_data.gripper_models.clear()

        # 获取当前姿态
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

        # 显示抓取点
        current_common_ids = anime_data.predict_common_ids[anime_data.counter]
        current_common_ids = np.where(current_common_ids==1)[0]
        if current_common_ids is not None:  # 检查是否有可行抓取
            for grasp_id in current_common_ids:
                grasp = anime_data.object_feasible_grasps._grasp_list[grasp_id]

                # 初始位置的抓取
                anime_data.gripper.grip_at_by_pose(
                    jaw_center_pos=init_rotmat @ grasp.ac_pos + init_pos,
                    jaw_center_rotmat=init_rotmat @ grasp.ac_rotmat,
                    jaw_width=grasp.ee_values
                )
                gripper_model = anime_data.gripper.gen_meshmodel(rgb=rm.const.hug_blue, alpha=.2)
                gripper_model.attach_to(base)
                anime_data.gripper_models.append(gripper_model)

                # 目标位置的抓取
                anime_data.gripper.grip_at_by_pose(
                    jaw_center_pos=goal_rotmat @ grasp.ac_pos + goal_pos,
                    jaw_center_rotmat=goal_rotmat @ grasp.ac_rotmat,
                    jaw_width=grasp.ee_values
                )
                gripper_model = anime_data.gripper.gen_meshmodel(rgb=rm.const.hug_blue, alpha=.2)
                gripper_model.attach_to(base)
                anime_data.gripper_models.append(gripper_model)

        anime_data.counter += 1
        time.sleep(0.1)  # 添加延时以避免按键重复触发

    return task.cont


def trajectory_update(anime_data, counter, task):
    if base.inputmgr.keymap['w']:  # 检测w键
        return task.done
    if counter[0] >= len(anime_data):
        counter[0] = 0
    if base.inputmgr.keymap['space']:
        if counter[0] > 0:
            anime_data.mesh_list[counter[0] - 1].detach()
        mesh_model = anime_data.mesh_list[counter[0]]
        mesh_model.attach_to(base)
        counter[0] += 1
    return task.again


def shared_grasp_data_preprocess(object_feasible_grasps, gripper,
                                 model, sample_number, data_loader,
                                 device='cpu', threshold=0.5):
    # predict shared_grasp from model and dataset;
    model.eval()
    all_labels = []
    all_predictions = []
    all_logits = []
    init_poses = []
    goal_poses = []

    with torch.no_grad():
        for index, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            predicted = (outputs > threshold).float()

            # 获取batch中的位置和欧拉角数据
            init_pos = inputs[:, :3].cpu().numpy()  # [batch_size, 3]
            init_euler = inputs[:, 3:6].cpu().numpy()  # [batch_size, 3]
            # 将每个样本的位置和欧拉角组合
            init_poses.extend(np.concatenate([init_pos, init_euler], axis=1))

            goal_pos = inputs[:, 11:14].cpu().numpy()  # [batch_size, 3]
            goal_euler = inputs[:, 14:17].cpu().numpy()  # [batch_size, 3]
            goal_poses.extend(np.concatenate([goal_pos, goal_euler], axis=1))

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())
            if index >= sample_number:
                break

    all_labels = np.vstack(all_labels)
    all_predictions = np.vstack(all_predictions)
    all_logits = np.vstack(all_logits)

    return SharedGraspAnimeData(init_poses, goal_poses, all_labels, all_predictions, all_logits,
                                gripper, object_feasible_grasps)

def trajectory_feasibility_identification(pp_planner, shared_grasp_data,
                                          obj_cmodel, robot, grasp_collection):
    common_gid_list_set = shared_grasp_data.predict_common_ids
    connect_feasible_grasp_ids = []
    if common_gid_list_set[0] is None:
        return None

    obj_cmodel_copy = obj_cmodel.copy()
    for common_list_index, common_gid_list in enumerate(common_gid_list_set):

        # init obj pose
        obj_cmodel_copy.pos = shared_grasp_data.init_poses[common_list_index][:3]
        obj_cmodel_copy.rotmat = rm.rotmat_from_euler(*shared_grasp_data.init_poses[common_list_index][3:])

        # only one goal pose
        goal_pose_list = [[shared_grasp_data.goal_poses[common_list_index][:3],
                           rm.rotmat_from_euler(*shared_grasp_data.goal_poses[common_list_index][3:])]]

        # generate pick and place trajectory from init to goal, so many of them.
        for index, common_gid_element in enumerate(common_gid_list):
            mot_data = pp_planner.gen_pick_and_place(obj_cmodel=obj_cmodel_copy,
                                                     start_jnt_values = robot.get_jnt_values(),
                                                     end_jnt_values=robot.get_jnt_values(),
                                                     grasp_collection=grasp_collection,
                                                     goal_pose_list=goal_pose_list,
                                                     place_approach_direction_list=[rm.np.array([0, 0, -1])],
                                                     place_approach_distance_list=[.07],
                                                     place_depart_direction_list=[rm.np.array([0, 0, 1])],
                                                     place_depart_distance_list=[.07],
                                                     pick_approach_direction=None,
                                                     pick_approach_distance=.07,
                                                     pick_depart_direction=None,
                                                     pick_depart_distance=.07,
                                                     linear_granularity=.02,
                                                     obstacle_list=[],
                                                     use_rrt=True,
                                                     reason_grasps=False,
                                                     common_gid_list_ref=[common_gid_element])

            if mot_data is not None:
                connect_feasible_grasp_ids.append(common_gid_element)

    return connect_feasible_grasp_ids

def trajectory_data_preprocess(pp_planner, shared_grasp_data,
                               obj_cmodel, robot, grasp_collection):
    common_gid_list_set = shared_grasp_data.predict_common_ids
    mot_data_dict = {}
    num = 0
    obj_cmodel_copy = obj_cmodel.copy()
    for common_list_index, common_gid_list in enumerate(common_gid_list_set):
        common_gid = np.where(common_gid_list == 1)[0].astype(int).tolist()

        # init obj pose
        obj_cmodel_copy.pos = shared_grasp_data.init_poses[common_list_index][:3]
        obj_cmodel_copy.rotmat = rm.rotmat_from_euler(*shared_grasp_data.init_poses[common_list_index][3:])

        # only one goal pose
        goal_pose_list = [[shared_grasp_data.goal_poses[common_list_index][:3],
                           rm.rotmat_from_euler(*shared_grasp_data.goal_poses[common_list_index][3:])]]

        for index, common_gid_element in enumerate(common_gid):
            # generate pick and place trajectory from init to goal, so many of them.
            mot_data = pp_planner.gen_pick_and_place(obj_cmodel=obj_cmodel_copy,
                                                     start_jnt_values = robot.get_jnt_values(),
                                                     end_jnt_values=robot.get_jnt_values(),
                                                     grasp_collection=grasp_collection,
                                                     goal_pose_list=goal_pose_list,
                                                     place_approach_direction_list=[rm.np.array([0, 0, -1])],
                                                     place_approach_distance_list=[.07],
                                                     place_depart_direction_list=[rm.np.array([0, 0, 1])],
                                                     place_depart_distance_list=[.07],
                                                     pick_approach_direction=None,
                                                     pick_approach_distance=.07,
                                                     pick_depart_direction=None,
                                                     pick_depart_distance=.07,
                                                     linear_granularity=.02,
                                                     obstacle_list=[],
                                                     use_rrt=True,
                                                     common_gid_list_ref=[common_gid_element])

            if mot_data is not None:
                mot_data_dict[f"Configuration_{common_list_index}_Trajectory_{common_gid_element}"] = mot_data
                num = num + 1
                if num == 3:
                    print("finished collect 3 feasible trajectories")
                    break

    return TrajectoryAnimeData(mot_data_dict)



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='抓取网络训练参数')
    # 数据相关参数
    parser.add_argument('--data_path', type=str,
                        default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\grasp_random_position_bottle_robot_table_WithOnePairStablelabel_109.pickle',
                        help='训练数据路径')
    parser.add_argument('--model_save_path', type=str,
                        default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\shared_best_model\shared_grasp_mlp_bottle_grasp_random_position_bottle_robot_table_WithOnePairStablelabel_109_normalize_data_without_standardization.pth',
                        help='模型保存路径')
    parser.add_argument('--grasp_data_path', type=str,
                        default=r'H:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle',
                        help='抓取数据路径')
    # 模型相关参数
    parser.add_argument('--input_dim', type=int, default=22, help='输入维度')
    parser.add_argument('--output_dim', type=int, default=109, help='输出维度')
    # 添加网络架构选择参数
    parser.add_argument('--network_type', type=str, default='mlp',
                        choices=['mlp', 'mlp_twoencoder', 'resnet', 'resnet_attention'],
                        help='选择网络架构类型')
    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='regrasp', help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str,
                        default='shared_grasp_mlp_bottle_+xy+xyt+same_pos_1.0_57',
                        help='wandb运行名称')
    # 添加数据标准化控制参数
    parser.add_argument('--normalize_data', type=bool, default=False,
                        help='是否对输入数据进行标准化')
    parser.add_argument('--stable_label', type=bool, default=False,
                        help='是否增加稳定标签到输入里面')
    return parser.parse_args()


if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    # 解析参数
    args = parse_args()

    # robot configuration
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([90, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    robot.goto_given_conf(jnt_values=init_jnv)
    # robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    obj_cmodel = mcm.CollisionModel(name='init_obj',
                                    initor=r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    # obj_cmodel.pos = rm.zeros(3)
    # obj_cmodel.rotmat = rm.eye(3)

    # import model , gripper, grasp_collection adn dataset
    model = GraspingNetwork(input_dim=args.input_dim, output_dim=args.output_dim)
    model.load_state_dict(torch.load(args.model_save_path))

    gripper = robot.end_effector
    grasp_collection = gg.GraspCollection.load_from_disk(file_name=args.grasp_data_path)
    test_dataset = GraspingDataset(args.output_dim, args.data_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 推理planner
    planner = PickPlacePlannerFromModel(robot)

    # 测试模型
    sample_number = 0
    data_loader = test_loader
    pp_planner = planner


    shared_grasp_data = shared_grasp_data_preprocess(
        grasp_collection, gripper, model, sample_number, data_loader, device='cpu'
    )
    tic = time.time()
    trajectory_data = trajectory_data_preprocess(
        pp_planner, shared_grasp_data, obj_cmodel, robot, grasp_collection
    )
    toc = time.time()
    print(f"Time cost: {toc - tic:.2f}s")
    base.run()
    anime_data = SharedGraspTrajectoryAnimeData(shared_grasp_data, trajectory_data)

    print("\nSelect display type:")
    print("[1] Show shared grasps")
    print("[2] Show trajectories")
    print("[3] Exit")

    choice = int(input("Enter choice (1-3): "))

    if choice == 1:
        n_grasps = len(anime_data.shared_grasp_data.init_poses)
        print(f"\nAvailable shared grasps: {n_grasps}")
        grasp_idx = int(input(f"Enter grasp index (0-{n_grasps - 1}): "))

        if 0 <= grasp_idx < n_grasps:
            taskMgr.doMethodLater(0.01, shared_grasp_update, "update",
                                  extraArgs=[anime_data.shared_grasp_data],
                                  appendTask=True)
            base.run()
        else:
            print("Error: Invalid index")

    elif choice == 2:
        trajectories = list(anime_data.trajectory_data.mot_data_dict.keys())
        print("\nAvailable trajectories:")
        for i, traj in enumerate(trajectories):
            print(
                f"[{i}] {traj} state: {'Valid' if anime_data.trajectory_data.mot_data_dict[traj] is not None else 'Invalid'}")

        traj_idx = int(input(f"Enter trajectory index (0-{len(trajectories) - 1}): "))
        counter = [0]
        
        if 0 <= traj_idx < len(trajectories):
            extraArgs = anime_data.trajectory_data.mot_data_dict[trajectories[traj_idx]]
            taskMgr.doMethodLater(0.01, trajectory_update, "update", 
                                 extraArgs=[extraArgs, counter], appendTask=True)
            base.run()
        else:
            print("Error: Invalid index")

    elif choice == 3:
        print("Exiting program")

    else:
        print("Error: Please enter 1-3")
