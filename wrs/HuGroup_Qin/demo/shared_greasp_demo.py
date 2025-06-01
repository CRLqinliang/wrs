import sys
import time
import numpy as np
import cv2
from cv2 import aruco

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(root_dir)

from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400, find_devices
from wrs import rm, wd, mcm, gg, mgm, rrtc, adp
from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApiDashboard
from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApiMove
from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApi
from wrs.HuGroup_Qin.robot_con.dynamixelGripper_driver import DynamixelGripper

from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import SharedGraspAnimeData
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import PickPlacePlannerFromModel
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import trajectory_data_preprocess
# from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import trajectory_update
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import shared_grasp_update
from wrs.HuGroup_Qin.Shared_grasp_project.util.anime_show_shared_grasp_trajectory import SharedGraspTrajectoryAnimeData
from wrs.manipulation.placement.flatsurface import FSReferencePoses
from wrs.HuGroup_Qin.Shared_grasp_project.script.shared_grasp_network.binary_ebm_grasp import GraspEnergyNetwork
import torch

try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

# 相机位姿 
T_base_cam = np.array([
    [-0.03754594,  0.89211193, -0.45025171,  0.23938792],
    [ 0.99612438, -0.00245088, -0.08792168,  0.46887631],
    [-0.07953949, -0.45180781, -0.88856242,  0.5423873 ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# 定义每个marker相对于物体坐标系的位姿变换矩阵
MARKER_POSES = {
    0: np.array([  # marker-0 的位姿：z轴与世界y同向，y轴与世界z同向
        [ -1, 0,  0, 0],       # x轴指向-y
        [ 0,  0,  1, 0.0375],  # y轴指向z
        [ 0,  1,  0, 0.065],   # z轴指向-x
        [ 0,  0,  0, 1]
    ]),
    1: np.array([  # marker-1的位姿：z轴与世界x反向，y轴与世界z同向
        [ 0,  0, -1, -0.0375], # z轴指向-x
        [ -1,  0,  0, 0],       # y轴指向z
        [ 0,  1,  0, 0.065],   # x轴指向y
        [ 0,  0,  0, 1]
    ]),
    2: np.array([  # marker-2的位姿：z轴与世界x反向，y轴与世界z同向
        [ 0,  0,  1, 0.0375],  # z轴指向x
        [ 1,  0,  0, 0],       # y轴指向z
        [ 0,  1,  0, 0.065],   # x轴指向y
        [ 0,  0,  0, 1] 
    ])
}

INIT_JNV = np.array([-92.02, -16.9479, 88.03, 18.9558, -89.9942, -1.9834]) #degree


def estimate_object_pose_from_markers(marker_poses_camera):
    """
    从检测到的marker位姿估计物体位姿
    :param marker_poses_camera: 字典，包含每个检测到的marker在相机坐标系下的位姿
    :return: 物体在相机坐标系下的位姿
    """
    all_object_poses = []
    all_weights = []
    
    for marker_id, marker_pose_camera in marker_poses_camera.items():
        if marker_id not in MARKER_POSES:
            continue
            
        # 获取marker相对于物体的已知位姿
        marker_pose_object = MARKER_POSES[marker_id]
        
        # 计算物体位姿 = marker在相机下的位姿 * marker相对于物体位姿的逆
        marker_pose_object_inv = np.linalg.inv(marker_pose_object)
        object_pose = marker_pose_camera @ marker_pose_object_inv
        
        # 根据marker在图像中的可见性或其他指标设置权重
        weight = 1.0  # 可以基于marker的检测置信度调整
        
        all_object_poses.append(object_pose)
        all_weights.append(weight)
    
    if not all_object_poses:
        return None
        
    # 如果检测到多个marker，进行加权平均
    if len(all_object_poses) > 1:
        # 将权重归一化
        weights = np.array(all_weights) / np.sum(all_weights)
        
        # 位置直接加权平均
        final_position = np.zeros(3)
        for pose, w in zip(all_object_poses, weights):
            final_position += pose[:3, 3] * w
            
        # 旋转矩阵的平均（使用SVD方法）
        R_sum = np.zeros((3, 3))
        for pose, w in zip(all_object_poses, weights):
            R = pose[:3, :3]
            R_sum += w * R
            
        # 使用SVD确保结果是有效的旋转矩阵
        U, _, Vt = np.linalg.svd(R_sum)
        final_rotation = U @ Vt
        
        # 确保是右手坐标系
        if np.linalg.det(final_rotation) < 0:
            Vt[2, :] *= -1
            final_rotation = U @ Vt
        
        # 组合最终的位姿矩阵
        final_pose = np.eye(4)
        final_pose[:3, :3] = final_rotation
        final_pose[:3, 3] = final_position
        
        return final_pose
    else:
        # 如果只检测到一个marker，直接返回计算的位姿
        return all_object_poses[0]


def main():
    # 初始化Panda3d世界
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    
    # 创建物体模型
    robot = robot_setup()
    obj_cmodel = obj_setup(name='bottle', pos=[0, 0, 0], 
                          rotmat=np.eye(3), alpha=0.5)

    # 显示相机坐标系
    mgm.gen_frame(pos=T_base_cam[:3, 3], 
                 rotmat=T_base_cam[:3, :3], 
                 ax_length=0.1).attach_to(base)

    # 查找并初始化RealSense相机
    serials, _ = find_devices()
    if not serials:
        print("未找到RealSense相机!")
        return
    
    # 初始化第一个相机
    rs_camera = RealSenseD400(device=serials[0])
    print(f"成功连接相机 {serials[0]}")

    # 初始化 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    try:
        while True:
            # 获取彩色图像
            color_img = rs_camera.get_color_img()
            
            # 检测AR码
            corners, ids, rejected = detector.detectMarkers(color_img)
            
            # 如果检测到AR码
            if ids is not None:
                # 绘制检测到的标记
                cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
                
                # 存储每个检测到的marker的位姿
                marker_poses_camera = {}
                
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    # 估计单个marker的位姿
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], 0.052,  # marker尺寸为5.2cm
                        rs_camera.intr_mat, 
                        rs_camera.intr_distcoeffs
                    )
                    
                    # 转换为变换矩阵
                    rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                    marker_pose = np.eye(4)
                    marker_pose[:3, :3] = rotation_matrix
                    marker_pose[:3, 3] = tvec[0][0]
                    
                    marker_poses_camera[marker_id] = marker_pose
                    
                    # 绘制marker坐标系
                    cv2.drawFrameAxes(color_img, rs_camera.intr_mat, 
                                    rs_camera.intr_distcoeffs, 
                                    rvec, tvec, 0.02)
                
                # 估计物体位姿
                object_pose_camera = estimate_object_pose_from_markers(marker_poses_camera)
                
                if object_pose_camera is not None:
                    # 将相机坐标系下的物体位姿转换到基坐标系
                    object_pose_base = T_base_cam @ object_pose_camera

                    # 更新Panda3d中物体的位置和姿态
                    obj_cmodel.detach()
                    obj_cmodel.pos = object_pose_base[:3, 3]
                    obj_cmodel.rotmat = object_pose_base[:3, :3]
                    obj_cmodel.attach_to(base)

                    # 打印位姿信息
                    print("\nObject Pose in Base Frame:")
                    print(f"Position (x,y,z): {obj_cmodel.pos}")
                    euler_angles = cv2.RQDecomp3x3(obj_cmodel.rotmat)[0]
                    print(f"Rotation (rx,ry,rz): {euler_angles}")
            
            # 显示图像
            cv2.imshow("RealSense Camera", color_img)
            
            # 更新Panda3d场景 - 使用taskMgr.step()
            taskMgr.step()
            
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        rs_camera.stop()
        cv2.destroyAllWindows()
        base.destroy()


def obj_setup(name, pos, rotmat, rgb=None, alpha=None):
    """设置物体"""
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha, 
                                   initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_cmodel.pos = pos
    obj_cmodel.rotmat = rotmat
    obj_cmodel.attach_to(base)
    # obj_cmodel.show_cdprim()
    obj_cmodel.show_local_frame()
    return obj_cmodel


def robot_setup():
    robot = nova2_gripper_v3(enable_cc=True)
    init_jnv = np.array([-92.02, -16.9479, 88.03, 18.9558, -89.9942, -1.9834]) * np.pi/180
    robot.manipulator.home_conf = init_jnv
    robot.goto_home_conf()
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    return robot


def get_stable_poses():
    obj_cmodel = mcm.CollisionModel(name="bottle", initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
    stable_poses = dict()
    for i, item in enumerate(fs_reference_poses._poses):
        stable_poses[i] = item
    return stable_poses


# 计算每个姿态的stable label
def get_stable_label(pose_array, stable_poses=None):
    """
    基于物体坐标系与世界坐标系的对齐关系确定stable label
    
    Args:
        pose_array: 包含位置和四元数的数组 [x,y,z, qx,qy,qz,qw]
        stable_poses: 未使用，保持参数兼容性
    
    Returns:
        int: stable label (0-4)
    """
    # 将四元数转换为旋转矩阵
    rotmat = rm.quaternion_to_rotmat(pose_array[3:])
    
    # 获取物体坐标系的三个轴向量
    obj_x = rotmat[:, 0]  # 物体的x轴在世界坐标系下的方向
    obj_y = rotmat[:, 1]  # 物体的y轴在世界坐标系下的方向
    obj_z = rotmat[:, 2]  # 物体的z轴在世界坐标系下的方向
    
    # 世界坐标系的轴向量
    world_z = np.array([0, 0, 1])
    
    # 容错阈值（角度）
    angle_threshold = np.cos(np.radians(10))  # 允许10度的误差
    
    # 计算向量夹角的余弦值
    def cos_angle(v1, v2):
        return np.abs(np.dot(v1, v2))
    
    # 判断两个向量是否平行（考虑正反方向）
    def is_parallel(v1, v2, threshold=angle_threshold):
        return cos_angle(v1, v2) > threshold
    
    # 判断两个向量是否垂直
    def is_perpendicular(v1, v2, threshold=0.1):  # 允许约5.7度的误差
        return abs(np.dot(v1, v2)) < threshold
    
    # 判断平面是否平行于xy平面
    def is_plane_parallel_to_xy(normal_vec):
        return is_parallel(np.abs(normal_vec), world_z)
    
    # Case 1: 物体z轴与世界z轴平行
    if is_parallel(obj_z, world_z):
        return 4
    
    # Case 2 & 3: 物体yz平面平行于世界xy平面 (物体x轴垂直于地面)
    if is_plane_parallel_to_xy(obj_x):
        # 检查x轴与世界z轴的关系
        if np.dot(obj_x, world_z) > 0:  # x轴向上
            return 2
        else:  # x轴向下
            return 3
    
    # Case 4 & 5: 物体xz平面平行于世界xy平面 (物体y轴垂直于地面)
    if is_plane_parallel_to_xy(obj_y):
        # 检查y轴与世界z轴的关系
        if np.dot(obj_y, world_z) > 0:  # y轴向上
            return 1
        else:  # y轴向下
            return 0
            
    # 如果没有匹配到任何稳定姿态，返回最接近的姿态
    # 计算与各个方向的对齐程度
    alignments = [
        (0, cos_angle(obj_y, -world_z)),  # y轴向下
        (1, cos_angle(obj_y, world_z)),   # y轴向上
        (2, cos_angle(obj_x, world_z)),   # x轴向上
        (3, cos_angle(obj_x, -world_z)),  # x轴向下
        (4, cos_angle(obj_z, world_z)),   # z轴向上
    ]
    
    # 返回对齐程度最高的姿态
    return max(alignments, key=lambda x: x[1])[0]


def execute_trajectory_on_robot(mot_data):
    """
    将轨迹数据发送到实际机器人执行
    """
    # 连接到机器人
    def ConnectRobot():
        try:
            ip = "192.168.5.100"  # Robot IP
            dashboardPort = 29999
            movePort = 30003
            feedPort = 30004
            print("正在建立连接...")
            dashboard = DobotApiDashboard(ip, dashboardPort)
            move = DobotApiMove(ip, movePort)
            feed = DobotApi(ip, feedPort)
            print(">.<连接成功>!<")
            return dashboard, move, feed
        except Exception as e:
            print(":(连接失败:(")
            raise e

    # init robot & gripper
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()
    dashboard.SpeedFactor(30)
    gripper_x = DynamixelGripper(device='COM3', motor_ids=[1], baudrate=57600,
                                  op_mode=5, gripper_limit=(0, 0.12),
                                  motor_max_current=170)

    move.JointMovJ(INIT_JNV[0], INIT_JNV[1], INIT_JNV[2], INIT_JNV[3], INIT_JNV[4], INIT_JNV[5], 30, 10)
    gripper_x.open_gripper()
    time.sleep(0.2)

    for jnt_values, ev_values in zip(mot_data._jv_list, mot_data._ev_list):
        jnt_value = jnt_values * 180 / np.pi
        move.JointMovJ(jnt_value[0], jnt_value[1], jnt_value[2],
                        jnt_value[3], jnt_value[4], jnt_value[5], 30, 10)
        time.sleep(1)
        if ev_values < 0.1:
            gripper_x.close_gripper()
        else:
            gripper_x.open_gripper()
        #gripper_x.set_gripper_width(width=ev_values-0.02, grasp_force=50)

    print("Trajectory execution completed")
    return


def trajectory_update(anime_data, counter, task):
    if base.inputmgr.keymap['w']:  # 检测w键
        base.taskMgr.stop()  # 停止任务管理器
        return task.done
    
    if counter[0] >= len(anime_data):
        counter[0] = 0
        # base.taskMgr.stop()  # 轨迹播放完成后停止
        # return task.done
        
    if base.inputmgr.keymap['space']:
        if counter[0] > 0:
            anime_data.mesh_list[counter[0] - 1].detach()
        mesh_model = anime_data.mesh_list[counter[0]]
        mesh_model.attach_to(base)
        counter[0] += 1
    return task.again


def demo():
    # 初始化Panda3d世界
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # 创建物体模型
    robot = robot_setup()
    init_obj_cmodel = mcm.CollisionModel(name="init_obj", 
                                       initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    goal_obj_cmodel = mcm.CollisionModel(name="goal_obj", 
                                        initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl",
                                        alpha=0.5)
    # 显示相机坐标系
    mgm.gen_frame(pos=T_base_cam[:3, 3], rotmat=T_base_cam[:3, :3], ax_length=0.1).attach_to(base)

    # 查找并初始化RealSense相机
    serials, _ = find_devices()
    if not serials:
        print("未找到RealSense相机!")
        return
    
    # 初始化第一个相机
    rs_camera = RealSenseD400(device=serials[0])
    print(f"成功连接相机")

    # 初始化 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 初始化状态变量
    state = "DETECT_GOAL"  # 状态: DETECT_GOAL, DETECT_INIT, SHOW_OPTIONS
    goal_pose = None
    init_pose = None
    space_pressed = False  # 添加按键状态标志
    
    # 加载模型和抓取数据
    # model = GraspEnergyNetwork(
    #                     input_dim=31,
    #                     hidden_dims=[256, 512, 256],
    #                     num_res_blocks=2,
    #                     dropout_rate=0.2)
    # checkpoint = torch.load(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\model\Binary_ebm_model\best_model_grasp_ebm_robot_table_withstablelabel_5M_109_h3_2048_lr0.0002_t0.1_dataratio_0.5_trainsplit_0.7.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    grasp_collection = gg.GraspCollection.load_from_disk(file_name=r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle")
    grasp_poses = np.array([np.concatenate([ np.array(grasp.ac_pos, dtype=np.float32).flatten(), rm.rotmat_to_quaternion(grasp.ac_rotmat)]) for grasp in grasp_collection], dtype=np.float32)

    while True:
        # 获取彩色图像
        color_img = rs_camera.get_color_img()
        
        # 检测AR码
        corners, ids, rejected = detector.detectMarkers(color_img)
        
        # 如果检测到AR码
        if ids is not None:
            # 绘制检测到的标记
            # cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
            
            # 存储每个检测到的marker的位姿
            marker_poses_camera = {}
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], 0.052,
                    rs_camera.intr_mat, 
                    rs_camera.intr_distcoeffs
                )
                
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                marker_pose = np.eye(4)
                marker_pose[:3, :3] = rotation_matrix
                marker_pose[:3, 3] = tvec[0][0]
                
                marker_poses_camera[marker_id] = marker_pose
                
                # cv2.drawFrameAxes(color_img, rs_camera.intr_mat,
                #                 rs_camera.intr_distcoeffs,
                #                 rvec, tvec, 0.02)
            
            # 估计物体位姿
            object_pose_camera = estimate_object_pose_from_markers(marker_poses_camera)
            if object_pose_camera is not None:
                # 转换位姿矩阵为旋转向量和平移向量
                obj_rvec, _ = cv2.Rodrigues(object_pose_camera[:3, :3])
                obj_tvec = object_pose_camera[:3, 3]
                
                # 绘制物体坐标系
                cv2.drawFrameAxes(color_img, rs_camera.intr_mat, 
                                rs_camera.intr_distcoeffs, 
                                obj_rvec, obj_tvec, 0.05)  # 0.05是坐标轴长度，可以调整


            if object_pose_camera is not None:
                # 将相机坐标系下的物体位姿转换到基坐标系
                object_pose_base = T_base_cam @ object_pose_camera
                
                if state == "DETECT_GOAL":
                    # 更新目标物体的位置和姿态
                    goal_obj_cmodel.detach()
                    goal_obj_cmodel.pos = object_pose_base[:3, 3]
                    goal_obj_cmodel.rotmat = object_pose_base[:3, :3]
                    goal_obj_cmodel.attach_to(base)
                    goal_obj_cmodel.show_local_frame()

                    # 检测空格键按下和释放
                    if base.inputmgr.keymap['space'] and not space_pressed:
                        goal_pose = (goal_obj_cmodel.pos, goal_obj_cmodel.rotmat)
                        print("目标姿态已确认!")
                        state = "DETECT_INIT"
                        space_pressed = True
                    elif not base.inputmgr.keymap['space'] and space_pressed:
                        space_pressed = False
                        
                elif state == "DETECT_INIT":
                    # 更新初始物体的位置和姿态
                    init_obj_cmodel.detach()
                    init_obj_cmodel.pos = object_pose_base[:3, 3]
                    init_obj_cmodel.rotmat = object_pose_base[:3, :3]
                    init_obj_cmodel.attach_to(base)
                    init_obj_cmodel.show_local_frame()

                    # 检测空格键按下和释放
                    if base.inputmgr.keymap['space'] and not space_pressed:
                        init_pose = (init_obj_cmodel.pos, init_obj_cmodel.rotmat)
                        print("初始姿态已确认!")
                        state = "SHOW_OPTIONS"
                        break
                        space_pressed = True
                    elif not base.inputmgr.keymap['space'] and space_pressed:
                        space_pressed = False
        
        # 显示图像
        cv2.imshow("RealSense Camera", color_img)
        
        # 更新Panda3d场景
        taskMgr.step()

        if cv2.waitKey(1) == 27:
            return
            

    if state == "SHOW_OPTIONS":
        # 准备共享抓取数据 - 转换为位置+四元数格式
        init_quat = rm.rotmat_to_quaternion(init_pose[1])
        goal_quat = rm.rotmat_to_quaternion(goal_pose[1])
        init_pose_array = np.concatenate([init_pose[0], init_quat])  # 7维：位置(3) + 四元数(4)
        goal_pose_array = np.concatenate([goal_pose[0], goal_quat])  # 7维：位置(3) + 四元数(4)
        
        # 获取初始姿态和目标姿态的stable label
        init_stable_label = get_stable_label(init_pose_array)
        goal_stable_label = get_stable_label(goal_pose_array)
        print(f"Initial pose stable label: {init_stable_label}")
        print(f"Goal pose stable label: {goal_stable_label}")
        
        # 创建one-hot编码
        def create_one_hot(label, num_classes=5):
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1
            return one_hot
        
        # 获取one-hot编码的stable labels
        init_stable_one_hot = create_one_hot(init_stable_label)
        goal_stable_one_hot = create_one_hot(goal_stable_label)
        
        # 首先创建一个列表来存储所有输入数据
        input_list = []
        for grasp in grasp_poses:
            # 对每个抓取姿态，创建一个包含所有必要信息的数组
            single_input = np.concatenate([
                init_pose_array,          # 7维：位置(3) + 四元数(4)
                goal_pose_array,          # 7维：位置(3) + 四元数(4)
                init_stable_one_hot,      # 5维：初始姿态的stable label的one-hot编码
                goal_stable_one_hot,      # 5维：目标姿态的stable label的one-hot编码
                grasp,                    # 7维：抓取姿态
            ])
            input_list.append(single_input)
        
        # 将列表转换为numpy数组，然后创建tensor
        input_data = np.array(input_list)
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0)
        model.eval()
        best_threshold = -4.70362 # check training log
        with torch.no_grad():
            output = model(input_tensor)
            predicted = (output < best_threshold).float()
            predicted = predicted.squeeze(0,2).numpy()

        if predicted.sum() == 0:
            print("No feasible grasp found")
            return
        
        # 将四元数转换为欧拉角
        init_rotmat = rm.quaternion_to_rotmat(init_pose_array[3:])
        goal_rotmat = rm.quaternion_to_rotmat(goal_pose_array[3:])
        init_euler = rm.rotmat_to_euler(init_rotmat)

        goal_euler = rm.rotmat_to_euler(goal_rotmat)
        init_pose = [init_pose_array[0], init_pose_array[1], init_pose_array[2],
                      init_euler[0], init_euler[1], init_euler[2]]
        goal_pose = [goal_pose_array[0], goal_pose_array[1], goal_pose_array[2],
                      goal_euler[0], goal_euler[1], goal_euler[2]]


        # 创建共享抓取动画数据
        shared_grasp_data = SharedGraspAnimeData(
            [init_pose], [goal_pose],
            [], [predicted], [],
            robot.end_effector, grasp_collection
        )
        # 创建轨迹规划器
        planner = PickPlacePlannerFromModel(robot)
        
        # 生成轨迹数据
        print("开始生成轨迹数据")
        trajectory_data = trajectory_data_preprocess(planner, shared_grasp_data, init_obj_cmodel.copy(), 
                                                     robot, grasp_collection)
        
        # 创建动画数据
        anime_data = SharedGraspTrajectoryAnimeData(shared_grasp_data, trajectory_data)

        while True:
            # 显示选项菜单
            print("\nSelect display type:")
            print("[1] Show shared grasps")
            print("[2] Show trajectories")
            print("[3] Exit")

            choice = int(input("Enter choice (1-3): "))

            if choice == 1:
                taskMgr.doMethodLater(0.01, shared_grasp_update, "update",
                                    extraArgs=[anime_data.shared_grasp_data],
                                    appendTask=True)
                try:
                    base.run()
                except SystemExit:
                    pass  # 捕获 SystemExit 异常，允许程序继续执行

            elif choice == 2:
                trajectories = list(anime_data.trajectory_data.mot_data_dict.keys())
                print("\nAvailable trajectories:")
                for i, traj in enumerate(trajectories):
                    print(f"[{i}] {traj}")

                traj_idx = int(input(f"Enter trajectory index (0-{len(trajectories)-1}): "))
                counter = [0]

                if 0 <= traj_idx < len(trajectories):
                    mot_data = anime_data.trajectory_data.mot_data_dict[trajectories[traj_idx]]

                    # 显示轨迹动画
                    print("Displaying trajectory animation. Press 'w' to stop or wait for completion.")
                    extraArgs = anime_data.trajectory_data.mot_data_dict[trajectories[traj_idx]]
                    taskMgr.doMethodLater(0.01, trajectory_update, "update",
                                        extraArgs=[extraArgs, counter], appendTask=True)
                    try:
                        base.run()
                    except SystemExit:
                        pass

                    # 动画结束后询问是否执行
                    execute = input("\nExecute this trajectory on the real robot? (y/n): ")
                    if execute.lower() == 'y':
                        execute_trajectory_on_robot(mot_data)

            elif choice == 3:
                print("system shutdown.")
                return


if __name__ == "__main__":

    demo()

    # base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
    # mgm.gen_frame().attach_to(base)
    # obj_setup(name='bottle', pos=[0, 0, 0], rotmat=rm.rotmat_from_euler(0, 0, 0), alpha=0.1)
    
    # # 显示每个marker的坐标系
    # for i in range(3):
    #     # 获取marker的位姿
    #     marker_pose = MARKER_POSES[i]
    #     # 提取位置和旋转矩阵
    #     tgt_pos = marker_pose[:3, 3]
    #     tgt_rotmat = marker_pose[:3, :3]
    #     # 生成并显示坐标系
    #     mgm.gen_frame(ax_length=.1, pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    
    # base.run()

