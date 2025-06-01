import numpy as np
import cv2
from cv2 import aruco
import glob
from scipy.spatial.transform import Rotation as R
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(root_dir)
import wrs.basis.robot_math as rm
import pyrealsense2 as rs
from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApiDashboard
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper import nova2_gripper_v3


# 全局变量
camera_matrix = None
dist_coeffs = None

def init_camera_params(device_serial=None):
    """
    初始化相机参数，从RealSense相机直接读取内参
    :param device_serial: 相机序列号，如果有多个相机时用于指定特定相机
    :return: camera_matrix, dist_coeffs
    """
    global camera_matrix, dist_coeffs
    
    # 初始化RealSense管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 如果指定了设备序列号，则启用特定设备
    if device_serial is not None:
        config.enable_device(device_serial)
        
    # 配置彩色图像流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 启动管道
    profile = pipeline.start(config)
    
    # 获取一帧，以便读取内参
    color_frame = pipeline.wait_for_frames().get_color_frame()
    color_intr = color_frame.profile.as_video_stream_profile().intrinsics
    
    # 构建相机内参矩阵
    camera_matrix = np.array([
        [color_intr.fx, 0, color_intr.ppx],
        [0, color_intr.fy, color_intr.ppy],
        [0, 0, 1]
    ])
    
    # 获取畸变系数
    dist_coeffs = np.array(color_intr.coeffs)
    
    # 停止管道
    pipeline.stop()
    
    print("相机参数初始化成功:")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:", dist_coeffs)
    
    return camera_matrix, dist_coeffs


def detect_markers_pose(color_img, marker_size=0.033):
    """
    检测ArUco markers并估计其位姿
    :param color_img: 输入的彩色图像
    :param marker_size: marker的实际尺寸（米）
    :return: 字典，key为marker ID，value为相机到marker的变换矩阵
    """
    global camera_matrix, dist_coeffs
    
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError("相机参数未初始化！请先调用init_camera_params()")
    
    # 初始化 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # 检测markers
    corners, ids, rejected = detector.detectMarkers(color_img)
    
    marker_poses = {}
    if ids is not None:
        # 估计每个marker的位姿
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs)
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            # 转换为变换矩阵
            R_cam_marker = cv2.Rodrigues(rvecs[i])[0]
            T_cam_marker = np.eye(4)
            T_cam_marker[:3, :3] = R_cam_marker
            T_cam_marker[:3, 3] = tvecs[i][0]
            marker_poses[marker_id] = T_cam_marker
            
    return marker_poses


def append_save(robot_jnv, marker_poses, save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/eye_on_hand_calibration_data"):
    """
    将标定数据追加保存到文件中
    :param robot_jnv: 机器人关节角度
    :param marker_poses: 检测到的marker位姿字典
    :param save_dir: 保存数据的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, 'calibration_data.npy')

    # 如果文件已存在，先加载现有数据
    data_list = []
    if os.path.exists(save_file):
        try:
            data_list = np.load(save_file, allow_pickle=True).tolist()
            if not isinstance(data_list, list):
                data_list = []
        except:
            data_list = []

    # 添加新数据
    new_data = {
        'robot_jnv': robot_jnv,
        'marker_poses': marker_poses
    }
    data_list.append(new_data)

    # 保存所有数据
    np.save(save_file, data_list)
    print(f"数据已追加保存，当前共有 {len(data_list)} 组数据")


def collect_calibration_data(robot, num_poses=10, save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/eye_on_hand_calibration_data"):
    """
    采集标定数据，通过按空格键记录当前位姿
    :param robot: 机器人控制器实例
    :param num_poses: 需要采集的位姿数量
    :param save_dir: 保存数据的目录
    :return: robot_poses, camera_poses
    """
    # 初始化相机
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(config)
    

    # 初始化 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    robot_jnv_list = []
    camera_poses = []
    
    print(f"请移动机器人到不同位置，按空格键记录数据（目标：{num_poses}组）")
    print("按ESC键退出")
    
    try:
        while len(robot_jnv_list) < num_poses:
            # 获取图像
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检测markers
            corners, ids, rejected = detector.detectMarkers(color_image)
            
            # 绘制检测到的markers
            display_image = color_image.copy()
            if ids is not None:
                # 绘制marker边框和ID
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                
                # 估计每个marker的姿态
                if camera_matrix is not None and dist_coeffs is not None:
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, 0.033, camera_matrix, dist_coeffs)
                    
                    # 为每个marker绘制坐标轴
                    for i in range(len(ids)):
                        cv2.drawFrameAxes(display_image, camera_matrix, dist_coeffs, 
                                        rvecs[i], tvecs[i], 0.03)
            
            # 在图像上显示检测到的marker数量
            detected_count = len(ids) if ids is not None else 0
            cv2.putText(display_image, f"Detected markers: {detected_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_image, f"Collected: {len(robot_jnv_list)}/{num_poses}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Calibration', display_image)
            key = cv2.waitKey(1)
            
            # 按空格键记录数据
            if key == 32:  # 32是空格键的ASCII码
                marker_poses = detect_markers_pose(color_image, marker_size=0.033)
                if marker_poses:
                    current_robot_jnv = robot.GetAngle()
                    print("robot_jnv:", current_robot_jnv)
                    robot_jnv_list.append(current_robot_jnv)
                    camera_poses.append(marker_poses)
                    # 保存数据
                    append_save(current_robot_jnv, marker_poses, save_dir)
                    print(f"成功记录第 {len(robot_jnv_list)}/{num_poses} 组数据")
                else:
                    print("未检测到markers，请调整位置后重试")
            
            # 按ESC键退出
            elif key == 27:
                break
                
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        
    if len(robot_jnv_list) > 0:
        print(f"共采集到 {len(robot_jnv_list)} 组数据")
    else:
        print("未采集到数据")
        
    return robot_jnv_list, camera_poses


def load_calibration_data(save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/eye_on_hand_calibration_data"):
    """
    从保存的文件中加载标定数据
    :param save_dir: 数据保存的目录
    :return: robot_poses, camera_poses
    """
    robot_jnv_list = []
    camera_poses = []
    
    save_file = os.path.join(save_dir, 'calibration_data.npy')
    if os.path.exists(save_file):
        data_list = np.load(save_file, allow_pickle=True)
        for data in data_list:
            robot_jnv_list.append(data['robot_jnv'])
            camera_poses.append(data['marker_poses'])
    
    print(f"已加载 {len(robot_jnv_list)} 组标定数据")
    return robot_jnv_list, camera_poses


def estimate_camera_to_ee(robot_s, robot_poses, camera_poses):
    """
    计算相机到机器人末端的变换 (T_ee_cam)
    :param robot_s: 机器人运动学模型
    :param robot_poses: 机器人关节角度列表
    :param camera_poses: marker位姿列表
    :return: T_ee_cam
    """
    R_base_ee = []    # 机器人基座到末端的旋转矩阵列表
    t_base_ee = []    # 机器人基座到末端的平移向量列表
    R_cam_marker = [] # 相机到marker的旋转矩阵列表
    t_cam_marker = [] # 相机到marker的平移向量列表
    
    # 选择一个固定的marker ID用于标定
    target_marker_id = None
    
    # 找到所有数据中都存在的marker ID
    common_marker_ids = set()
    for marker_poses in camera_poses:
        if not common_marker_ids:
            common_marker_ids = set(marker_poses.keys())
        else:
            common_marker_ids &= set(marker_poses.keys())
    
    if not common_marker_ids:
        raise ValueError("没有在所有位姿中都检测到的marker，无法进行标定")
    
    # 选择第一个公共marker作为标定目标
    target_marker_id = list(common_marker_ids)[0]
    print(f"使用marker ID {target_marker_id}进行标定")
    
    for jnt_str, marker_poses in zip(robot_poses, camera_poses):
        if target_marker_id not in marker_poses:
            continue
            
        # 解析关节角度
        start_idx = jnt_str.find('{') + 1
        end_idx = jnt_str.find('}')
        angles_str = jnt_str[start_idx:end_idx]
        jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
        
        # 计算机器人末端在基座下的位姿
        current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
        T_base_ee_current = rm.homomat_from_posrot(current_pos, current_rotmat)
        
        # 获取当前相机到marker的变换
        T_cam_marker_current = marker_poses[target_marker_id]
        
        # 添加到标定数据中
        R_base_ee.append(T_base_ee_current[:3, :3])
        t_base_ee.append(T_base_ee_current[:3, 3].reshape(3,1))
        R_cam_marker.append(T_cam_marker_current[:3, :3])
        t_cam_marker.append(T_cam_marker_current[:3, 3].reshape(3,1))
    
    print(f"\nUsing {len(R_base_ee)} poses for calibration")
    
    # 执行手眼标定 (eye-on-hand)
    R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_ee,
        t_gripper2base=t_base_ee,
        R_target2cam=R_cam_marker,
        t_target2cam=t_cam_marker,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 构建最终的变换矩阵
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R_ee_cam
    T_ee_cam[:3, 3] = t_ee_cam.squeeze()
    
    return T_ee_cam


if __name__ == "__main__":
    import time

    robot_x = DobotApiDashboard("192.168.5.100", 29999)
    robot_x.EnableRobot(load=1.0, centerX=0.0, centerY=0.0, centerZ=0.0)
    time.sleep(1)
    print("开始采集标定数据...")
    
    # 初始化相机参数
    camera_matrix, dist_coeffs = init_camera_params()
    robot_poses, camera_poses = collect_calibration_data(robot_x, num_poses=10)
    robot_x.DisableRobot()

    robot_s = nova2_gripper_v3()
    robot_poses, camera_poses = load_calibration_data()
    T_ee_cam = estimate_camera_to_ee(robot_s, robot_poses, camera_poses)
    print("\nHand-Eye Calibration Result (T_ee_cam):\n", T_ee_cam)
    
    # 验证标定结果
    print("\nValidation:")
    # 选择一组数据进行验证
    test_idx = 0
    if len(robot_poses) > test_idx:
        # 解析关节角度
        jnt_str = robot_poses[test_idx]
        start_idx = jnt_str.find('{') + 1
        end_idx = jnt_str.find('}')
        angles_str = jnt_str[start_idx:end_idx]
        jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
        
        # 计算机器人末端在基座下的位姿
        current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
        T_base_ee = rm.homomat_from_posrot(current_pos, current_rotmat)
        
        # 计算相机在基座下的位姿
        T_base_cam = T_base_ee @ T_ee_cam
        
        print(f"Robot end-effector position: {T_base_ee[:3, 3]}")
        print(f"Camera position: {T_base_cam[:3, 3]}")
        
        # 验证标定结果
        test_marker_poses = camera_poses[test_idx]
        for marker_id, T_cam_marker in test_marker_poses.items():
            # 通过标定结果计算marker在基座坐标系下的位置
            T_base_marker = T_base_cam @ T_cam_marker
            print(f"\nMarker {marker_id} position in base frame: {T_base_marker[:3, 3]}") 