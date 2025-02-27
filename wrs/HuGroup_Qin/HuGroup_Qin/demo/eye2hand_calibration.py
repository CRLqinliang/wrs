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

# 全局变量定义
# 两个marker相对于机器人末端的变换矩阵
T_ee_marker = {
    0: np.array([  # marker-0的位姿：z轴与tcp x同向，y轴与tcp-z同向
        [1, 0, 0, -0.043],  # z轴指向x
        [0, 0, 1, 0],  # x轴指向y
        [0, -1, 0, 0],  # y轴指向-z
        [0, 0, 0, 1]
    ]),
    4: np.array([  # marker-0的位姿：z轴与tcp x同向，y轴与tcp-z同向
        [1, 0, 0, 0.043],  # z轴指向x
        [0, 0, 1, 0],  # x轴指向y
        [0, -1, 0, 0],  # y轴指向-z
        [0, 0, 0, 1]
    ]),
}

# 在文件开头添加全局变量
camera_matrix = None
dist_coeffs = None

def init_camera_params(device_serial=None):
    """
    初始化相机参数，从RealSense相机直接读取内参
    :param device_serial: 相机序列号，如果有多个相机时用于指定特定相机
    :return: camera_matrix, dist_coeffs
    """
    global camera_matrix, dist_coeffs
    
    try:
        # 初始化RealSense管道
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 如果指定了设备序列号，则启用特定设备
        if device_serial is not None:
            config.enable_device(device_serial)
            
        # 配置彩色图像流
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        
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
        
    except Exception as e:
        print(f"初始化相机参数失败: {str(e)}")
        print("请确保RealSense相机已正确连接")
        return None, None

def detect_markers_pose(color_img, marker_size=0.052):
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

def append_save(robot_jnv, marker_poses, save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/calibration_data"):
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

def collect_calibration_data(robot, num_poses=10, save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/calibration_data"):
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
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
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
                        corners, 0.052, camera_matrix, dist_coeffs)
                    
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
                marker_poses = detect_markers_pose(color_image, marker_size=0.052)
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

def load_calibration_data(save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/calibration_data"):
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

def estimate_camera_to_base(robot_s, robot_poses, camera_poses):
    """
    计算固定相机到机器人基座的变换 (T_base_cam)
    :param robot_s: 机器人运动学模型
    :param robot_poses: 机器人关节角度列表
    :param camera_poses: marker位姿列表
    :return: T_base_cam
    """
    global T_ee_marker
    
    R_base_ee = []    # 机器人基座到末端的旋转矩阵列表
    t_base_ee = []    # 机器人基座到末端的平移向量列表
    R_cam_marker = [] # 相机到marker的旋转矩阵列表
    t_cam_marker = [] # 相机到marker的平移向量列表
    
    for jnt_str, marker_poses in zip(robot_poses, camera_poses):
        # 解析关节角度
        start_idx = jnt_str.find('{') + 1
        end_idx = jnt_str.find('}')
        angles_str = jnt_str[start_idx:end_idx]
        jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
        
        # 计算机器人末端在基座下的位姿
        current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
        T_base_ee_current = rm.homomat_from_posrot(current_pos, current_rotmat)
        
        # 处理每个检测到的marker
        for marker_id, T_cam_marker_current in marker_poses.items():
            if marker_id in T_ee_marker:
                # 计算 T_base_marker = T_base_ee * T_ee_marker
                T_base_marker = T_base_ee_current @ T_ee_marker[marker_id]
                
                # 添加到标定数据中
                R_base_ee.append(T_base_marker[:3, :3])
                t_base_ee.append(T_base_marker[:3, 3].reshape(3,1))
                R_cam_marker.append(T_cam_marker_current[:3, :3])
                t_cam_marker.append(T_cam_marker_current[:3, 3].reshape(3,1))
    
    print(f"\nUsing {len(R_base_ee)} poses for calibration")
    
    # 执行手眼标定
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_ee,
        t_gripper2base=t_base_ee,
        R_target2cam=R_cam_marker,
        t_target2cam=t_cam_marker,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 构建最终的变换矩阵
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = t_base_cam.squeeze()
    
    return T_base_cam

def collect_fixed_marker_data(num_poses=20, save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/calibration_data_fixed"):
    """
    采集固定位置marker的图像数据
    :param num_poses: 需要采集的图像数量
    :param save_dir: 保存数据的目录
    :return: None
    """
    # 初始化相机
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipe.start(config)
    
    # 初始化 ArUco 检测器
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    collected_data = []
    print(f"请从不同角度拍摄固定marker，按空格键记录数据（目标：{num_poses}组）")
    print("按ESC键退出")
    
    try:
        while len(collected_data) < num_poses:
            # 获取图像
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检测markers
            corners, ids, rejected = detector.detectMarkers(color_image)
            marker_poses = detect_markers_pose(color_image, marker_size=0.052)
            
            # 绘制检测到的markers
            display_image = color_image.copy()
            if ids is not None:
                # 绘制marker边框和ID
                cv2.aruco.drawDetectedMarkers(display_image, corners, ids)
                
                # 绘制坐标轴
                if camera_matrix is not None and dist_coeffs is not None:
                    for i in range(len(ids)):
                        marker_id = ids[i][0]
                        if marker_id in marker_poses:
                            # 从marker_poses中获取位姿信息
                            T_cam_marker = marker_poses[marker_id]
                            rvec, _ = cv2.Rodrigues(T_cam_marker[:3, :3])
                            tvec = T_cam_marker[:3, 3]
                            
                            # 绘制坐标轴
                            cv2.drawFrameAxes(display_image, camera_matrix, dist_coeffs, 
                                            rvec, tvec, 0.03)
                            
                            # 显示位置信息
                            pos = tvec
                            cv2.putText(display_image, 
                                      f"ID{marker_id}: {pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}",
                                      (10, 30 + 30 * marker_id), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示采集进度
            cv2.putText(display_image, f"Collected: {len(collected_data)}/{num_poses}",
                       (10, display_image.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示检测到的marker数量
            detected_count = len(ids) if ids is not None else 0
            cv2.putText(display_image, f"Detected markers: {detected_count}",
                       (10, display_image.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Collect Fixed Marker Data', display_image)
            key = cv2.waitKey(1)
            
            # 按空格键记录数据
            if key == 32 and marker_poses:  # 32是空格键的ASCII码
                collected_data.append(marker_poses)
                print(f"成功记录第 {len(collected_data)}/{num_poses} 组数据")
                
                # 确保保存目录存在
                os.makedirs(save_dir, exist_ok=True)
                # 保存数据
                np.save(os.path.join(save_dir, 'fixed_marker_data.npy'), collected_data)
            
            # 按ESC键退出
            elif key == 27:
                break
                
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    
    print(f"共采集到 {len(collected_data)} 组数据")
    return collected_data

def calibrate_camera_from_fixed_markers(marker_poses_list, T_base_marker):
    """
    使用固定位置的marker数据标定相机
    :param marker_poses_list: 采集的marker位姿数据列表
    :param T_base_marker: marker相对于机器人基座的已知变换矩阵
    :return: T_base_cam 相机相对于机器人基座的变换矩阵
    """
    R_base_marker = []    # 基座到marker的旋转矩阵列表
    t_base_marker = []    # 基座到marker的平移向量列表
    R_cam_marker = []     # 相机到marker的旋转矩阵列表
    t_cam_marker = []     # 相机到marker的平移向量列表
    
    for marker_poses in marker_poses_list:
        # 处理每一帧中的marker
        for marker_id, T_cam_marker in marker_poses.items():
            # 添加到标定数据中
            R_base_marker.append(T_base_marker[int(marker_id)][:3, :3])
            t_base_marker.append(T_base_marker[int(marker_id)][:3, 3].reshape(3,1))
            R_cam_marker.append(T_cam_marker[:3, :3])

            t_cam_marker.append(T_cam_marker[:3, 3].reshape(3,1))

    
    print(f"\nUsing {len(R_base_marker)} poses for calibration")
    
    # 执行手眼标定
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=R_base_marker,  # 基座到marker的旋转矩阵
        t_gripper2base=t_base_marker,  # 基座到marker的平移向量
        R_target2cam=R_cam_marker,     # 相机到marker的旋转矩阵
        t_target2cam=t_cam_marker,     # 相机到marker的平移向量
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # 构建最终的变换矩阵
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = t_base_cam.squeeze()
    
    return T_base_cam


def calculate_camera_pose(marker_poses, T_base_marker):
    """
    直接计算相机在机器人基座标系下的位姿
    :param marker_poses: 相机检测到的marker位姿 (T_cam_marker)
    :param T_base_marker: marker在机器人基座标系下的已知位姿
    :return: T_base_cam 相机相对于机器人基座的变换矩阵
    """
    # 对于每个检测到的marker
    for marker_id, T_cam_marker in marker_poses.items():
        if marker_id in T_base_marker:
            # 已知: T_base_marker = T_base_cam * T_cam_marker
            # 因此: T_base_cam = T_base_marker * inv(T_cam_marker)
            T_base_cam = T_base_marker[marker_id] @ np.linalg.inv(T_cam_marker)
            return T_base_cam
    
    return None


if __name__ == "__main__":
    import time

    # robot_x = DobotApiDashboard("192.168.5.100", 29999)
    # robot_x.EnableRobot(load=1.0, centerX=0.0, centerY=0.0, centerZ=0.0)
    # robot_x.StartDrag()
    # time.sleep(1)
    # print("开始采集标定数据...")
    #
    # 初始化相机参数
    camera_matrix, dist_coeffs = init_camera_params()
    # robot_poses, camera_poses = collect_calibration_data(robot_x, num_poses=50)
    # robot_x.DisableRobot()

    # robot_s = nova2_gripper_v3()
    # robot_poses, camera_poses = load_calibration_data()
    # T_base_cam = estimate_camera_to_base(robot_s, robot_poses, camera_poses)
    # print("\nHand-Eye Calibration Result (T_base_cam):\n", T_base_cam)
    
    # # 验证标定结果
    # print("\nValidation:")
    # for marker_id in [0, 4]:
    #     print(f"\nTesting with marker {marker_id}:")
    #     # 选择一组数据进行验证
    #     test_pose = camera_poses[0]
    #     if marker_id in test_pose:
    #         T_cam_marker = test_pose[marker_id]
    #         # 通过标定结果计算marker在基座坐标系下的位置
    #         T_base_marker_calc = T_base_cam @ T_cam_marker
    #         print(f"Calculated marker position: {T_base_marker_calc[:3, 3]}")

    # 定义两个marker相对于基座的已知位置
    T_base_marker = {
        4: np.array([
            [0, -1, 0, 0.16],
            [1, 0, 0, 0.14],
            [0, 0, 1, 0.006],
            [0, 0, 0, 1]
        ])
    }
    
    # 1. 采集一帧数据即可
    marker_data = collect_fixed_marker_data(num_poses=1)
    
    if marker_data and len(marker_data) > 0:
        # 2. 计算相机位姿
        T_base_cam = calculate_camera_pose(marker_data[0], T_base_marker)
        if T_base_cam is not None:
            print("\nCamera pose in robot base frame (T_base_cam):\n", T_base_cam)
        else:
            print("未能检测到指定的marker")

# cam - base - matrix
# [[-0.03754594  0.89211193 -0.45025171  0.23938792]
#  [ 0.99612438 -0.00245088 -0.08792168  0.46887631]
#  [-0.07953949 -0.45180781 -0.88856242  0.5423873 ]
#  [ 0.          0.          0.          1.        ]]