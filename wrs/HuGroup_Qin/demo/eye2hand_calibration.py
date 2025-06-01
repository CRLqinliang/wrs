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


# 在文件开加全局变量
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
    # camera_matrix = np.array([
    #     [color_intr.fx, 0, color_intr.ppx],
    #     [0, color_intr.fy, color_intr.ppy],
    #     [0, 0, 1]
    # ])
    camera_matrix = np.array([
        [640.375, 0, 645.285],
        [0, 640.375, 359.196],
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


def detect_markers_pose(color_img, marker_size=0.047):
    """
    检测ArUco markers并估计其位姿
    :param color_img: 输入的彩色图像
    :param marker_size: marker黑色正方形的实际尺寸（米），默认0.047米
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


def collect_calibration_data(robot, num_poses=10, valid_marker_ids=range(14), save_dir="E:/Qin/wrs/wrs/HuGroup_Qin/demo/calibration_data", marker_size=0.047):
    """
    采集眼在手上标定数据，通过按空格键记录当前位姿
    :param robot: 机器人控制器实例
    :param num_poses: 需要采集的位姿数量
    :param valid_marker_ids: 有效的marker ID列表
    :param save_dir: 保存数据的目录
    :param marker_size: marker黑色正方形的实际尺寸（米），默认0.047米
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
    
    print(f"请移动机器人到不同位置，确保相机能看到至少一个固定marker")
    print(f"按空格键记录数据（目标：{num_poses}组）")
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
                        corners, marker_size, camera_matrix, dist_coeffs)
                    
                    # 为每个marker绘制坐标轴
                    for i in range(len(ids)):
                        cv2.drawFrameAxes(display_image, camera_matrix, dist_coeffs, 
                                        rvecs[i], tvecs[i], marker_size/2)
            
            # 检查是否检测到有效marker
            valid_markers_detected = []
            if ids is not None:
                valid_markers_detected = [id[0] for id in ids if id[0] in valid_marker_ids]
            
            # 在图像上显示检测状态
            detected_count = len(ids) if ids is not None else 0
            valid_count = len(valid_markers_detected)
            
            cv2.putText(display_image, f"checked markers: {detected_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_image, f"valid markers: {valid_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if valid_count > 0 else (0, 0, 255), 2)
            cv2.putText(display_image, f"collected: {len(robot_jnv_list)}/{num_poses}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('eye2hand calibration', display_image)
            key = cv2.waitKey(1)
            
            # 按空格键记录数据
            if key == 32:  # 32是空格键的ASCII码
                marker_poses = detect_markers_pose(color_image, marker_size=marker_size)
                # 过滤出有效的marker
                valid_poses = {id: pose for id, pose in marker_poses.items() if id in valid_marker_ids}
                
                if valid_poses:
                    # TODO 换成从真实机器人上获得TCP的位姿；
                    current_robot_jnv = robot.GetAngle()
                    print("robot_jnv:", current_robot_jnv)
                    print(f"检测到的有效markers: {list(valid_poses.keys())}")
                    
                    robot_jnv_list.append(current_robot_jnv)
                    camera_poses.append(valid_poses)
                    
                    # 保存数据
                    append_save(current_robot_jnv, valid_poses, save_dir)
                    print(f"成功记录第 {len(robot_jnv_list)}/{num_poses} 组数据")
                else:
                    print(f"未检测到有效marker，请调整位置后重试")
            
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


def estimate_camera_to_ee_multi_marker(robot_s, robot_poses, camera_poses, min_observations=3, max_reproj_error=0.05):
    """
    使用多个marker进行眼在手上标定，并添加数据过滤功能
    :param robot_s: 机器人运动学模型
    :param robot_poses: 机器人关节角度列表
    :param camera_poses: marker位姿列表，每个元素是一个字典 {marker_id: T_cam_marker, ...}
    :param min_observations: 每个marker最少需要的观测次数
    :param max_reproj_error: 重投影误差最大阈值，用于过滤数据
    :return: T_ee_cam, T_base_markers, calibration_results
    """
    # 统计每个marker的观测次数
    marker_counts = {}
    for pose_data in camera_poses:
        for marker_id in pose_data.keys():
            marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
    
    # 筛选出观测次数足够的marker
    valid_markers = [marker_id for marker_id, count in marker_counts.items() if count >= min_observations]
    
    if not valid_markers:
        raise ValueError(f"没有marker被观测到至少{min_observations}次，无法进行标定")
    
    print(f"有效marker列表: {valid_markers}")
    print(f"每个marker的观测次数: {marker_counts}")
    
    # 为每个有效marker创建标定数据
    calibration_results = {}
    T_base_markers = {}
    
    # 尝试不同的算法，选择误差最小的结果
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai方法"),
        (cv2.CALIB_HAND_EYE_PARK, "Park方法"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud方法"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff方法")
    ]
    
    best_method = None
    best_error = float('inf')
    best_T_ee_cam = None
    best_T_base_markers = {}
    best_results = {}
    
    for method_id, method_name in methods:
        print(f"\n尝试{method_name}:")
        try:
            # 为每个有效marker创建标定数据
            method_calibration_results = {}
            method_T_base_markers = {}
            method_total_error = 0
            method_count = 0
            
            for marker_id in valid_markers:
                print(f"\n使用marker {marker_id}进行标定:")
                
                R_base_ee = []
                t_base_ee = []
                R_cam_marker = []
                t_cam_marker = []
                
                for jnt_str, marker_poses in zip(robot_poses, camera_poses):
                    if marker_id not in marker_poses:
                        continue
                        
                    # 解析关节角度
                    start_idx = jnt_str.find('{') + 1
                    end_idx = jnt_str.find('}')
                    angles_str = jnt_str[start_idx:end_idx]
                    jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
                    
                    # 计算机器人末端在基座下的位姿
                    current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
                    T_base_ee_current = rm.homomat_from_posrot(current_pos, current_rotmat)
                    
                    # 获取相机观察到的marker位姿
                    T_cam_marker_current = marker_poses[marker_id]
                    
                    # 添加到标定数据中
                    R_base_ee.append(T_base_ee_current[:3, :3])
                    t_base_ee.append(T_base_ee_current[:3, 3].reshape(3,1))
                    R_cam_marker.append(T_cam_marker_current[:3, :3])
                    t_cam_marker.append(T_cam_marker_current[:3, 3].reshape(3,1))
                
                if len(R_base_ee) < 3:
                    print(f"Marker {marker_id}的有效数据不足，跳过")
                    continue
                    
                print(f"使用 {len(R_base_ee)} 组位姿进行标定")
                
                # 执行手眼标定
                R_ee_cam, t_ee_cam = cv2.calibrateHandEye(
                    R_gripper2base=R_base_ee,
                    t_gripper2base=t_base_ee,
                    R_target2cam=R_cam_marker,
                    t_target2cam=t_cam_marker,
                    method=method_id
                )
                
                # 构建相机到末端的变换矩阵
                T_ee_cam_current = np.eye(4)
                T_ee_cam_current[:3, :3] = R_ee_cam
                T_ee_cam_current[:3, 3] = t_ee_cam.squeeze()
                
                # 计算marker在基座坐标系下的位姿
                # 找到包含该marker的第一组数据
                idx = next((i for i, pose in enumerate(camera_poses) if marker_id in pose), 0)
                T_cam_marker_first = camera_poses[idx][marker_id]
                jnt_str = robot_poses[idx]
                
                start_idx = jnt_str.find('{') + 1
                end_idx = jnt_str.find('}')
                angles_str = jnt_str[start_idx:end_idx]
                jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
                
                current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
                T_base_ee_first = rm.homomat_from_posrot(current_pos, current_rotmat)
                
                # T_base_marker = T_base_ee * T_ee_cam * T_cam_marker
                T_base_marker = T_base_ee_first @ T_ee_cam_current @ T_cam_marker_first
                
                # 保存结果
                method_calibration_results[marker_id] = T_ee_cam_current
                method_T_base_markers[marker_id] = T_base_marker
                
                # 计算验证误差
                for jnt_str, marker_poses in zip(robot_poses, camera_poses):
                    if marker_id not in marker_poses:
                        continue
                        
                    # 解析关节角度
                    start_idx = jnt_str.find('{') + 1
                    end_idx = jnt_str.find('}')
                    angles_str = jnt_str[start_idx:end_idx]
                    jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
                    
                    # 计算机器人末端在基座下的位姿
                    current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
                    T_base_ee = rm.homomat_from_posrot(current_pos, current_rotmat)
                    
                    # 获取相机观察到的marker位姿
                    T_cam_marker = marker_poses[marker_id]
                    
                    # 通过标定结果计算marker在基座坐标系下的位置
                    T_base_marker_calc = T_base_ee @ T_ee_cam_current @ T_cam_marker
                    
                    # 计算误差
                    error_translation = np.linalg.norm(T_base_marker[:3, 3] - T_base_marker_calc[:3, 3])
                    method_total_error += error_translation
                    method_count += 1
            
            if method_count > 0:
                method_avg_error = method_total_error / method_count
                print(f"\n{method_name}平均位置误差: {method_avg_error:.6f} 米")
                
                if method_avg_error < best_error:
                    best_error = method_avg_error
                    best_method = method_name
                    
                    # 计算所有有效marker的平均结果
                    T_ee_cam_sum = np.zeros((4, 4))
                    for T_ee_cam in method_calibration_results.values():
                        T_ee_cam_sum += T_ee_cam
                    
                    best_T_ee_cam = T_ee_cam_sum / len(method_calibration_results)
                    
                    # 确保旋转矩阵是正交的
                    U, _, Vt = np.linalg.svd(best_T_ee_cam[:3, :3])
                    best_T_ee_cam[:3, :3] = U @ Vt
                    
                    best_T_base_markers = method_T_base_markers.copy()
                    best_results = method_calibration_results.copy()
        except Exception as e:
            print(f"{method_name}出错: {e}")
    
    if best_method is None:
        raise ValueError("所有标定方法都失败，请检查数据")
        
    print(f"\n最佳方法: {best_method}，平均误差: {best_error:.6f} 米")
    
    return best_T_ee_cam, best_T_base_markers, best_results


if __name__ == "__main__":
    import time

    # 设置有效的marker ID列表和正确的marker尺寸
    VALID_MARKER_IDS = list(range(14))  # 0-13
    MARKER_SIZE = 0.047  # 47毫米，黑色正方形的实际尺寸
    
    robot_x = DobotApiDashboard("192.168.5.100", 29999)
    robot_x.EnableRobot(load=1.0, centerX=0.0, centerY=0.0, centerZ=0.0)
    robot_x.StartDrag()
    time.sleep(1)
    print("start collecting eye2hand calibration data...")
    
    # 初始化相机参数
    camera_matrix, dist_coeffs = init_camera_params()
    
    # 采集标定数据
    robot_poses, camera_poses = collect_calibration_data(
        robot_x, 
        num_poses=50,  # 增加采集数据量
        valid_marker_ids=VALID_MARKER_IDS,
        marker_size=MARKER_SIZE
    )
    robot_x.StopDrag()
    robot_x.DisableRobot()

    # 加载机器人模型
    robot_s = nova2_gripper_v3()
    
    # 如果没有采集新数据，则加载已有数据
    # if len(robot_poses) == 0:
    robot_poses, camera_poses = load_calibration_data()
    
    # 执行眼在手上标定（使用多个marker）
    T_ee_cam, T_base_markers, individual_results = estimate_camera_to_ee_multi_marker(
        robot_s, robot_poses, camera_poses, min_observations=5  # 增加最小观测次数要求
    )
    
    print("\n眼在手上标定最终结果:")
    print("相机到末端的变换 (T_ee_cam):\n", T_ee_cam)
    
    # # 显示每个marker的单独标定结果
    # print("\n各个marker的单独标定结果:")
    # for marker_id, T_ee_cam_marker in individual_results.items():
    #     print(f"Marker {marker_id}:")
    #     print(T_ee_cam_marker)
    #     print()
    
    # # 显示每个marker在基座坐标系下的位姿
    # print("\n各个marker在基座坐标系下的位姿:")
    # for marker_id, T_base_marker in T_base_markers.items():
    #     print(f"Marker {marker_id}:")
    #     print(T_base_marker)
    #     print()
    
    # 验证标定结果
    print("\n验证结果:")
    total_error = 0
    count = 0
    
    # 留出一部分数据用于交叉验证
    validate_indices = list(range(0, len(robot_poses), 5))  # 每5个取1个作为验证数据
    train_indices = [i for i in range(len(robot_poses)) if i not in validate_indices]
    
    for i in validate_indices:
        if i < len(robot_poses) and i < len(camera_poses):
            jnt_str = robot_poses[i]
            marker_pose = camera_poses[i]
            
            print(f"\n验证第 {i+1} 组数据:")
            
            # 解析关节角度
            start_idx = jnt_str.find('{') + 1
            end_idx = jnt_str.find('}')
            angles_str = jnt_str[start_idx:end_idx]
            jnt_values = np.array([float(x.strip()) for x in angles_str.split(',')])
            
            # 计算机器人末端在基座下的位姿
            current_pos, current_rotmat = robot_s.fk(jnt_values * np.pi / 180)
            T_base_ee = rm.homomat_from_posrot(current_pos, current_rotmat)
            
            # 验证每个检测到的marker
            for marker_id, T_cam_marker in marker_pose.items():
                if marker_id not in T_base_markers:
                    continue
                    
                # 通过标定结果计算marker在基座坐标系下的位置
                T_base_marker_calc = T_base_ee @ T_ee_cam @ T_cam_marker
                
                # 计算误差
                error_translation = np.linalg.norm(T_base_markers[marker_id][:3, 3] - T_base_marker_calc[:3, 3])
                total_error += error_translation
                count += 1
                
                print(f"Marker {marker_id}:")
                print(f"  实际位置: {T_base_markers[marker_id][:3, 3]}")
                print(f"  计算位置: {T_base_marker_calc[:3, 3]}")
                print(f"  位置误差: {error_translation:.6f} 米")
    
    if count > 0:
        print(f"\n平均位置误差: {total_error/count:.6f} 米")
        
    # 保存最终的标定结果
    np.save("E:/Qin/wrs/wrs/HuGroup_Qin/demo/T_ee_cam.npy", T_ee_cam)
    print("\n标定结果已保存")

# cam - base - matrix
# [[-0.03754594  0.89211193 -0.45025171  0.23938792]
#  [ 0.99612438 -0.00245088 -0.08792168  0.46887631]
#  [-0.07953949 -0.45180781 -0.88856242  0.5423873 ]
#  [ 0.          0.          0.          1.        ]]