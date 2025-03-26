"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate the camera
"""
__VERSION__ = '0.0.1'

import os
import sys
sys.path.append("E:/Qin/wrs")
from pathlib import Path
import json
import numpy as np
from abc import ABC, abstractmethod
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri
import numpy as np
import open3d as o3d  # 添加open3d库
import cv2
import time


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


class ManualCalibrationBase(ABC):
    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 move_resolution=.001, rotation_resolution=rm.np.radians(5)):
        """
        Class to manually calibrate the point cloud data
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix. If it is None, the init calibration matrix will be identity matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = rm.eye(4) if init_calib_mat is None else init_calib_mat

        # variable stores robot plot and the point cloud plot
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        #
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # add task
        taskMgr.doMethodLater(.05, self.sync_rbt, "sync rbt", )
        taskMgr.doMethodLater(.02, self.adjust, "manual adjust the mph")
        taskMgr.doMethodLater(.5, self.sync_pcd, "sync mph", )

    @abstractmethod
    def get_pcd(self):
        """
        An abstract method to get the point cloud
        :return: An Nx3 ndarray represents the point cloud
        """
        pass

    @abstractmethod
    def get_rbtx_jnt_values(self):
        """
        An abstract method to get the robot joint angles
        :return: 1xn ndarray, n is degree of freedom of the robot
        """
        pass

    @abstractmethod
    def align_pcd(self, pcd):
        """
        Abstract method to align the mph according to the calibration matrix
        implement the Eye-in-hand or eye-to-hand transformation here
        https://support.zivid.com/en/latest/academy/applications/hand-eye/system-configurations.html
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by moving
        :param dir: The local move motion_vec based on the calibration matrix coordinate
        :param dir_global: The global move motion_vec based on the world coordinate
        :return:
        """
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by rotating
        :param dir: The local motion_vec of the calibration matrix
        :param dir_global: The global motion_vec
        :return:
        """
        self._init_calib_mat[:3, :3] = rm.rotmat_from_axangle(dir_global, rm.radians(
            self.rotation_resolution)) @ self._init_calib_mat[:3, :3]

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n'):
        def add_key(keys: str or list):
            """
            Add key to  the keymap. The default keymap can be seen in visualization/panda/inputmanager.py
            :param keys: the keys added to the keymap
            """
            assert isinstance(keys, str) or isinstance(keys, list)

            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw

    def sync_pcd(self, task):
        """
        Synchronize the real r obot and the simulation robot
        :return: None
        """

        self._pcd = self.get_pcd()
        self.plot()
        return task.again

    def sync_rbt(self, task):
        jnt_values = self.get_rbtx_jnt_values()
        jnt_values = np.array(jnt_values) * np.pi / 180
        self._rbt_s.goto_given_conf(jnt_values)
        self.plot()
        return task.again

    def save(self):
        """
        Save manual calibration results
        :return:
        """
        dump_json({'affine_mat': self._init_calib_mat.tolist()}, "manual_calibration.json", reminder=False)

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot
        :param task:
        :return:
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel()
        self._plot_node_rbt.attach_to(base)
        pcd = self._pcd
        if pcd is not None:
            if pcd.shape[1] == 6:
                pcd, pcd_color = pcd[:, :3], pcd[:, 3:6]
                pcd_color_rgba = rm.np.append(pcd_color, rm.np.ones((len(pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = rm.np.array([1, 1, 1, 1])
            pcd_r = self.align_pcd(pcd)
            self._plot_node_pcd = mgm.gen_pointcloud(points=pcd_r, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
        if task is not None:
            return task.again

    def adjust(self, task):
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x')
            print(self._init_calib_mat)
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x_cw')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y_cw')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_ccw')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z_cw')
            print(self._init_calib_mat)
        elif base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_ccw')
            print(self._init_calib_mat)
        else:
            return task.again
        self.plot()
        self.save()
        return task.again


class Nova2WRSV3GripperManualCalib(ManualCalibrationBase):
        """
        Eye in hand example
        """

        def get_pcd(self):
            pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth()
            return rm.np.hstack((pcd, pcd_color))

        def get_rbtx_jnt_values(self):
            return_str = self._rbt_x.GetAngle()
            # 解析格式为 "ErrorID,{J1,J2,J3,J4,J5,J6},GetAngle();" 的字符串
            bracket_content = return_str.split('{')[1].split('}')[0]
            # 分割并转换为浮点数
            jnt_values = [float(i) for i in bracket_content.split(',')]
            return jnt_values

        def align_pcd(self, pcd):
            r2cam_mat = self._init_calib_mat
            rbt_pose_str = self._rbt_x.GetPose()
            # 解析格式为 "ErrorID,{X,Y,Z,A,B,C},GetPose();" 的字符串
            bracket_content = rbt_pose_str.split('{')[1].split('}')[0]
            # 分割并转换为浮点数
            rbt_pose = [float(i) for i in bracket_content.split(',')]
            pos = rbt_pose[:3]
            pos = [i / 1000 for i in pos]
            rotmat = rm.rotmat_from_euler(rbt_pose[3], rbt_pose[4], rbt_pose[5])
            w2r_mat = rm.homomat_from_posrot(pos, rotmat)
            w2c_mat = w2r_mat.dot(r2cam_mat)
            return rm.transform_points_by_homomat(w2c_mat, points=pcd)


def visualize_camera_pointcloud(base, robot_interface, camera_interface, tcp2cam_matrix):
    """
    使用已知的TCP到相机的转换矩阵，将相机捕获的RGB深度点云显示在世界坐标系中
    
    参数:
    base - 世界坐标系对象，用于显示点云
    robot_interface - 机器人接口，用于获取机器人当前位姿
    camera_interface - 相机接口，用于获取点云数据
    tcp2cam_matrix - TCP到相机的转换矩阵（4x4齐次变换矩阵）
    
    返回:
    pointcloud_node - 点云节点对象，已经attach到base上
    """

    # 获取点云数据
    pcd, pcd_color, _, _ = camera_interface.get_pcd_texture_depth()
    
    # 获取机器人当前位姿
    rbt_pose_str = robot_interface.GetPose()
    # 解析格式为 "ErrorID,{X,Y,Z,A,B,C},GetPose();" 的字符串
    bracket_content = rbt_pose_str.split('{')[1].split('}')[0]
    # 分割并转换为浮点数
    rbt_pose = [float(i) for i in bracket_content.split(',')]
    # 提取位置和旋转信息
    pos = rbt_pose[:3]
    # 将毫米转换为米
    pos = [i / 1000 for i in pos]
    # 从欧拉角创建旋转矩阵
    rotmat = rm.rotmat_from_euler(rbt_pose[3]/180*np.pi, rbt_pose[4]/180*np.pi, rbt_pose[5]/180*np.pi)
    
    # 创建世界到机器人TCP的变换矩阵
    w2tcp_mat = rm.homomat_from_posrot(pos, rotmat)
    mgm.gen_frame(w2tcp_mat[:3, 3], w2tcp_mat[:3, :3]).attach_to(base)
    # 计算世界到相机的变换矩阵
    w2cam_mat = w2tcp_mat @ tcp2cam_matrix
    mgm.gen_frame(w2cam_mat[:3, 3], w2cam_mat[:3, :3]).attach_to(base)
    
    # 将点云从相机坐标系转换到世界坐标系
    transformed_pcd = rm.transform_points_by_homomat(w2cam_mat, points=pcd)
    
    # 创建RGBA颜色数组
    pcd_color_rgba = rm.np.append(pcd_color, rm.np.ones((len(pcd_color), 1)), axis=1)
    
    # 生成点云对象并附加到世界坐标系
    pointcloud_node = mgm.gen_pointcloud(points=transformed_pcd, rgba=pcd_color_rgba)
    
    # 可视化相机坐标系
    mgm.gen_frame(w2cam_mat[:3, 3], w2cam_mat[:3, :3]).attach_to(pointcloud_node)
    
    # 将点云附加到世界坐标系
    pointcloud_node.attach_to(base)
    
    return pointcloud_node


def realtime_pointcloud_visualization(base, robot_s,robot_interface, camera_interface, tcp2cam_matrix, update_interval=0.5):
    """
    实时更新点云显示
    
    参数:
    base - 世界坐标系对象，用于显示点云
    robot_interface - 机器人接口，用于获取机器人当前位姿
    camera_interface - 相机接口，用于获取点云数据
    tcp2cam_matrix - TCP到相机的转换矩阵（4x4齐次变换矩阵）
    update_interval - 更新间隔（秒），默认为0.5秒
    
    返回:
    task_id - 任务ID，可用于停止实时更新
    """
    pointcloud_node = None
    robot_s_node = None

    def get_rbtx_jnt_values(robot_x):
        return_str = robot_x.GetAngle()
        # 解析格式为 "ErrorID,{J1,J2,J3,J4,J5,J6},GetAngle();" 的字符串
        bracket_content = return_str.split('{')[1].split('}')[0]
        # 分割并转换为浮点数
        jnt_values = [float(i) for i in bracket_content.split(',')]
        return jnt_values

    def update_pointcloud(task):
        nonlocal pointcloud_node
        nonlocal robot_s_node
        # 清除之前的点云
        if pointcloud_node is not None:
            pointcloud_node.detach()
        if robot_s_node is not None:
            robot_s_node.detach()
        
        # 获取新的点云并显示
        pointcloud_node = visualize_camera_pointcloud(base, robot_interface, camera_interface, tcp2cam_matrix)
        jnt_values = get_rbtx_jnt_values(robot_interface)
        jnt_values = np.array(jnt_values) * np.pi / 180
        robot_s.goto_given_conf(jnt_values)
        robot_s_node = robot_s.gen_meshmodel()
        robot_s_node.attach_to(base)
        # 继续任务
        return task.again
    
    # 创建定时任务
    task_id = base.taskMgr.doMethodLater(
        update_interval,
        update_pointcloud,
        "update_pointcloud_task"
    )
    
    return task_id


def detect_ar_marker(camera_interface, marker_id, marker_size=0.05, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    """
    使用OpenCV检测相机图像中的AR标记并返回相机到标记的变换矩阵
    
    参数:
    camera_interface - 相机接口，需要有get_color_img()方法
    marker_id - 要检测的AR标记ID
    marker_size - AR标记的实际尺寸（米），默认为0.05米（5厘米）
    aruco_dict_type - ArUco字典类型，默认为DICT_4X4_50
    
    返回:
    cam2marker_matrix - 相机到AR标记的4x4变换矩阵，如果未检测到则返回None
    """
    try:
        # 获取RGB图像
        color_image = camera_interface.get_color_img()
        if color_image is None:
            print(f"无法获取相机图像")
            return None
        
        # 获取相机内参
        camera_matrix = camera_interface.intr_mat
        dist_coeffs = camera_interface.intr_distcoeffs
        
        # 初始化 ArUco 检测器
        # 检查OpenCV版本，使用适当的API
        opencv_version = cv2.__version__.split('.')
        major_version = int(opencv_version[0])
        
        if major_version >= 4 and hasattr(cv2.aruco, 'ArucoDetector'):
            # OpenCV 4.7+的新API
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            # 检测markers
            corners, ids, rejected = detector.detectMarkers(color_image)
        else:
            # 旧版API
            aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters_create()
            
            # 检测markers
            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=parameters)
        
        # 可视化检测结果（调试用）
        debug_image = color_image.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)
        
        # 保存调试图像
        debug_dir = "ar_marker_debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(f"{debug_dir}/detect_marker_{marker_id}_{time.time()}.jpg", debug_image)
        
        # 检查是否检测到目标标记
        if ids is None or marker_id not in ids.flatten():
            print(f"未检测到ID为{marker_id}的AR标记")
            return None
        
        # 找到目标标记的索引
        marker_idx = np.where(ids.flatten() == marker_id)[0][0]
        
        # 估计标记的位姿
        if major_version >= 4 and hasattr(cv2.aruco, 'estimatePoseSingleMarkers'):
            # 使用新API
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[marker_idx]], marker_size, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
        else:
            # 旧API
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[marker_idx:marker_idx+1], marker_size, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0]
            tvec = tvecs[0]
        
        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # 构建相机到标记的变换矩阵
        cam2marker_matrix = np.eye(4)
        cam2marker_matrix[:3, :3] = rotation_matrix
        cam2marker_matrix[:3, 3] = tvec
        
        # 可视化位姿估计结果（调试用）
        pose_image = color_image.copy()
        cv2.drawFrameAxes(pose_image, camera_matrix, dist_coeffs, rvec, tvec, marker_size/2)
        cv2.imwrite(f"{debug_dir}/pose_marker_{marker_id}_{time.time()}.jpg", pose_image)
        
        print(f"成功检测到ID为{marker_id}的AR标记")
        return cam2marker_matrix
    
    except Exception as e:
        print(f"AR标记检测出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def calibrate_environment_camera(robot_interface, tcp_camera_interface, env_camera_interface, 
                                tcp2cam_matrix, ar_marker_ids, marker_size=0.05, num_samples=10, 
                                visualize=True, base=None):
    """
    通过AR标记计算环境相机(D435)在机器人基坐标系下的位姿
    
    参数:
    robot_interface - 机器人接口，用于获取机器人当前位姿
    tcp_camera_interface - 安装在TCP上的相机接口(D405)
    env_camera_interface - 环境中的相机接口(D435)
    tcp2cam_matrix - TCP到相机的转换矩阵（4x4齐次变换矩阵）
    ar_marker_ids - AR标记ID列表
    marker_size - AR标记的实际尺寸（米），默认为0.05米（5厘米）
    num_samples - 采样次数，默认为10
    visualize - 是否可视化结果，默认为True
    base - 世界坐标系对象，用于显示结果，仅当visualize=True时需要
    
    返回:
    world2env_cam_matrix - 世界坐标系到环境相机的转换矩阵（4x4齐次变换矩阵）
    """
    import numpy as np
    import time
    import cv2
    
    # 用于存储所有采样的转换矩阵
    all_world2env_matrices = []
    
    # 对每个AR标记进行多次采样
    for marker_id in ar_marker_ids:
        print(f"正在处理AR标记 {marker_id}...")
        
        marker_samples = []
        valid_samples = 0
        
        # 尝试获取足够的有效样本
        attempts = 0
        max_attempts = num_samples * 3  # 最多尝试次数
        
        while valid_samples < num_samples and attempts < max_attempts:
            attempts += 1
            print(f"  尝试采样 {attempts}/{max_attempts}，已获得有效样本 {valid_samples}/{num_samples}")
            
            # 1. 通过TCP相机(D405)获取AR标记在相机坐标系下的位姿
            tcp_cam2marker_matrix = detect_ar_marker(tcp_camera_interface, marker_id, marker_size)
            if tcp_cam2marker_matrix is None:
                print(f"  TCP相机未检测到标记 {marker_id}，跳过此采样")
                time.sleep(0.5)  # 等待一段时间再尝试
                continue
                
            # 2. 通过环境相机(D435)获取AR标记在相机坐标系下的位姿
            env_cam2marker_matrix = detect_ar_marker(env_camera_interface, marker_id, marker_size)
            if env_cam2marker_matrix is None:
                print(f"  环境相机未检测到标记 {marker_id}，跳过此采样")
                time.sleep(0.5)  # 等待一段时间再尝试
                continue
            
            # 3. 获取机器人当前位姿
            rbt_pose_str = robot_interface.GetPose()
            bracket_content = rbt_pose_str.split('{')[1].split('}')[0]
            rbt_pose = [float(i) for i in bracket_content.split(',')]
            pos = rbt_pose[:3]
            pos = [i / 1000 for i in pos]  # 毫米转米
            rotmat = rm.rotmat_from_euler(rbt_pose[3]/180*np.pi, rbt_pose[4]/180*np.pi, rbt_pose[5]/180*np.pi)
            world2tcp_matrix = rm.homomat_from_posrot(pos, rotmat)
            
            # 4. 计算世界坐标系到TCP相机的转换矩阵
            world2tcp_cam_matrix = world2tcp_matrix @ tcp2cam_matrix
            
            # 5. 计算世界坐标系到AR标记的转换矩阵
            world2marker_matrix = world2tcp_cam_matrix @ tcp_cam2marker_matrix
            
            # 6. 计算环境相机到AR标记的逆矩阵（AR标记到环境相机）
            marker2env_cam_matrix = np.linalg.inv(env_cam2marker_matrix)
            
            # 7. 计算世界坐标系到环境相机的转换矩阵
            world2env_cam_matrix = world2marker_matrix @ marker2env_cam_matrix
            
            marker_samples.append(world2env_cam_matrix)
            valid_samples += 1
            
            # 保存中间结果（以防程序中断）
            if valid_samples % 2 == 0:
                temp_avg = average_transformation_matrices(marker_samples)
                dump_json({
                    'marker_id': marker_id,
                    'samples_collected': valid_samples,
                    'temp_world2d435_matrix': temp_avg.tolist()
                }, f"d435_calib_marker_{marker_id}_temp.json", reminder=False)
            
            time.sleep(1.0)  # 等待一段时间，确保获取不同的采样
        
        if marker_samples:
            # 对当前标记的所有有效采样取平均
            avg_matrix = average_transformation_matrices(marker_samples)
            all_world2env_matrices.append(avg_matrix)
            
            # 保存每个标记的结果
            dump_json({
                'marker_id': marker_id,
                'world2d435_matrix': avg_matrix.tolist()
            }, f"d435_calib_marker_{marker_id}.json", reminder=False)
    
    if not all_world2env_matrices:
        print("未能获取有效的转换矩阵！")
        return None
    
    # 对所有标记的结果取平均，得到最终的转换矩阵
    final_world2env_cam_matrix = average_transformation_matrices(all_world2env_matrices)
    
    print("环境相机在世界坐标系下的位姿计算完成！")
    print("转换矩阵:")
    print(final_world2env_cam_matrix)
    
    # 可视化结果
    if visualize and base is not None:
        # 显示世界坐标系
        mgm.gen_frame(np.zeros(3)).attach_to(base)
        
        # 显示环境相机坐标系
        env_cam_frame = mgm.gen_frame(
            final_world2env_cam_matrix[:3, 3], 
            final_world2env_cam_matrix[:3, :3],
            ax_length=0.1
        )
        env_cam_frame.attach_to(base)
        
        print("已在世界坐标系中显示环境相机位姿")
    
    return final_world2env_cam_matrix


def average_transformation_matrices(matrices):
    """
    计算多个变换矩阵的平均值
    
    参数:
    matrices - 变换矩阵列表，每个矩阵是4x4的齐次变换矩阵
    
    返回:
    avg_matrix - 平均变换矩阵
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    
    if len(matrices) == 1:
        return matrices[0]
    
    # 提取所有位置并计算平均位置
    positions = np.array([matrix[:3, 3] for matrix in matrices])
    avg_position = np.mean(positions, axis=0)
    
    # 提取所有旋转矩阵并转换为四元数
    rotations = [Rotation.from_matrix(matrix[:3, :3]) for matrix in matrices]
    quats = np.array([rot.as_quat() for rot in rotations])
    
    # 确保四元数在同一半球（避免平均时的问题）
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]
    
    # 计算平均四元数
    avg_quat = np.mean(quats, axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)  # 归一化
    
    # 转换回旋转矩阵
    avg_rotation = Rotation.from_quat(avg_quat).as_matrix()
    
    # 构建平均变换矩阵
    avg_matrix = np.eye(4)
    avg_matrix[:3, :3] = avg_rotation
    avg_matrix[:3, 3] = avg_position
    
    return avg_matrix


def realtime_env_camera_visualization(base, robot_s, env_camera_interface, world2env_cam_matrix, update_interval=0.5):
    """
    实时更新环境相机点云显示
    
    参数:
    base - 世界坐标系对象，用于显示点云
    robot_s - 仿真机器人对象，用于显示机器人模型
    env_camera_interface - 环境相机接口，用于获取点云数据
    world2env_cam_matrix - 世界坐标系到环境相机的转换矩阵（4x4齐次变换矩阵）
    update_interval - 更新间隔（秒），默认为0.5秒
    
    返回:
    task_id - 任务ID，可用于停止实时更新
    """
    pointcloud_node = None
    robot_s_node = None
    env_camera_frame = None

    def update_pointcloud(task):
        nonlocal pointcloud_node
        nonlocal robot_s_node
        nonlocal env_camera_frame
        
        # 清除之前的点云和机器人模型
        if pointcloud_node is not None:
            pointcloud_node.detach()
        if robot_s_node is not None:
            robot_s_node.detach()
        if env_camera_frame is not None:
            env_camera_frame.detach()
        
        try:
            # 获取点云数据
            pcd, pcd_color, _, _ = env_camera_interface.get_pcd_texture_depth()
            
            if pcd is not None and len(pcd) > 0:
                # 将点云从相机坐标系转换到世界坐标系
                transformed_pcd = rm.transform_points_by_homomat(world2env_cam_matrix, points=pcd)
                
                # 创建RGBA颜色数组
                pcd_color_rgba = rm.np.append(pcd_color, rm.np.ones((len(pcd_color), 1)), axis=1)
                
                # 生成点云对象并附加到世界坐标系
                pointcloud_node = mgm.gen_pointcloud(points=transformed_pcd, rgba=pcd_color_rgba)
                pointcloud_node.attach_to(base)
            
            # 显示环境相机坐标系
            env_camera_frame = mgm.gen_frame(
                world2env_cam_matrix[:3, 3], 
                world2env_cam_matrix[:3, :3],
                length=0.1,
                thickness=0.01,
                rgba=[0, 0, 1, 1]  # 蓝色
            )
            env_camera_frame.attach_to(base)
            
            # 更新机器人模型显示
            if robot_s is not None:
                robot_s_node = robot_s.gen_meshmodel()
                robot_s_node.attach_to(base)
        
        except Exception as e:
            print(f"更新点云时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 继续任务
        return task.again
    
    # 创建定时任务
    task_id = base.taskMgr.doMethodLater(
        update_interval,
        update_pointcloud,
        "update_env_pointcloud_task"
    )
    
    print(f"已启动环境相机实时点云显示，任务ID: {task_id}")
    return task_id


def realtime_dual_camera_visualization(base, robot_s, robot_interface, tcp_camera_interface, 
                                      env_camera_interface, tcp2cam_matrix, world2env_cam_matrix, 
                                      update_interval=0.5):
    """
    同时实时更新TCP相机和环境相机的点云显示
    
    参数:
    base - 世界坐标系对象，用于显示点云
    robot_s - 仿真机器人对象，用于显示机器人模型
    robot_interface - 机器人接口，用于获取机器人当前位姿
    tcp_camera_interface - TCP相机接口，用于获取点云数据
    env_camera_interface - 环境相机接口，用于获取点云数据
    tcp2cam_matrix - TCP到相机的转换矩阵（4x4齐次变换矩阵）
    world2env_cam_matrix - 世界坐标系到环境相机的转换矩阵（4x4齐次变换矩阵）
    update_interval - 更新间隔（秒），默认为0.5秒
    
    返回:
    task_id - 任务ID，可用于停止实时更新
    """
    tcp_pointcloud_node = None
    env_pointcloud_node = None
    robot_s_node = None
    tcp_camera_frame = None
    env_camera_frame = None

    def get_rbtx_jnt_values(robot_x):
        return_str = robot_x.GetAngle()
        # 解析格式为 "ErrorID,{J1,J2,J3,J4,J5,J6},GetAngle();" 的字符串
        bracket_content = return_str.split('{')[1].split('}')[0]
        # 分割并转换为浮点数
        jnt_values = [float(i) for i in bracket_content.split(',')]
        return jnt_values

    def update_pointclouds(task):
        nonlocal tcp_pointcloud_node
        nonlocal env_pointcloud_node
        nonlocal robot_s_node
        nonlocal tcp_camera_frame
        nonlocal env_camera_frame
        
        # 清除之前的点云和机器人模型
        if tcp_pointcloud_node is not None:
            tcp_pointcloud_node.detach()
        if env_pointcloud_node is not None:
            env_pointcloud_node.detach()
        if robot_s_node is not None:
            robot_s_node.detach()
        if tcp_camera_frame is not None:
            tcp_camera_frame.detach()
        if env_camera_frame is not None:
            env_camera_frame.detach()
        
        try:
            # 更新机器人模型
            jnt_values = get_rbtx_jnt_values(robot_interface)
            jnt_values = np.array(jnt_values) * np.pi / 180
            robot_s.goto_given_conf(jnt_values)
            robot_s_node = robot_s.gen_meshmodel()
            robot_s_node.attach_to(base)
            
            # 获取机器人当前位姿
            rbt_pose_str = robot_interface.GetPose()
            bracket_content = rbt_pose_str.split('{')[1].split('}')[0]
            rbt_pose = [float(i) for i in bracket_content.split(',')]
            pos = rbt_pose[:3]
            pos = [i / 1000 for i in pos]  # 毫米转米
            rotmat = rm.rotmat_from_euler(rbt_pose[3]/180*np.pi, rbt_pose[4]/180*np.pi, rbt_pose[5]/180*np.pi)
            world2tcp_matrix = rm.homomat_from_posrot(pos, rotmat)
            
            # 计算世界坐标系到TCP相机的转换矩阵
            world2tcp_cam_matrix = world2tcp_matrix @ tcp2cam_matrix
            
            # 显示TCP相机坐标系
            tcp_camera_frame = mgm.gen_frame(
                world2tcp_cam_matrix[:3, 3], 
                world2tcp_cam_matrix[:3, :3],
                ax_length=0.05
            )
            tcp_camera_frame.attach_to(base)
            
            # 获取并显示TCP相机点云
            tcp_pcd, tcp_pcd_color, _, _ = tcp_camera_interface.get_pcd_texture_depth()
            if tcp_pcd is not None and len(tcp_pcd) > 0:
                # 将点云从相机坐标系转换到世界坐标系
                transformed_tcp_pcd = rm.transform_points_by_homomat(world2tcp_cam_matrix, points=tcp_pcd)
                
                # 创建RGBA颜色数组
                tcp_pcd_color_rgba = rm.np.append(tcp_pcd_color, rm.np.ones((len(tcp_pcd_color), 1)), axis=1)
                
                # 生成点云对象并附加到世界坐标系
                tcp_pointcloud_node = mgm.gen_pointcloud(points=transformed_tcp_pcd, rgba=tcp_pcd_color_rgba)
                tcp_pointcloud_node.attach_to(base)
            
            # 显示环境相机坐标系
            env_camera_frame = mgm.gen_frame(
                world2env_cam_matrix[:3, 3], 
                world2env_cam_matrix[:3, :3],
                ax_length=0.1
            )
            env_camera_frame.attach_to(base)
            
            # 获取并显示环境相机点云
            env_pcd, env_pcd_color, _, _ = env_camera_interface.get_pcd_texture_depth()
            if env_pcd is not None and len(env_pcd) > 0:
                # 将点云从相机坐标系转换到世界坐标系
                transformed_env_pcd = rm.transform_points_by_homomat(world2env_cam_matrix, points=env_pcd)
                
                # 创建RGBA颜色数组
                env_pcd_color_rgba = rm.np.append(env_pcd_color, rm.np.ones((len(env_pcd_color), 1)), axis=1)
                
                # 生成点云对象并附加到世界坐标系
                env_pointcloud_node = mgm.gen_pointcloud(points=transformed_env_pcd, rgba=env_pcd_color_rgba)
                env_pointcloud_node.attach_to(base)
        
        except Exception as e:
            print(f"更新点云时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 继续任务
        return task.again
    
    # 创建定时任务
    task_id = base.taskMgr.doMethodLater(
        update_interval,
        update_pointclouds,
        "update_dual_pointcloud_task"
    )
    
    print(f"已启动双相机实时点云显示，任务ID: {task_id}")
    return task_id


if __name__ == "__main__":
    import time
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400, find_devices
    from wrs.robot_sim.robots.xarmlite6_wg import x6wg2
    from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApiDashboard
    from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper import nova2_gripper_v3

    base = wd.World(cam_pos=rm.vec(2, 0, 1.5), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame(np.zeros(3)).attach_to(base)

    rbtx = DobotApiDashboard("192.168.5.100", 29999)
    rbtx.EnableRobot(load=1.0, centerX=0.0, centerY=0.0, centerZ=0.0)
    rbtx.StartDrag()
    time.sleep(1)
    print("start collecting eye2hand calibration data...")
  
    serials, _ = find_devices()
    if not serials:
        print("未找到RealSense相机!")
    # the first frame contains no data information
    rs_pipe_d405 = RealSenseD400(
        device=serials[0],
        enable_filters=True,
        decimation_magnitude=1,      # 抽取滤波器倍数
        threshold_min=0.1,           # 最小距离阈值（米）
        threshold_max=0.6,           # 最大距离阈值（米）
        spatial_magnitude=1,         # 空间滤波器平滑程度
        spatial_smooth_alpha=0.8,    # 空间滤波器alpha参数
        spatial_smooth_delta=20,     # 空间滤波器delta参数
        temporal_alpha=0.2,          # 时间滤波器alpha参数
        temporal_delta=60            # 时间滤波器delta参数
    )
    rs_pipe_d435 = RealSenseD400(
            device=serials[1],
            enable_filters=True,
            decimation_magnitude=1,      # 抽取滤波器倍数
            threshold_min=0.1,           # 最小距离阈值（米）
            threshold_max=0.8,           # 最大距离阈值（米）
            spatial_magnitude=1,         # 空间滤波器平滑程度
            spatial_smooth_alpha=0.3,    # 空间滤波器alpha参数
            spatial_smooth_delta=10,     # 空间滤波器delta参数
            temporal_alpha=0.1,          # 时间滤波器alpha参数
            temporal_delta=50            # 时间滤波器delta参数
    )
    
    rbt = nova2_gripper_v3()
    # robot2_tcp2cam = np.array([[-0.20228920862090363, -0.7317049162736713, 0.6509124300371265, 0.07700000000000004], 
    #                            [0.4795315203506582, 0.5055156166228459, 0.7172889810537633, 0.029000000000000012], 
    #                            [-0.8538902722635181, 0.4572328475206661, 0.24861521691653746, 0.03700000000000003], 
    #                            [0.0, 0.0, 0.0, 1.0]])
    # xarm_mc = Nova2WRSV3GripperanualCalib(rbt_s=rbt, rbt_x=rbtx, init_calib_mat=robot2_tcp2cam, sensor_hdl=rs_pipe)
    # 示例1：使用visualize_camera_pointcloud函数显示单帧点云
    # pointcloud_node = visualize_camera_pointcloud(base, rbtx, rs_pipe, robot2_tcp2cam)
    # print("已将相机点云显示在世界坐标系中")
    
    # 示例2：使用realtime_pointcloud_visualization函数实时更新点云显示
    # 每0.5秒更新一次点云
    # 相机到TCP的转换矩阵
    d405_tcp2cam_matrix_pos = np.array([-0.020838, -0.022082, 0.08223])
    d405_tcp2cam_matrix_rotmat = np.array([[1, 0.0, 0.0],
                                      [0.0, 1, 0.0],
                                      [0.0, 0.0, 1.0]])
    d405_tcp2cam_matrix = rm.homomat_from_posrot(d405_tcp2cam_matrix_pos, d405_tcp2cam_matrix_rotmat)
    
    # task_id = realtime_pointcloud_visualization(base, rbt, rbtx, rs_pipe_d405, d405_tcp2cam_matrix, update_interval=0.1)
    # print("已启动实时点云显示，任务ID:", task_id)
    
    # 定义AR标记ID列表和尺寸
    ar_marker_ids = [0, 1, 2, 3]  # 使用4个AR标记
    marker_size = 0.085  # 标记尺寸为8厘米
    
    # 计算D435在机器人基坐标系下的位姿
    # world2d435_matrix = calibrate_environment_camera(
    #     robot_interface=rbtx,
    #     tcp_camera_interface=rs_pipe_d405,
    #     env_camera_interface=rs_pipe_d435,
    #     tcp2cam_matrix=d405_tcp2cam_matrix,
    #     ar_marker_ids=ar_marker_ids,
    #     marker_size=marker_size,
    #     num_samples=3,  # 每个标记采样3次
    #     visualize=True,
    #     base=base
    # )
    
    world2d435_matrix = np.array([[-0.05629431854175559, 0.8789956408417414, -0.47349510354504803, 0.2406849734831277], 
                                  [0.9974927606788353, 0.02914481594138041, -0.06448854237039894, 0.4399293462223976], 
                                  [-0.04288521998585332, -0.4759382765495538, -0.8784324759603357, 0.5448600598464484], 
                                  [0.0, 0.0, 0.0, 1.0]])
   
    # 同时显示TCP相机和环境相机的点云
    dual_task_id = realtime_dual_camera_visualization(
        base=base,
        robot_s=rbt,
        robot_interface=rbtx,
        tcp_camera_interface=rs_pipe_d405,
        env_camera_interface=rs_pipe_d435,
        tcp2cam_matrix=d405_tcp2cam_matrix,
        world2env_cam_matrix=world2d435_matrix,
        update_interval=0.2  # 每0.2秒更新一次
    )
    print(f"已启动双相机实时点云显示，任务ID: {dual_task_id}")

    
    base.run()
    rbtx.StopDrag()
    rbtx.DisableRobot()