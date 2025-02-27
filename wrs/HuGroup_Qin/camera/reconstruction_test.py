import open3d as o3d
import numpy as np
import cv2

def generate_point_cloud(rgb_image, depth_image, intrinsic_matrix):
    # 获取图像的高和宽
    height, width = depth_image.shape

    # 将RGB图像和深度图像转为numpy数组
    rgb = np.asarray(rgb_image)
    depth = np.asarray(depth_image).astype(np.float32) / 1000.0  # 将16位深度图转换为米，假设深度以毫米为单位

    # 使用Open3D创建RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,  # 深度已转换为米，无需额外缩放
        depth_trunc=10.0  # 设置深度截断距离，防止无效数据
    )

    # 使用相机内参矩阵创建相机内参对象
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    # 从RGBD图像生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # 将点云翻转，使其与通常的视觉坐标系对齐
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

# 读取RGB和深度图像
rgb_image = cv2.imread('d405_rgb_img2_Color.png')

# 读取16位.raw格式的深度图像
depth_image = np.fromfile('d405_2_Depth.raw', dtype=np.uint16).reshape((480, 848))

# 定义相机的内参矩阵（fx, fy 为焦距，cx, cy 为光轴在图像上的位置）
fx, fy, cx, cy = 428.365143, 428.365143, 424.40979, 241.511627
intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 生成点云
pcd = generate_point_cloud(rgb_image, depth_image, intrinsic_matrix)

# 可视化点云（允许拖动和缩放）
o3d.visualization.draw_geometries([pcd])
