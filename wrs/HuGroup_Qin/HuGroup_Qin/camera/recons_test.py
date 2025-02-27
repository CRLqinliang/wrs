import open3d as o3d
import numpy as np
import cv2

def generate_point_cloud(rgb_image, depth_image, intrinsic_matrix):
    # 获取图像的高和宽
    height, width = depth_image.shape

    # 将RGB图像和深度图像转为numpy数组
    rgb = np.asarray(rgb_image)
    # 处理深度图像，假设其已归一化到0-255，并以16位保存，将其转换回实际深度值（单位：米）
    depth = (np.asarray(depth_image).astype(np.float32) / 255.0) * 10.0  # 假设深度最大为10米

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
    intrinsic.set_intrinsics(width, height, intrinsic_matrix[0, 0], intrinsic_matrix[1, 1],
                             intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])

    # 从RGBD图像生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # 将点云翻转，使其与通常的视觉坐标系对齐
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def main():
    # 相机内参矩阵 (fx, fy, cx, cy)
    fx, fy, cx, cy = 428.365143, 428.365143, 424.40979, 241.511627
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # 输入RGB和深度图像的路径
    rgb_path = 'd405_rgb_img2_Color.png'
    depth_path = 'depth_refined_image_DA2.png'

    # 读取RGB图像
    rgb_image = cv2.imread(rgb_path)

    # 读取16位PNG格式的深度图像
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_numpy_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
    depth_numpy_normalized = depth_numpy_normalized.astype('uint16')

    # 生成点云
    point_cloud = generate_point_cloud(rgb_image, depth_numpy_normalized, intrinsic_matrix)

    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud], window_name='Point Cloud Viewer', width=848, height=480)


if __name__ == '__main__':
    # main()
    point_cloud = o3d.io.read_point_cloud("H:\Qin\wrs\wrs\HuGroup_Qin\camera\d405_rgb_img2_Color.ply")
    o3d.visualization.draw_geometries([point_cloud])