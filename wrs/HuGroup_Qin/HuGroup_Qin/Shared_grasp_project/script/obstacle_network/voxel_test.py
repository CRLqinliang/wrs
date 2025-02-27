import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_occupied_voxels(mesh, origin, voxel_size, shape):
    """
    计算物体在指定网格空间中占据的体素索引。

    参数：
        mesh (trimesh.Trimesh): 三维网格对象
        origin (np.ndarray): 网格空间原点，形如[x, y, z]
        voxel_size (float): 体素边长
        shape (tuple): 网格维度，形如(nx, ny, nz)

    返回：
        occupied_indices (np.ndarray): 被占据体素的索引数组，每行为[x_idx, y_idx, z_idx]
    """
    # 复制网格以避免修改原始数据
    mesh = mesh.copy()

    # 将物体的包围盒最小角移至网格空间原点
    current_min = mesh.bounds[0]
    translation = origin - current_min
    mesh.apply_translation(translation)

    # 执行体素化
    try:
        voxel_grid = mesh.voxelized(pitch=voxel_size)
    except Exception as e:
        print(f"体素化失败: {e}")
        return np.array([])

    # 获取被占据的体素索引
    if not voxel_grid.is_empty:
        occupied_indices = voxel_grid.sparse_indices
        # 过滤超出网格范围的索引
        in_bounds = np.all((occupied_indices >= 0) & (occupied_indices < np.array(shape)), axis=1)
        return occupied_indices[in_bounds]
    else:
        return np.array([])


def visualize_voxels(mesh, occupied_indices, voxel_size, origin, shape):
    """
    可视化网格和占据的体素。

    参数：
        mesh (trimesh.Trimesh): 原始三维网格对象
        occupied_indices (np.ndarray): 被占据体素的索引数组
        voxel_size (float): 体素边长
        origin (np.ndarray): 网格空间原点
        shape (tuple): 网格维度
    """
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制体素
    for idx in occupied_indices:
        # 计算体素的实际位置
        pos = origin + idx * voxel_size

        # 创建体素的顶点
        x, y, z = pos
        vertices = np.array([
            [x, y, z],
            [x+voxel_size, y, z],
            [x+voxel_size, y+voxel_size, z],
            [x, y+voxel_size, z],
            [x, y, z+voxel_size],
            [x+voxel_size, y, z+voxel_size],
            [x+voxel_size, y+voxel_size, z+voxel_size],
            [x, y+voxel_size, z+voxel_size]
        ])

        # 绘制体素的边
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        for start, end in edges:
            ax.plot3D(
                vertices[[start, end], 0],
                vertices[[start, end], 1],
                vertices[[start, end], 2],
                'b-', alpha=0.3
            )

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 设置坐标轴范围
    max_dim = max(shape) * voxel_size
    ax.set_xlim([origin[0], origin[0] + max_dim])
    ax.set_ylim([origin[1], origin[1] + max_dim])
    ax.set_zlim([origin[2], origin[2] + max_dim])

    plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建一个示例物体（单位立方体）
    mesh = trimesh.primitives.Box(extents=[1, 1, 1])

    # 应用任意变换（例如旋转和平移）
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
    mesh.apply_translation([0.5, 0.5, 0.5])  # 平移物体

    # 定义网格空间参数
    origin = np.array([0, 0, 0])  # 原点
    voxel_size = 0.2  # 体素边长
    shape = (10, 10, 10)  # 网格维度

    # 计算被占据的体素
    occupied = get_occupied_voxels(mesh, origin, voxel_size, shape)
    print("被占据的体素索引:\n", occupied)

    # 可视化结果
    visualize_voxels(mesh, occupied, voxel_size, origin, shape)