"""
ERTConnect算法可视化示例
Author: Liang Qin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from ERTConnect_python import ERTConnect, Motion
import time

class Environment2D:
    """2D环境类"""
    def __init__(self):
        self.obstacles = []
        
    def add_circle_obstacle(self, center, radius):
        """添加圆形障碍物"""
        self.obstacles.append(('circle', center, radius))
        
    def add_rectangle_obstacle(self, bottom_left, width, height):
        """添加矩形障碍物"""
        self.obstacles.append(('rectangle', bottom_left, width, height))
        
    def is_valid(self, point):
        """检查点是否有效（不与障碍物碰撞）"""
        x, y = point
        for obs in self.obstacles:
            if obs[0] == 'circle':
                center, radius = obs[1], obs[2]
                if np.linalg.norm(point - center) <= radius:
                    return False
            elif obs[0] == 'rectangle':
                bl, w, h = obs[1], obs[2], obs[3]
                if (bl[0] <= x <= bl[0] + w and 
                    bl[1] <= y <= bl[1] + h):
                    return False
        return True
        
    def check_path(self, p1, p2, resolution=0.01):
        """检查路径是否有效（不与障碍物碰撞）"""
        vec = p2 - p1
        distance = np.linalg.norm(vec)
        
        if distance < resolution:
            return self.is_valid(p1) and self.is_valid(p2)
        
        steps = max(int(distance / resolution), 1)
        
        for i in range(steps + 1):
            t = i / steps
            point = p1 + t * vec
            if not self.is_valid(point):
                return False
        return True

    def visualize(self, ax=None):
        """可视化环境"""
        if ax is None:
            ax = plt.gca()
            
        for obs in self.obstacles:
            if obs[0] == 'circle':
                center, radius = obs[1], obs[2]
                circle = Circle(center, radius, color='gray', alpha=0.5)
                ax.add_patch(circle)
            elif obs[0] == 'rectangle':
                bl, w, h = obs[1], obs[2], obs[3]
                rect = Rectangle(bl, w, h, color='gray', alpha=0.5)
                ax.add_patch(rect)

class SimpleRRTPlanner:
    """简单的RRT规划器，用于生成经验路径"""
    def __init__(self, env, start, goal, step_size=0.1, max_iter=10000):
        self.env = env
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.max_iter = max_iter
        # 使用Motion类创建起始节点
        self.nodes = [Motion(start, 0.0)]
        
    def random_state(self):
        """生成随机状态"""
        return np.random.uniform(-2, 2, 2)
        
    def nearest_node(self, point):
        """找到最近的节点"""
        distances = [np.linalg.norm(node.state - point) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
        
    def steer(self, from_point, to_point):
        """控制步长"""
        vec = to_point - from_point
        dist = np.linalg.norm(vec)
        if dist > self.step_size:
            vec = vec / dist * self.step_size
        return from_point + vec
        
    def plan(self):
        """生成路径"""
        for _ in range(self.max_iter):
            rand_point = self.random_state()
            nearest = self.nearest_node(rand_point)
            new_point = self.steer(nearest.state, rand_point)
            
            if self.env.check_path(nearest.state, new_point):
                # 计算新节点的相位
                phase = (len(self.nodes) + 1) / self.max_iter
                new_node = Motion(new_point, phase, parent=nearest)
                new_node.segment = [new_point]
                self.nodes.append(new_node)
                
                if np.linalg.norm(new_point - self.goal) < self.step_size:
                    if self.env.check_path(new_point, self.goal):
                        final_node = Motion(self.goal, 1.0, parent=new_node)
                        final_node.segment = [self.goal]
                        self.nodes.append(final_node)
                        return self.extract_path(final_node)
        return None
        
    def extract_path(self, node):
        """提取路径"""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return np.array(path[::-1])

def create_complex_2d_environment():
    """创建复杂的2D示例环境"""
    env = Environment2D()
    
    # 添加圆形障碍物
    env.add_circle_obstacle(np.array([0.0, 0.0]), 0.3)
    env.add_circle_obstacle(np.array([1.0, 1.0]), 0.3)
    env.add_circle_obstacle(np.array([-1.0, -1.0]), 0.3)
    env.add_circle_obstacle(np.array([0.5, -0.5]), 0.2)
    env.add_circle_obstacle(np.array([-0.5, 0.5]), 0.2)
    
    # 添加矩形障碍物
    env.add_rectangle_obstacle(np.array([-1.5, 0.2]), 1.0, 0.2)
    env.add_rectangle_obstacle(np.array([0.5, -0.2]), 1.0, 0.2)
    env.add_rectangle_obstacle(np.array([-0.2, -1.5]), 0.2, 1.0)
    env.add_rectangle_obstacle(np.array([0.2, 0.5]), 0.2, 1.0)
    
    return env

def main():
    # 创建环境
    env = create_complex_2d_environment()
    
    # 定义起点和终点
    start = np.array([-1.5, -1.5])
    goal = np.array([1.5, 1.5])
    
    # 首先使用RRT生成经验路径
    print("使用RRT生成经验路径...")
    rrt_planner = SimpleRRTPlanner(env, start, goal, step_size=0.1)
    experience = rrt_planner.plan()
    
    if experience is None:
        print("无法生成有效的经验路径！")
        return
    
    print("成功生成经验路径，开始ERTConnect规划...")
    
    # 创建ERTConnect规划器
    planner = ERTConnect(
        start=start,
        goal=goal,
        experience=experience,
        env=env,
        space_dim=2,
        epsilon=0.5,  # 变形系数
        omega=(0.05, 0.1)  # 相位跨度范围
    )
    
    # 执行规划
    start_time = time.time()
    path = planner.plan(max_iter=2000)
    planning_time = time.time() - start_time
    print(f"规划完成，用时: {planning_time:.6f}秒")
    
    # 可视化结果
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # 设置坐标轴范围
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    # 绘制环境
    env.visualize(ax)
    
    # 绘制经验路径
    plt.plot(experience[:,0], experience[:,1], 'y--', label='Experience Path', alpha=0.5)
    
    # 使用ERTConnect的可视化方法
    if path is not None:
        print("成功找到路径！")
        planner.visualize(path)
    else:
        print("未找到可行路径。")
        planner.visualize()
    
    plt.grid(True)
    plt.title('ERTConnect Path Planning')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()