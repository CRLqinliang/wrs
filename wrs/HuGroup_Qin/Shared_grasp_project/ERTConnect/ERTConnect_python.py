import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

class Motion:
    def __init__(self, state, phase_end, parent=None):
        self.state = np.array(state)      # 状态向量
        self.phase_end = phase_end        # 相位终点
        self.parent = parent              # 父节点
        self.segment = []                 # 路径片段
        self.selection_count = 0          # 选择次数统计

class ERTConnect:
    def __init__(self, start, goal, experience, env, space_dim=2, 
                 epsilon=1.0, omega=(0.05, 0.1)):
        # 初始化参数
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.experience = experience      # 经验路径（状态列表）
        self.space_dim = space_dim
        self.epsilon = epsilon            # 变形系数
        self.omega_min, self.omega_max = omega
        self.env = env
        
        # 初始化双树结构
        self.t_start = []  # 存储Motion节点的列表
        self.t_goal = []   # 存储Motion节点的列表
        
        # 概率参数
        self.connect_prob = 0.05
        
        # 映射经验路径
        self.mapped_experience = self.map_experience()

    def map_experience(self):
        """将经验路径映射到当前问题空间"""
        mapped = []
        start_exp = self.experience[0]
        goal_exp = self.experience[-1]
        
        # 计算仿射变换参数
        b = self.start - start_exp
        l = self.goal - (goal_exp + b)
        
        # 应用变换
        for i, s in enumerate(self.experience):
            t = i / (len(self.experience)-1)
            new_state = s + b + t*l
            mapped.append(new_state)
        return mapped

    def generate_segment(self, q_init, direction='forward', q_target=None):
        """改进后的路径生成方法（支持双向扩展）"""
        alpha_init = q_init.phase_end
        
        if q_target is not None:  # 连接模式
            alpha_target = q_target.phase_end
            phase_span = abs(alpha_target - alpha_init)
            
            # 提取微经验
            seg_start = int(alpha_init * len(self.mapped_experience))
            seg_end = int(alpha_target * len(self.mapped_experience))
            micro_exp = self.mapped_experience[seg_start:seg_end+1]
            
            # 计算变换参数
            b = q_init.state - micro_exp[0]
            l = q_target.state - (micro_exp[-1] + b)
        else:  # 探索模式
            # 根据方向采样相位跨度
            if direction == 'forward':
                phase_span = np.random.uniform(self.omega_min, self.omega_max)
                alpha_target = min(alpha_init + phase_span, 1.0)
            else:  # backward
                phase_span = np.random.uniform(self.omega_min, self.omega_max)
                alpha_target = max(alpha_init - phase_span, 0.0)
            
            # 根据方向确定经验路径采样方向
            if direction == 'forward':
                seg_start = int(alpha_init * len(self.mapped_experience))
                seg_end = int(alpha_target * len(self.mapped_experience))
            else:
                seg_start = int(alpha_target * len(self.mapped_experience))
                seg_end = int(alpha_init * len(self.mapped_experience))
            
            micro_exp = self.mapped_experience[seg_start:seg_end+1]
            
            # 保证路径顺序与扩展方向一致
            if direction == 'backward':
                micro_exp = micro_exp[::-1]
            
            b = q_init.state - micro_exp[0]
            l = np.random.uniform(-self.epsilon, self.epsilon, self.space_dim)
        
        # 生成变形后的路径（保持原有逻辑）
        new_segment = []
        for i, s in enumerate(micro_exp):
            t = i / (len(micro_exp)-1)
            new_state = s + b + t*l
            new_segment.append(new_state)
        
        return new_segment, alpha_target

    def is_valid(self, segment):
        """使用环境对象进行碰撞检测
        参数:
            segment: 路径段（状态点数组）
        返回:
            bool: 路径段是否有效（无碰撞）
        """
        # 检查路径段中的每个状态点
        for i in range(len(segment)):
            # 检查当前点是否有效
            if not self.env.is_valid(segment[i]):
                return False
            
            # 检查相邻点之间的路径是否有效
            if i > 0:
                # 使用环境的check_path方法检查连续路径
                if not self.env.check_path(segment[i-1], segment[i]):
                    return False
        
        return True

    def plan(self, max_iter=1000):
        # 初始化双树
        start_node = Motion(self.start, 0.0)
        goal_node = Motion(self.goal, 1.0)
        self.t_start.append(start_node)
        self.t_goal.append(goal_node)
        
        for _ in range(max_iter):
            # 交替扩展双树并传递方向参数
            for tree, direction in [(self.t_start, 'forward'), 
                                  (self.t_goal, 'backward')]:
                
                # 选择节点（加权随机选择）
                weights = [1/(n.selection_count+1) for n in tree]
                selected = np.random.choice(tree, p=weights/np.sum(weights))
                selected.selection_count += 1
                
                # 生成路径片段（添加方向参数）
                if np.random.rand() < self.connect_prob and tree is self.t_start:
                    segment, alpha = self.generate_segment(selected, direction, self.t_goal[0])
                else:
                    segment, alpha = self.generate_segment(selected, direction)
                
                if self.extend(tree, segment, selected):
                    # 如果一方到达目标，则返回该树
                    if direction == 'forward':
                        if np.allclose(segment[-1], self.goal, atol=1e-3):
                            return self.extract_path(selected, goal_node)
                    else:
                        if np.allclose(segment[-1], self.start, atol=1e-3):
                            return self.extract_path(start_node, selected)
                    
                    # 尝试连接双树
                    other_tree = self.t_goal if tree is self.t_start else self.t_start
                    nearest = self.find_nearest(selected.state, other_tree)

                    segment, alpha = self.generate_segment(nearest, 
                                          'forward' if direction == 'backward' else 'backward',
                                          selected)
                    
                    # 检查连接
                    if self.extend(other_tree, segment, nearest):
                        return self.extract_path(selected, nearest)
        
        return None  # 未找到路径

    def find_nearest(self, state, tree):
        """查找最近邻节点"""
        states = [n.state for n in tree]
        kdtree = KDTree(states)
        dist, idx = kdtree.query(state)
        return tree[idx]

    def extend(self, tree, segment, start_node):
        """扩展树结构
        参数:
            tree: Motion对象的列表
            segment: 路径段（状态点数组）
            parent_node: 父节点（Motion对象）
        返回:   
            bool: 扩展是否成功
        """
       
        # 检查路径段是否有效(碰撞检测)
        if not self.is_valid(segment):
            return False
        
        # 计算相位增量
        phase_increment = 1.0 / len(self.mapped_experience)
        current_parent = start_node
        
        # 为路径段中的每个状态点创建Motion对象
        for state in segment:   
            # 计算新的相位
            if tree is self.t_start:
                new_phase = current_parent.phase_end + phase_increment
            else:
                new_phase = current_parent.phase_end - phase_increment
            
            # 创建新的Motion节点
            new_node = Motion(state, new_phase, parent=current_parent)
            new_node.segment = [state]  # 保存当前状态点
            
            # 将新节点添加到树中
            tree.append(new_node)
            
            # 更新父节点为当前节点，为下一个状态点做准备
            current_parent = new_node
        
        return True

    def check_connection(self, node_a, node_b):
        """检查两节点是否可连接"""
        segment = self.interpolate(node_a.state, node_b.state)
        return self.is_valid(segment)

    def extract_path(self, node_a, node_b):
        """提取完整路径"""
        path = []
        
        # 反向追溯起点树
        current = node_a
        while current:
            path.insert(0, current.segment)
            current = current.parent
        
        # 正向追溯终点树
        current = node_b
        while current:
            path.append(current.segment[::-1])  # 反转顺序
            current = current.parent
        
        return np.concatenate(path)

    def visualize(self, path=None):
        """可视化结果"""
        plt.figure(figsize=(10, 6))
        
        # 绘制经验路径
        exp = np.array(self.mapped_experience)
        plt.plot(exp[:,0], exp[:,1], 'y--', label="Mapped Experience")
        
        # 绘制树结构
        for node in self.t_start + self.t_goal:
            if node.parent:
                seg = np.array(node.parent.segment + node.segment)
                plt.plot(seg[:,0], seg[:,1], 'g-', alpha=0.3)
        
        # 绘制最终路径
        if path is not None:
            plt.plot(path[:,0], path[:,1], 'r-', linewidth=2, label="Solution Path")
        
        plt.scatter(self.start[0], self.start[1], c='g', s=100, label="Start")
        plt.scatter(self.goal[0], self.goal[1], c='r', s=100, label="Goal")
        plt.legend()
        plt.grid(True)
        plt.show()


# 测试用例
if __name__ == "__main__":
    # 示例经验路径（简单直线）
    experience = [np.array([1,1]), np.array([5,5]), np.array([9,9])]
    
    # 初始化规划器
    ert = ERTConnect(start=[2,2], goal=[8,8], 
                    experience=experience, space_dim=2)
    
    # 执行规划
    path = ert.plan(max_iter=500)
    
    # 可视化
    if path is not None:
        ert.visualize(path)
    else:
        print("No path found!")