""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240620 Osaka Univ.

"""
import numpy as np
from robot_sim.manipulators.dobot_nova2.nova2 import Nova2


def nova2_ik(tgt_pos: np.ndarray, tgt_rotmat: np.ndarray):

    # DH parameters of nova2
    a2 = -0.280
    a3 = -0.22501
    d1 = 0.2234
    d4 = 0.1175
    d5 = 0.120
    d6 = 0.088004

    n = tgt_rotmat[:, 0]
    o = tgt_rotmat[:, 1]
    a = tgt_rotmat[:, 2]
    p = tgt_pos

    q = np.zeros((8, 6))
    m1 = d6 * a[1] - p[1]
    n1 = d6 * a[0] - p[0]
    k = m1**2 + n1**2 - d4**2
    if -1e-8 < k < 0:
        k = 0
    for index in range(4):
        q[index][0] = np.arctan2(m1, n1) - np.arctan2(d4, np.sqrt(k))
        q[index + 4][0] = np.arctan2(m1, n1) - np.arctan2(d4, -np.sqrt(k))
    for index in range(4):
        q5 = np.arccos(a[0] * np.sin(q[2 * index + 1][0]) - a[1] * np.cos(q[2 * index + 1][0]))
        if index % 2 == 0:
            q[2 * index][4] = q5
            q[2 * index + 1][4] = q5
        else:
            q[2 * index][4] = -q5
            q[2 * index + 1][4] = -q5
    for index in range(8):
        m6 = n[0] * np.sin(q[index][0]) - n[1] * np.cos(q[index][0])
        n6 = o[0] * np.sin(q[index][0]) - o[1] * np.cos(q[index][0])
        q[index][5] = np.arctan2(m6, n6) - np.arctan2(np.sin(q[index][4]), 0)
        m3 = d5 * (np.sin(q[index][5]) * (n[0] * np.cos(q[index][0]) + n[1] * np.sin(q[index][0]))
                   + np.cos(q[index][5]) * (o[0] * np.cos(q[index][0]) + o[1] * np.sin(q[index][0]))) \
             + p[0] * np.cos(q[index][0]) + p[1] * np.sin(q[index][0]) - d6 * (a[0] * np.cos(q[index][0]) + a[1] * np.sin(q[index][0]))
        n3 = p[2] - d1 - a[2] * d6 + d5 * (o[2] * np.cos(q[index][5]) + n[2] * np.sin(q[index][5]))
        k3 = (m3**2 + n3**2 - a2**2 - a3**2) / (2 * a2 * a3)
        if k3 - 1 > 1e-6 or k3 + 1 < -1e-6:
            q3 = np.nan
        elif 0 <= k3 - 1 <= 1e-6:
            q3 = 0
        elif 0 <= k3 + 1 < 1e-6:
            q3 = np.pi
        else:
            q3 = np.arccos(k3)
        q[index][2] = q3 if index % 2 == 0 else -q3
        s2 = ((a3 * np.cos(q[index][2]) + a2) * n3 - a3 * np.sin(q[index][2]) * m3) / \
             (a2**2 + a3**2 + 2 * a2 * a3 * np.cos(q[index][2]))
        c2 = (m3 + a3 * np.sin(q[index][2]) * s2) / (a3 * np.cos(q[index][2]) + a2)
        q[index][1] = np.arctan2(s2, c2)
        s234 = -np.sin(q[index][5]) * (n[0] * np.cos(q[index][0]) + n[1] * np.sin(q[index][0])) -\
               np.cos(q[index][5]) * (o[0] * np.cos(q[index][0]) + o[1] * np.sin(q[index][0]))
        c234 = o[2] * np.cos(q[index][5]) + n[2] * np.sin(q[index][5])
        q[index][3] = np.arctan2(s234, c234) - q[index][1] - q[index][2]

    # ur5 -> nova2
    q[:, 1] = q[:, 1] + np.ones(8) * np.pi/2
    q[:, 3] = q[:, 3] + np.ones(8) * np.pi/2
    q[:, 0] = -q[:, 0]
    q[:, 4] = -q[:, 4]
    q[:, 3] = -q[:, 3]

    for index_i in range(8):
        for index_j in range(6):
            if q[index_i][index_j] < -np.pi:
                q[index_i][index_j] += 2 * np.pi
            elif q[index_i][index_j] >= np.pi:
                q[index_i][index_j] -= 2 * np.pi

    # 复制数据到Q_result并移除无效解
    Q_result = np.zeros((8, 6))
    for temp_row in range(8):
        for temp_col in range(6):
            Q_result[temp_row, temp_col] = q[temp_row, temp_col]

    # 过滤无效解
    Result = []
    for i in range(8):
        if not np.isnan(Q_result[i, 2]):
            Result.append(Q_result[i, :])
    Result = np.array(Result)

    if Result.shape[0] == 0:
        raise ValueError("No valid solutions found")

    return Result


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    robot = Nova2(enable_cc=True)
    rand_conf = robot.rand_conf()
    print("random conf: {}".format(rand_conf))

    # iteration solution
    fk_sol = robot.fk(rand_conf)
    print("fk sol: {}, \n{}".format(fk_sol[0], fk_sol[1]))
    print("----------------------------------------")

    temp_pos = np.array([0.48000, -0.45000, 0.20000])
    temp_romat = np.array([[1,0,0],[0,0,1],[0,1,0]])

    # ik solution
    time_start = time.time()
    ik_sol_analysis = nova2_ik(temp_pos, temp_romat)
    time_end = time.time()
    print("ik time: {:.12f}ms".format((time_end - time_start) * 1000.0))
    for i in np.arange(ik_sol_analysis.shape[0]):
        print("ik {} sol: {}".format(i, ik_sol_analysis[i, :]))
        robot.goto_given_conf(ik_sol_analysis[i, :])
        robot.gen_meshmodel(alpha=0.3).attach_to(base)
    base.run()