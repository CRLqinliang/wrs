import random
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.ur3e_dual.ur3e_dual as u3ed
import basis.constant as bc
import motion.probabilistic.rrt_connect as rrtc


class Data(object):
    def __init__(self):
        self.counter = 0
        self.mot_data = None


base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("objects/bunnysim.stl")
object.pos = np.array([.55, -.3, 1.3])
object.rgba = np.array([.5, .7, .3, 1])
object.attach_to(base)
# robot
robot = u3ed.UR3e_Dual()
# robot.use_lft()
# planner
rrtc_planner = rrtc.RRTConnect(robot)

anime_data = Data()


def update(robot, rrtsc_planner, anime_data, task):
    if anime_data.mot_data is not None and anime_data.counter >= len(anime_data.mot_data):
        anime_data.mot_data.mesh_list[- 1].detach()
        anime_data.mot_data = None
        anime_data.counter = 0
    if anime_data.counter == 0:
        value = random.choice([1, 2, 3])
        if value == 1:
            robot.use_both()
        elif value == 2:
            robot.use_lft()
        else:
            robot.use_rgt()
        while True:
            # plan
            start_conf = robot.get_jnt_values()
            robot.goto_given_conf(jnt_values=start_conf)
            goal_conf = robot.rand_conf()
            robot.goto_given_conf(jnt_values=goal_conf)
            mot_data = rrtsc_planner.plan(start_conf=start_conf,
                                          goal_conf=goal_conf,
                                          ext_dist=.1,
                                          max_time=10,
                                          smoothing_n_iter=100)
            if mot_data is not None:
                # print(anime_data.path)
                anime_data.mot_data = mot_data
                mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
                mesh_model.attach_to(base)
                if base.inputmgr.keymap['space']:
                    anime_data.counter += 1
                break
            else:
                continue
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot, rrtc_planner, anime_data],
                      appendTask=True)
base.run()
