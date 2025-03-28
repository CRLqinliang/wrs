import os
import time
import numpy as np
import networkx as nx
from wrs import wd, rm, mcm, ur3ed, fsp, fsreg, gg, mgm
from wrs.robot_sim.robots.cobotta import cobotta_wrstweezer as cbtwt

mesh_name = "part_b"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name + ".stl")
fsref_pose_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_fsref_pose.pickle")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_grasp.pickle")
regspot_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_regspot.pickle")

base = wd.World(cam_pos=rm.vec(0, -1, 2), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame(pos=[0,0,0], rotmat=rm.rotmat_from_euler(0,0,0)).attach_to(base)
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, .01), pos=rm.vec(0, 0, -0.005))
ground.attach_to(base)
obj = mcm.CollisionModel(mesh_path)
robot = cbtwt.CobottaTweezer()
jnv_value = np.array([-1.22, -6.78, 93.4, 97.4, -59.06, -41.21]) * np.pi / 180
robot.goto_given_conf(jnv_value)
robot.gen_meshmodel(alpha=.7, toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name=fsref_pose_path)
reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
fsreg_planner = fsreg.FSRegraspPlanner(robot=robot,
                                       obj_cmodel=obj,
                                       fs_reference_poses=fs_reference_poses,
                                       reference_gc=reference_grasps)
fsreg_planner.add_fsregspot_collection_from_disk(regspot_path)

start_pose = (fs_reference_poses[0][0] + rm.np.array([.3, -.1, 0.005]), fs_reference_poses[0][1])
goal_pose = (fs_reference_poses[2][0] + rm.np.array([.3, -.2, 0.005]), fs_reference_poses[2][1])

obj_start = obj.copy()
obj_start.rgb = rm.const.red
obj_start.pose = start_pose
obj_start.attach_to(base)
obj_goal = obj.copy()
obj_goal.rgb = rm.const.green
obj_goal.pose = goal_pose
obj_goal.attach_to(base)
spot_pos0 = rm.np.array([.2, .0, .005])
spot_pos1 = rm.np.array([.25, .0, .005])
spot_pos2 = rm.np.array([.3, 0, .005])
mgm.gen_sphere(pos=spot_pos0).attach_to(base)
mgm.gen_sphere(pos=spot_pos1).attach_to(base)
mgm.gen_sphere(pos=spot_pos2).attach_to(base)
# base.run()
result = fsreg_planner.plan_by_obj_poses(start_pose=start_pose, goal_pose=goal_pose, obstacle_list=[ground], toggle_dbg=False)
if result is None:
    print("No solution found.")
    exit()

# start_node_list = fsreg_planner.add_start_pose(obj_pose=start_pose, obstacle_list=[])
# goal_node_list = fsreg_planner.add_goal_pose(obj_pose=goal_pose, obstacle_list=[])
#
#
# min_path = None
# for start in start_node_list:
#     for goal in goal_node_list:
#         path = nx.shortest_path(fsreg_planner._graph, source=start, target=goal)
#         min_path = path if min_path is None else path if len(path) < len(min_path) else min_path
#
# print(min_path)
# # fsreg_planner.show_graph_with_path(min_path)
#
# result = fsreg_planner.gen_regrasp_motion(path=min_path, obstacle_list=[], linear_distance=.15)
# print(result)
# mesh_model_list = []
# if result[0] == "success":
#     mesh_model_list = result[1]


class Data(object):
    def __init__(self, mesh_model_list):
        self.counter = 0
        self.mesh_model_list = mesh_model_list

ev_list = result.ev_list
jnv_list = result.jv_list
np.save("./ev_list.pickle", ev_list)
np.save("./jnv_list.pickle", jnv_list)

anime_data = Data(mesh_model_list=result.mesh_list)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_model_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mesh_model_list):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    anime_data.mesh_model_list[anime_data.counter].attach_to(base)
    # anime_data.mesh_model_list[anime_data.counter].show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    # time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
