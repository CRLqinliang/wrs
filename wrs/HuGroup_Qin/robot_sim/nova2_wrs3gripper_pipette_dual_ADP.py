""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240709 Osaka Univ.

"""
from motion.primitives.approach_depart_planner import ADPlanner
import nova2_wrsv3gripper_pipette_dual as dual_arms
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as mgm
import modeling.collision_model as cm
import wrs.HuGroup_Qin.utils.file_sys as fs


def collision_check(robot, jnt_value):
    # check if the robot is self-collided (use_left or use_right)
    current_jnt_values = robot.get_jnt_values()
    robot.goto_given_conf(jnt_values=jnt_value)
    is_collided = robot.is_collided()
    robot.goto_given_conf(jnt_values=current_jnt_values)
    return is_collided


def pipe_cfgrasp_jnv(robot, obj_pose, grasps_collection, reference_grasp_pose_rotmat=None):
    # return collision-free grasp joint value given current obj_pose.
    collision_free_grasp_jnv = []
    for ind, grasp in enumerate(grasps_collection):
        if reference_grasp_pose_rotmat is not None:
            if np.allclose(reference_grasp_pose_rotmat, grasp.ac_rotmat, rtol=1e-3):
                grasp_pose = rm.homomat_from_posrot(grasp.ac_pos, grasp.ac_rotmat)
            else:
                continue
        else:
            grasp_pose = rm.homomat_from_posrot(grasp.ac_pos, grasp.ac_rotmat)
        rbt_ee_pose_init = np.dot(obj_pose, grasp_pose)

        # solve the IK for the initial pose of the pipe : uint: rad
        ik_sol_init = robot.ik(tgt_pos=rbt_ee_pose_init[:3, 3],
                                             tgt_rotmat=rbt_ee_pose_init[:3, :3],
                                             seed_jnt_values=robot.get_jnt_values())
        if ik_sol_init is not None:
            if collision_check(robot, ik_sol_init) is False:
                collision_free_grasp_jnv.append(ik_sol_init)
        if reference_grasp_pose_rotmat is not None and \
                np.allclose(reference_grasp_pose_rotmat, grasp.ac_rotmat, rtol=1e-3):
            break
    if collision_free_grasp_jnv is None:
        print("No collision-free grasp joint value")
    return collision_free_grasp_jnv


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    if base.inputmgr.keymap["space"]:
        anime_data.counter += 1
    return task.again

class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data

if __name__ == '__main__':


    base = wd.World(cam_pos=[3, 0, 0.35], lookat_pos=[0, 0, 0.35])
    mgm.gen_frame().attach_to(base)

    robot = dual_arms.hugroup_lab_nova2_dual(enable_cc=True)
    robot.use_rgt()
    robot.lft_arm.goto_given_conf(jnt_values=np.radians(np.array([0, -54.6387, 68.4399, -102.8725, 90.71, -180])))
    grasps_collection = fs.load_pickle("/wrs/HuGroup_Qin\data\grasps\pipe_50ml_grasps.pkl")

    obj_mdl_1 = cm.CollisionModel(initor="H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl")
    obj_mdl_1.pos = np.array([0.088, -0.22, 0.22])
    obj_mdl_1.rotmat = rm.rotmat_from_euler(-np.pi/2, 0, 0)
    obj_mdl_1.show_local_frame()
    obj_mdl_1.attach_to(base)
    obj_mdl_1_homo = rm.homomat_from_posrot(obj_mdl_1.pos, obj_mdl_1.rotmat)

    obj_mdl_2 = cm.CollisionModel(initor="H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl")
    obj_mdl_2.pos = np.array([0.088, -0.22, 0.14])
    obj_mdl_2.rotmat = rm.rotmat_from_euler(-np.pi/2, 0, 0)
    obj_mdl_2.show_local_frame()
    obj_mdl_2.attach_to(base)
    obj_mdl_2_homo = rm.homomat_from_posrot(obj_mdl_2.pos, obj_mdl_2.rotmat)

    # base.run()
    adp = ADPlanner(robot)

    pipe_grasp_jnv_1 = pipe_cfgrasp_jnv(robot, obj_mdl_1_homo, grasps_collection,
                                        reference_grasp_pose_rotmat=np.dot(rm.rotmat_from_euler(np.pi, 0, 0),
                                                                  rm.rotmat_from_euler(0, -np.pi / 2, 0)))[0]
    pipe_grasp_jnv_2 = pipe_cfgrasp_jnv(robot, obj_mdl_2_homo, grasps_collection,
                                        reference_grasp_pose_rotmat=np.dot(rm.rotmat_from_euler(np.pi, 0, 0),
                                                                  rm.rotmat_from_euler(0, -np.pi / 2, 0)))[0]

    if pipe_grasp_jnv_1 is not None and pipe_grasp_jnv_2 is not None:
        mot_data = adp.gen_depart_approach_with_given_conf(start_jnt_values=pipe_grasp_jnv_1, end_jnt_values=pipe_grasp_jnv_2,
                                                           approach_direction=np.array([-1, 0, 0]), depart_direction=np.array([1, 0, 0]),
                                                           depart_ee_values=.03, approach_ee_values=.01,
                                                           use_rrt=False)
    else:
        print("no feasible solution.")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()

    # Plot each column as a separate curve
    # for i in range(robot.n_dof):
    #     ax.plot(np.asarray(mot_data.jv_list)[:, i], label=f'Column {i + 1}')
    # plt.show()

    anime_data = Data(mot_data)
    taskMgr.doMethodLater(0.03, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()