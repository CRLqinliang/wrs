""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240623 Osaka Univ.

"""
"""
This is an example to
1. generate the linear motion for the simulation robot
2. RRT to move a object from one place to the other place
(Run the 4_define_grasp.py First)
"""
import basis.robot_math as rm
import modeling.collision_model as cm
import numpy as np
import visualization.panda.world as wd
import wrs.HuGroup_Qin.utils.file_sys as fs
import wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_pipette_dual as nova2_wrsv3gripper_dual
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc

# create the virtual environment
base = wd.World(cam_pos=[0.55, -3, 0.35], lookat_pos=[0.55, 0, 0.35])

# generate the nova2 dual robot platform
hugroup_lab_nova2_dual = nova2_wrsv3gripper_dual.hugroup_lab_nova2_dual(enable_cc=True)

rgt_init_jnt_values = np.array([2.02, 14.16, 56.78, -96.41, -66.86, 94.23])
lft_init_jnt_values = np.array([-12.93, -24.28, -93.69, 3.67, 21.95, -0.22])
hugroup_lab_nova2_dual.rgt_arm.goto_given_conf(jnt_values=np.radians(rgt_init_jnt_values))
hugroup_lab_nova2_dual.lft_arm.goto_given_conf(jnt_values=np.radians(lft_init_jnt_values))

# generate the tube to be grasped and define the initial and target pose for the object
obj_mdl = cm.CollisionModel(initor="H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl")
obj_mdl.pos = np.array([0.25, -.14, 0.1])
obj_mdl.rotmat = rm.rotmat_from_euler(0, np.pi/2, 0)
obj_mdl.show_local_frame()
obj_mdl.attach_to(base)

base.run()
# target pose of the pipe
obj_mdl_tgt_pose = rm.homomat_from_posrot(np.array([0.25, -.3, 0.21]),
                                          rm.rotmat_from_euler(0, np.pi/2, 0))

# initialize the module to generate the linear motion
inik_svlr = inik.IncrementalNIK(robot_s=hugroup_lab_nova2_dual.lft_arm)

# initialize the module for RRT
rrtc_planner = rrtc.RRTConnect(hugroup_lab_nova2_dual.lft_arm)

# load the grasp poses for the tube
grasps_collection = fs.load_pickle("/wrs/HuGroup_Qin\data\grasps\pipe_50ml_grasps.pkl")

# the list to save the path
path = []
hugroup_lab_nova2_dual.use_rgt()

# search the grasp that can move the object to target pose
for ind, grasp in enumerate(grasps_collection):
    print(f"--------------------- grasp pose index: {ind} ---------------------------")
    # build a homogenous matrix (a 4x4 matrix with transition and orientation)
    obj_pose = obj_mdl.homomat
    grasp_pose = rm.homomat_from_posrot(grasp.ac_pos, grasp.ac_rotmat)
    print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")

    # solve the IK for the initial pose of the pipe
    rbt_ee_pose_init = np.dot(obj_pose, grasp_pose)
    ik_sol_init = hugroup_lab_nova2_dual.ik(tgt_pos=rbt_ee_pose_init[:3, 3],
                            tgt_rotmat=rbt_ee_pose_init[:3, :3])

    # solve the IK for the target pose of the pipe
    rbt_ee_pose_tgt = np.dot(obj_mdl_tgt_pose, grasp_pose)
    ik_sol_tgt = hugroup_lab_nova2_dual.ik(tgt_pos=rbt_ee_pose_tgt[:3, 3],
                           tgt_rotmat=rbt_ee_pose_tgt[:3, :3],
                           seed_jnt_values=ik_sol_init)

    if ik_sol_init is not None and ik_sol_tgt is not None:  # check IK-feasible
        hugroup_lab_nova2_dual.fk(ik_sol_init)
        is_self_collided_init = hugroup_lab_nova2_dual.is_collided()
        is_self_collided_tgt = hugroup_lab_nova2_dual.is_collided()

        if is_self_collided_init or is_self_collided_tgt:  # check if is self-collided
            print(">>> The robot is self-collided")
            continue
        else:
            # generate the motion to raise the tube up by the robot, first move to the init pose of object;
            hugroup_lab_nova2_dual.lft_arm.goto_given_conf(ik_sol_init)
            rbt_tcp_pos, rbt_tcp_rot = hugroup_lab_nova2_dual.lft_arm.gl_tcp_pos, \
                                        hugroup_lab_nova2_dual.lft_arm.gl_tcp_rotmat

            obj_mdl_grasped = obj_mdl.copy()
            hugroup_lab_nova2_dual.lft_arm.hold(obj_cmodel=obj_mdl_grasped, jaw_width=grasp.ee_values)
            path_up = inik_svlr.gen_linear_motion(start_tcp_pos=rbt_tcp_pos,
                                                  start_tcp_rotmat=rbt_tcp_rot,
                                                  goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .05]),
                                                  goal_tcp_rotmat=rbt_tcp_rot,
                                                  granularity=0.01)
            if path_up is not None:
                # generate the motion to move the tube to the target pose
                rrt_path = rrtc_planner.plan(start_conf=np.array(path_up[-1]),
                                             goal_conf=np.array(ik_sol_tgt),
                                             obstacle_list=[],
                                             other_robot_list=[],
                                             ext_dist=.05,
                                             max_time=300)

                # generate the motion to move the tube to the init pose
                final_path = rrtc_planner.plan(start_conf=np.array(rrt_path.jv_list[-1]),
                                               goal_conf=np.array(path_up[0]),
                                               obstacle_list=[],
                                               other_robot_list=[],
                                               ext_dist=.05,
                                               max_time=300)

                if rrt_path and final_path is not None:
                    # show the path
                    path = path_up + rrt_path.jv_list + final_path.jv_list
                    for jnts_s in path:
                        hugroup_lab_nova2_dual.lft_arm.goto_given_conf(jnt_values=jnts_s)
                        hugroup_lab_nova2_dual.gen_meshmodel(alpha=.8).attach_to(base)
                        hugroup_lab_nova2_dual.gen_stickmodel().attach_to(base)
                        hugroup_lab_nova2_dual.show_cdprim()
                        obj_mdl_draw = obj_mdl_grasped.copy()
                        obj_mdl_draw.attach_to(base)
                    base.run()
                else:
                    print(">>> Cannot generate the path to move the object to the target pose by RRT")
                    hugroup_lab_nova2_dual.lft_arm.release(obj_cmodel=obj_mdl_grasped)
            else:
                print(">>> Cannot generate the path to raise the tube")
                hugroup_lab_nova2_dual.lft_arm.release(obj_cmodel=obj_mdl_grasped)
    else:
        print("No IK solution at init or target")
exit(-1)
