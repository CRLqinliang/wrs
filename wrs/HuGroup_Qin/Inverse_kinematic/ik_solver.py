""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240620 Osaka Univ.

"""

"""
This is an example to
1. solve inverse kinematics (IK)
2. check the self-collision of the simulation robot
(Run the 4_define_grasp.py First)
"""

import basis.robot_math as rm
import modeling.collision_model as cm
import numpy as np
import visualization.panda.world as wd
import wrs.HuGroup_Qin.utils.file_sys as fs
import wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_dual as nova2_wrsv3gripper_dual

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
hugroup_lab_nova2_dual = nova2_wrsv3gripper_dual.hugroup_lab_nova2_dual(enable_cc=True)

# generate the tube to be grasped
obj_mdl = cm.CollisionModel(initor="H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl")
obj_mdl.pos = np.array([0.2, -.45, 0.2])
obj_mdl.rotmat = rm.rotmat_from_euler(0, np.pi/2, 0)
obj_mdl.attach_to(base)

# load the grasp poses for the tube
grasps_collection = fs.load_pickle("/wrs/HuGroup_Qin\data\grasps\pipe_50ml_grasps.pkl")


#nova2_wrsv3gripper.lft_arm.manipulator.jlc._ik_solver.test_success_rate()


for ind, grasp in enumerate(grasps_collection):
    print(f"--------------------- grasp pose index: {ind} ---------------------------")

    obj_pose = obj_mdl.homomat
    grasp_pose = rm.homomat_from_posrot(grasp.ac_pos, grasp.ac_rotmat)

    print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")
    rbt_ee_pose = np.dot(obj_pose, grasp_pose)


    ik_sol = hugroup_lab_nova2_dual.lft_arm.ik(tgt_pos=rbt_ee_pose[:3, 3],
                                           tgt_rotmat=rbt_ee_pose[:3, :3])

    if ik_sol is not None:  # check IK-feasible
        hugroup_lab_nova2_dual.lft_arm.fk(ik_sol)
        if hugroup_lab_nova2_dual.lft_arm.is_collided():  # check if is self-collided
            print("The robot is self-collided")
            continue
        else:
            # illustrate the ik solution for grasping the tube
            hugroup_lab_nova2_dual.lft_arm.end_effector.change_jaw_width(grasp.ee_values)
            hugroup_lab_nova2_dual.lft_arm.goto_given_conf(ik_sol)
            hugroup_lab_nova2_dual.gen_meshmodel(alpha=.8).attach_to(base)
            hugroup_lab_nova2_dual.gen_stickmodel().attach_to(base)
            hugroup_lab_nova2_dual.show_cdprim()
            base.run()
    else:
        print("The robot is not able to reach the target pose")
exit(-1)

