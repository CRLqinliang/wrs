""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240706 Osaka Univ.

"""

import numpy as np
import json
import time
from wrs.HuGroup_Qin.robot_con.nova2_pipette import Nova2Pipette
from wrs.HuGroup_Qin.robot_con.nova2_gripper import Nova2Gripper
import box


def data_process(trj_jnv):
    return [np.array(sublist) for sublist in trj_jnv]


def lft_arm_excute_func(lft_arm, control_frequency, path_data):
    trj_jnv = data_process(path_data)
    values_watch = lft_arm.get_jnt_values()
    # trj_jnv.insert(0, lft_arm.get_jnt_values())
    lft_arm.move_jntspace_path(path=trj_jnv, max_jntvel=[1.5]*trj_jnv[0].shape[0],
                               control_frequency=control_frequency)


def rgt_arm_excute_func(rgt_arm, control_frequency, path_data):
    trj_jnv = data_process(path_data)
    values_watch = rgt_arm.get_jnt_values()
    # trj_jnv.insert(0, rgt_arm.get_jnt_values())
    rgt_arm.move_jntspace_path(path=trj_jnv, max_jntvel=[1.5]*trj_jnv[0].shape[0],
                               control_frequency=control_frequency)


def lft_arm_excute_movej_func(lft_arm, path_data):
    trj_jnv = data_process(path_data)
    for index in range(len(trj_jnv)):
        lft_arm.move_j(jnt_val=trj_jnv[index])


def rgt_arm_excute_movej_func(rgt_arm, path_data):
    trj_jnv = data_process(path_data)
    for index in range(len(trj_jnv)):
        rgt_arm.move_j(jnt_val=trj_jnv[index])


def lft_arm_excute_movel_func(lft_arm, dive_depth):
    pos, rot = lft_arm.get_pose()
    pos = pos + np.array([0, 0, -dive_depth])
    lft_arm.move_l(pos, rot)


def rgt_arm_excute_movel_func(rgt_arm, dive_depth):
    pos, rot = rgt_arm.get_pose()
    pos = pos + np.array([0, 0, -dive_depth])
    rgt_arm.move_l(pos, rot)


if __name__ == '__main__':
    # rgt_arm : 192.168.5.10
    # lft_arm : 192.168.5.100

    path_data = json.load(open("path_jnv.json", "r"))
    path_data = box.Box(path_data)
    control_frequency = 1 / 200

    lft_arm = Nova2Pipette(ip="192.168.5.100", init_enable_rbt=True)
    lft_arm.set_speed(20)
    rgt_arm = Nova2Gripper(ip="192.168.5.10", has_gripper=True, init_enable_rbt=True)

    time.sleep(2)
    print(">>> Start to move to initial position")
    # new lft_arm 5th joint has -180 deg offset
    tgt_jnv = np.radians(np.array([0, -54.6387, 68.4399, -102.8725, 90.71, -180]))
    lft_arm.move_jntspace_path([lft_arm.get_jnt_values(), tgt_jnv], control_frequency=control_frequency)

    tgt_jnv = np.radians(np.array([-180, 25, 85, 0, -90, 0]))
    rgt_arm.move_jntspace_path([rgt_arm.get_jnt_values(), tgt_jnv], control_frequency=control_frequency)

    time.sleep(2)
    # task excute - pipe_load_operation_path
    print(">>> Start to excute the pipe_load_operation_path task")
    rgt_arm.gripper_x.open_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_operation_path.rgt_to_pipe_path)
    rgt_arm.gripper_x.close_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_operation_path.rgt_pipe_pullback_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_operation_path.rgt_pipe_to_reload_path)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe_load_operation_path.lft_to_reload_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_operation_path.rgt_pipe_up_path)
    rgt_arm.gripper_x.open_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_operation_path.rgt_reload_to_waiting_path)

    # task excute - liquid_dropping_path - moveJ
    print(">>> Start to excute the liquid_dropping_path task")
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_absorb_path_move)
    lft_arm.pipette.hold()
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_absorb_path_down)
    lft_arm.pipette.abosrb(1.5)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_absorb_path_up)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_release_path_move)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_release_path_down)
    lft_arm.pipette.release(5)
    lft_arm.pipette.hold()
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_dropping_path.lft_release_path_up)


    # task excute - pipe1_deposit_operation_path
    print(">>> Start to excute the pipe1_deposit_operation_path task")
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe1_deposit_operation_path.lft_to_reload_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe1_deposit_operation_path.rgt_wait_to_pipe_path)
    rgt_arm.gripper_x.close_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe1_deposit_operation_path.rgt_pipe_to_pullout_path)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe1_deposit_operation_path.lft_reload_to_standby_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe1_deposit_operation_path.rgt_pullout_to_pre_release_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe1_deposit_operation_path.rgt_pre_release_to_release_path)
    rgt_arm.gripper_x.open_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe1_deposit_operation_path.rgt_release_to_waiting_path)


    # task excute - pipe_load_replace_path
    print(">>> Start to excute the pipe_load_replace_path task")
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_replace_path.rgt_to_pipe_path)
    rgt_arm.gripper_x.close_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_replace_path.rgt_pipe_pullback_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_replace_path.rgt_pipe_to_reload_path)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe_load_replace_path.lft_to_reload_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_replace_path.rgt_pipe_up_path)
    rgt_arm.gripper_x.open_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe_load_replace_path.rgt_reload_to_waiting_path)


    # task liquid_adding_path
    print(">>> Start to excute the liquid_adding_path task")
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_absorb_path_move)
    lft_arm.pipette.hold()
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_absorb_path_down)
    lft_arm.pipette.abosrb(1.5)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_absorb_path_up)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_release_path_move)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_release_path_down)
    lft_arm.pipette.release(5)
    lft_arm.pipette.hold()
    lft_arm_excute_func(lft_arm, control_frequency, path_data.liquid_adding_path.lft_release_path_up)


    # task excute - pipe2_deposit_operation_path
    print(">>> Start to excute the pipe2_deposit_operation_path task")
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe2_deposit_operation_path.lft_to_reload_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe2_deposit_operation_path.rgt_wait_to_pipe_path)
    rgt_arm.gripper_x.close_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe2_deposit_operation_path.rgt_pipe_to_pullout_path)
    lft_arm_excute_func(lft_arm, control_frequency, path_data.pipe2_deposit_operation_path.lft_reload_to_standby_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe2_deposit_operation_path.rgt_pullout_to_pre_release_path)
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe2_deposit_operation_path.rgt_pre_release_to_release_path)
    rgt_arm.gripper_x.open_gripper()
    rgt_arm_excute_func(rgt_arm, control_frequency, path_data.pipe2_deposit_operation_path.rgt_release_to_waiting_path)

