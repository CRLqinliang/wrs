""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240618 Osaka Univ.

"""
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.manipulators.dobot_nova2.nova2 as nova2
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
import wrs.robot_sim.robots.single_arm_robot_interface as ri


class nova2_gripper_v3(ri.SglArmRobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name="nova2_gripper_v3", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        self.manipulator = nova2.Nova2(pos=self.pos, rotmat=self.rotmat,
                                     ik_solver=ik_solver, name=name + "_manipulator", enable_cc=False)

        self.end_effector = wrs_gripper_v3.WRSGripper3(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat, name=name + "_eef")

        # tool center point - mount the end effector to the manipulator's end.
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, el0, el1, ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3, ml4]

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)

    def _ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray, seed_jnt_values=np.zeros(6), option='multiple'):
        # considering the base coordinate system.
        # If there is no base coordination, robot_to_world = np.eye(4)
        robot_to_world = rm.homomat_from_posrot(self.pos, self.rotmat)
        target_in_world = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        target_in_robot = np.dot(np.linalg.inv(robot_to_world), target_in_world)
        return self.manipulator.ik(target_in_robot[:3, 3], target_in_robot[:3, :3], seed_jnt_values, option=option)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

def show_path(robot, path, base):
    for jnts_s in path:
        robot.goto_given_conf(jnt_values=jnts_s)
        robot.gen_meshmodel(alpha=.5, toggle_jnt_frames=True).attach_to(base)
        robot.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    base.run()


if __name__ == '__main__':
    import time
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.motion.optimization_based.incremental_nik as inik

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = nova2_gripper_v3(enable_cc=True)

    current_jnv = np.array([-105.7156, -5.8495, 107.7660, -14.8353, -96.9777, -3.9212]) * np.pi / 180
    robot.goto_given_conf(jnt_values=current_jnv)
    # robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot.show_cdprim()
    # robot.unshow_cdprim()
    base.run()
    
    cur_pos, cur_rotmat = robot.fk(current_jnv)
    tgt_pos = cur_pos
    rotz = 540 * math.pi / 180 # spin angle target
    tgt_rotmat = np.dot(cur_rotmat , rm.rotmat_from_euler(0, 0, rotz)) # rotmation matrix of the target in the world frame.

    # jnv_list = inik_planner.gen_linear_motion(start_tcp_pos=cur_pos, start_tcp_rotmat=cur_rotmat,
    #                                           goal_tcp_pos=tgt_pos, goal_tcp_rotmat=tgt_rotmat)

    mgm.gen_frame(ax_length=.5, pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot._ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=None, option='multiple')
    print("IK result: ", jnt_values * 180 / math.pi)
    if jnt_values is not None:
            robot.goto_given_conf(jnt_values=jnt_values)
            robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
            robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.show_cdprim()
    robot.unshow_cdprim()
    base.run()

