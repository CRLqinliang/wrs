""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240618 Osaka Univ.

"""
import math
import numpy as np
import wrs.robot_sim.manipulators.dobot_nova2.nova2 as nova2
import wrs.HuGroup_Qin.end_effector.pipette_sim as pipette
import wrs.robot_sim.robots.single_arm_robot_interface as ri


class nova2_pipette(ri.SglArmRobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name="nova2_pipette", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        self.manipulator = nova2.Nova2(pos=self.pos, rotmat=self.rotmat,
                                       ik_solver=ik_solver, name=name + "_manipulator", enable_cc=False,
                                       pipette=True)

        self.end_effector = pipette.pipette(pos=self.manipulator.gl_flange_pos,
                                            rotmat=self.manipulator.gl_flange_rotmat,
                                            name=name + "_eef")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # TODO when pose is changed, oih info goes wrong
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])

        # manipulator
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)

        from_list = [elb, ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        oiee_into_list = []
        # TODO oiee?

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def _ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray, seed_jnt_values=np.zeros(6)):
        # considering the base coordinate system.
        robot_to_world = rm.homomat_from_posrot(self.pos, self.rotmat)
        target_in_world = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)
        target_in_robot = np.dot(np.linalg.inv(robot_to_world), target_in_world)

        return self.manipulator._ik(target_in_robot[:3, 3], target_in_robot[:3, :3], seed_jnt_values)

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)

if __name__ == '__main__':
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = nova2_pipette(enable_cc=True)

    # robot.cc.show_cdprim()
    # robot.goto_given_conf(jnt_values=np.radians(np.array([20, -90, 120, 30, 0, 40, 0])))
    robot.goto_given_conf(jnt_values=np.radians([0, -54.6387, 68.4399, -102.8725, 90.71, -180]))
    robot.goto_given_conf(jnt_values=np.radians([-35.5627, -39.4677, 33.2763, -82.1391, 90.6088, -188.5230]))
    robot.gen_meshmodel(alpha=.8, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    base.run()

    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print("IK result: ", jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.show_cdprim()
    robot.unshow_cdprim()
    base.run()



