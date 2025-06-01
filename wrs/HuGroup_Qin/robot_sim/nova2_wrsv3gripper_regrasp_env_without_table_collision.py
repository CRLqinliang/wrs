""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241202 Osaka Univ.

"""
import math
import os
import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(root_dir)
import wrs.basis.constant as constant
import wrs.basis.robot_math as rm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.manipulators.dobot_nova2.nova2 as nova2
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
import wrs.robot_sim.robots.single_arm_robot_interface as ri
from panda3d.core import NodePath, CollisionNode, CollisionBox, Point3
from wrs.modeling import collision_model as mcm
from wrs import modeling as m


class nova2_gripper_v3_without_table(ri.SglArmRobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='d', name="nova2_gripper_v3", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # the body anchor
        self.body = rkjlc.rkjl.Anchor(name="regrasp_nova2_base", pos=self.pos, rotmat=self.rotmat, n_flange=1,
                                      n_lnk=1)
        # need to be updated
        self.body.loc_flange_pose_list[0] = [np.array([0, 0, 0]),
                                             rm.rotmat_from_euler(0, 0, 0)]
        self.body.lnk_list[0].name = "regrasp_nova2_base_link"
        self.body.lnk_list[0].cmodel = mcm.CollisionModel(
                        initor=os.path.join(current_file_dir, "meshes", "regrasp_table_env.stl"),
                        cdprim_type=m.constant.CDPrimType.USER_DEFINED,
                        userdef_cdprim_fn=self._base_cdprim)
        self.body.lnk_list[0].cmodel.rgba = constant.hug_gray

        self.manipulator = nova2.Nova2(pos=self.body.gl_flange_pose_list[0][0],
                                       rotmat=self.body.gl_flange_pose_list[0][1],
                                       ik_solver=ik_solver, name=name + "_manipulator",
                                       enable_cc=False)

        self.end_effector = wrs_gripper_v3.WRSGripper3(pos=self.manipulator.gl_flange_pos,
                                            rotmat=self.manipulator.gl_flange_rotmat,
                                            name=name + "_eef")

        # tool center point - mount the end effector to the manipulator's end.
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat

        if self.cc is not None:
            self.setup_cc()


    @staticmethod
    def _base_cdprim(name=None, ex_radius=None):
        pdcnd = CollisionNode('regrasp_nova2_base')
        pdcnd.addSolid(CollisionBox(Point3(0, 0.5, -.22), 1, 1, .22))
        pdcnd.addSolid(CollisionBox(Point3(-.29, 0, 0), 0.01, 1, 2))
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def setup_cc(self):
        # end effector
        ee_cces = []
        for id, cdlnk in enumerate(self.end_effector.cdelements):
            ee_cces.append(self.cc.add_cce(cdlnk))

        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list =  ee_cces + [ml4, ml5]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3, ml4], type="into")
        self.cc.dynamic_ext_list = ee_cces


    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='nova2_regrasp_exp'):

        m_col = mmc.ModelCollection(name=self.name+'_meshmodel')

        if self._manipulator is not None:
            self._manipulator.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_flange_frame=toggle_flange_frame,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self.end_effector is not None:
            self.end_effector.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col

def obj_setup(name, pos, rotmat, rgb=None, alpha=None):
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha,
                                    initor=r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_cmodel.pos = pos
    obj_cmodel.rotmat = rotmat
    return obj_cmodel

if __name__ == '__main__':
    import time
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    # 两个marker相对于机器人末端的变换矩阵
    T_ee_marker = {
        0: np.array([  # marker-0的位姿：z轴与tcp x同向，y轴与tcp-z同向
            [1, 0, 0, -0.043],  # z轴指向x
            [0, 0, 1, 0],       # x轴指向y
            [0, -1, 0, 0],  # y轴指向-z
            [0, 0, 0, 1]
        ]),
        4: np.array([  # marker-0的位姿：z轴与tcp x同向，y轴与tcp-z同向
            [1, 0, 0, 0.043],            # z轴指向x
            [0, 0, 1, 0],        # x轴指向y
            [0, -1, 0, 0],           # y轴指向-z
            [0, 0, 0, 1]
        ]),
    }

    T_base_marker = {
        0: np.array([
            [0, -1, 0, 0.16],
            [1, 0, 0, 0.14],
            [0, 0, 1, 0.006],
            [0, 0, 0, 1]
        ])
    }


    # world configuration
    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = nova2_gripper_v3_without_table(enable_cc=True)

    # robot configuration [0.11929658 0.19255743 0.02712538]
    # [[ 0.67696922  0.28883446 -0.67696922]
    #  [ 0.2042368  -0.95737906 -0.2042368 ]
    #  [-0.70710678  0.         -0.70710678]]
       # current_jnv = np.array([-113.4576, 13.06, 127.8234, -98.2558, -100.2281, -168.2637]) * np.pi / 180
       #  target_pos, target_rotmat = robot.fk(current_jnv)
       #  print(target_pos)
       #  robot.goto_given_conf(jnt_values=current_jnv)
       #  robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
       #  robot.show_cdprim()
       #  base.run()

    # # 显示两个marker的坐标系
    # tcp_pos, tcp_rotmat = robot.fk(current_jnv)
    # tcp_homomat = rm.homomat_from_posrot(tcp_pos, tcp_rotmat)
    #
    # # 显示marker 0的坐标系
    # marker0_homomat = tcp_homomat @ T_ee_marker[0]
    # marker0_pos = marker0_homomat[:3, 3]
    # marker0_rotmat = marker0_homomat[:3, :3]
    # mgm.gen_frame(pos=marker0_pos, rotmat=marker0_rotmat, ax_length=0.05 ,alpha=0.7).attach_to(base)
    #
    # # 显示marker 4的坐标系
    # marker1_homomat = tcp_homomat @ T_ee_marker[4]
    # marker1_pos = marker1_homomat[:3, 3]
    # marker1_rotmat = marker1_homomat[:3, :3]
    # mgm.gen_frame(pos=marker1_pos, rotmat=marker1_rotmat, ax_length=0.05, alpha=0.7).attach_to(base)
    #
    # # 显示基座marker的坐标系
    # marker2_homomat = T_base_marker[0]
    # marker2_pos = marker2_homomat[:3, 3]
    # marker2_rotmat = marker2_homomat[:3, :3]
    # mgm.gen_frame(pos=marker2_pos, rotmat=marker2_rotmat, ax_length=0.05, alpha=0.7).attach_to(base)

    target_pos = [0.11929658 ,0.19255743 ,0.02712538]
    target_rotmat = np.array([[0.67696922, 0.28883446, -0.67696922],
                              [0.2042368, -0.95737906, -0.2042368],
                              [-0.70710678, 0., -0.70710678]])
    ik_jnvs = robot.ik(target_pos, target_rotmat, option='multiple')
    print(ik_jnvs * 180/np.pi)
    for ik_jnv in ik_jnvs:
        robot.goto_given_conf(ik_jnv)
        if not robot.is_collided():
            robot.gen_meshmodel(alpha=0.8, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
            robot.show_cdprim()
    # robot.unshow_cdprim()

    base.run()
