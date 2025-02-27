""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240618 Osaka Univ.

"""
import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.model_collection as mmc
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.HuGroup_Qin.robot_sim.nova2_pipette as nova2_pipette
import wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper as nova2_wrsv3gripper
from panda3d.core import NodePath, CollisionNode, CollisionBox, Point3
import wrs.robot_sim.robots.robot_interface as ri


class hugroup_lab_nova2_dual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='hugroup_lab_nova2_dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # the body anchor
        self.body = rkjlc.rkjl.Anchor(name="hugroup_lab_nova2_dual_base", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=1)
        # need to be updated
        self.body.loc_flange_pose_list[0] = [np.array([.15, -0.40, 0]),
                                             rm.rotmat_from_euler(0, 0, 0)]
        self.body.loc_flange_pose_list[1] = [np.array([.15, 0.40, 0]),
                                             rm.rotmat_from_euler(0, 0, 0)]
        # need to be updated
        self.body.lnk_list[0].name = "hugroup_lab_nova2_dual_base_link"
        self.body.lnk_list[0].cmodel = mcm.CollisionModel(
                                        initor=os.path.join(current_file_dir, "meshes", "hugroup_lab_nova2_dual_base.stl"),
                                        cdprim_type=mcm.mc.CDPType.USER_DEFINED,
                                        userdef_cdprim_fn=self._base_cdprim)
        self.body.lnk_list[0].cmodel.rgba = rm.bc.hug_gray

        # right arm
        self.rgt_arm = nova2_wrsv3gripper.nova2_gripper_v3(pos=self.body.gl_flange_pose_list[0][0],
                                                           rotmat=self.body.gl_flange_pose_list[0][1],
                                                           ik_solver=None)

        self.rgt_arm.home_conf = np.array([0, 0, 0, 0, 0, 0])
        self.rgt_arm.manipulator.jnts[0].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.rgt_arm.manipulator.jnts[1].motion_range = np.array([-np.pi, np.pi])
        self.rgt_arm.manipulator.jnts[2].motion_range = np.array([-np.pi * 5/6, np.pi * 5/6])
        self.rgt_arm.manipulator.jnts[3].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.rgt_arm.manipulator.jnts[4].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.rgt_arm.manipulator.jnts[5].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.rgt_arm.manipulator.jlc.finalize(ik_solver='d', identifier_str=self.rgt_arm.name + "_dual_rgt")

        # left arm
        self.lft_arm = nova2_pipette.nova2_pipette(pos=self.body.gl_flange_pose_list[1][0],
                                                   rotmat=self.body.gl_flange_pose_list[1][1],
                                                   ik_solver=None)
        self.lft_arm.home_conf = np.array([0, 0, 0, 0, 0, 0])
        self.lft_arm.manipulator.jnts[0].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.lft_arm.manipulator.jnts[1].motion_range = np.array([-np.pi, np.pi])
        self.lft_arm.manipulator.jnts[2].motion_range = np.array([-np.pi * 5/6, np.pi * 5/6])
        self.lft_arm.manipulator.jnts[3].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.lft_arm.manipulator.jnts[4].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.lft_arm.manipulator.jnts[5].motion_range = np.array([-2*np.pi, 2*np.pi])
        self.lft_arm.manipulator.jlc.finalize(ik_solver='d', identifier_str=self.lft_arm.name + "_dual_lft")
        if self.cc is not None:
            self.setup_cc()
        # go home
       # self.goto_home_conf()

    @staticmethod
    def _base_cdprim(ex_radius=None):
        pdcnd = CollisionNode("nova2_dual_base")
        collision_primitive_r0 = CollisionBox(Point3(.66/2, 0.54+0.01/2, 0.7/2 - 0.03),
                                              x=.66/2 + ex_radius, y=0.01/2 + ex_radius, z=.7/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)

        collision_primitive_r1 = CollisionBox(Point3(.66/2, -(0.54+0.01/2), 0.7/2 - 0.03),
                                              x=.66/2 + ex_radius, y=0.01/2 + ex_radius, z=.7/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_r1)

        collision_primitive_b0 = CollisionBox(Point3(.66/2, 0, -0.03/2),
                                              x=.66/2 + ex_radius, y=1.1/2 + ex_radius, z=.03/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_b0)

        # collision_primitive_u0 = CollisionBox(Point3(.66/2, 0, 0.685 - 0.03),
        #                                         x=.66/2 + ex_radius, y=1.1/2 + ex_radius, z=.03/2 + ex_radius)
        # pdcnd.addSolid(collision_primitive_u0)

        collision_primitive_bc = CollisionBox(Point3(-0.02/2, 0, 0.7/2 - 0.03),
                                                x=0.02/2 + ex_radius, y=1.1/2 + ex_radius, z=.7/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_bc)

        collision_primitive_cam = CollisionBox(Point3(0.07, 0, 0.51),
                                               x=.14/2 + ex_radius, y=.14/2 + ex_radius, z=.09/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_cam)

        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @property
    def n_dof(self):
        if self.delegator is None:
            return self.lft_arm.n_dof + self.rgt_arm.n_dof
        else:
            return self.delegator.n_dof

    def setup_cc(self):
        """
        author: weiwei
        date: 20240309
        """
        # dual arm
        # body
        bd = self.cc.add_cce(self.body.lnk_list[0])
        # left ee
        lft_elb = self.cc.add_cce(self.lft_arm.end_effector.jlc.anchor.lnk_list[0])
        # lft_el0 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[0].lnk)
        # lft_el1 = self.cc.add_cce(self.lft_arm.end_effector.jlc.jnts[1].lnk)
        # left manipulator
        lft_mlb = self.cc.add_cce(self.lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_elb = self.cc.add_cce(self.rgt_arm.end_effector.jlc.anchor.lnk_list[0])
        rgt_el0 = self.cc.add_cce(self.rgt_arm.end_effector.jlc.jnts[0].lnk)
        rgt_el1 = self.cc.add_cce(self.rgt_arm.end_effector.jlc.jnts[1].lnk)
        # right manipulator
        rgt_mlb = self.cc.add_cce(self.rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs - self collision and env collision (2nd joint will not be in self-collision)
        from_list = [lft_ml3, lft_ml4, lft_ml5, lft_elb,
                     rgt_ml3, rgt_ml4, rgt_ml5, rgt_elb, rgt_el0, rgt_el1]
        into_list = [bd, lft_mlb, lft_ml0, lft_ml1, rgt_mlb, rgt_ml0, rgt_ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs - two robot end efforts collision
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ml5, lft_elb]
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5, rgt_elb, rgt_el0, rgt_el1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # point low-level cc to the high-level one
        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc

    def use_both(self):
        self.delegator = None

    def use_lft(self):
        self.delegator = self.lft_arm

    def use_rgt(self):
        self.delegator = self.rgt_arm

    def backup_state(self):
        if self.delegator is None:
            self.rgt_arm.backup_state()
            self.lft_arm.backup_state()
        else:
            self.delegator.backup_state()

    def restore_state(self):
        if self.delegator is None:
            self.rgt_arm.restore_state()
            self.lft_arm.restore_state()
        else:
            self.delegator.restore_state()

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.body.pos = self.pos
        self.body.rotmat = self.rotmat
        self.lft_arm.fix_to(pos=self.body.gl_flange_pose_list[0][0],
                            rotmat=self.body.gl_flange_pose_list[0][1])
        self.rgt_arm.fix_to(pos=self.body.gl_flange_pose_list[1][0],
                            rotmat=self.body.gl_flange_pose_list[1][1])

    def fk(self, jnt_values, toggle_jacobian=False):
        if self.delegator is None:
            raise AttributeError("FK is not available in multi-arm mode.")
        else:
            return self.delegator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, toggle_dbg=False):
        if self.delegator is None:
            raise AttributeError("IK is not available in multi-arm mode.")
        else:
            # we use the analytical IK solver "_IK" in the nova2_pipette class
            return self.delegator._ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat,
                                      seed_jnt_values=seed_jnt_values)

    def goto_given_conf(self, jnt_values, ee_values=None):
        """
        :param jnt_values: nparray 1x14, 0:7lft, 7:14rgt
        :return:
        author: weiwei
        date: 20240307
        """
        if self.delegator is None:
            if len(jnt_values) != self.lft_arm.manipulator.n_dof + self.rgt_arm.manipulator.n_dof:
                raise ValueError("The given joint values do not match total n_dof")
            self.lft_arm.goto_given_conf(jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof])
            self.rgt_arm.goto_given_conf(jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])  # TODO
        else:
            self.delegator.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def goto_home_conf(self):
        if self.delegator is None:
            self.lft_arm.goto_home_conf()
            self.rgt_arm.goto_home_conf()
        else:
            self.delegator.goto_home_conf()

    def get_jnt_values(self):
        if self.delegator is None:
            return np.concatenate((self.lft_arm.get_jnt_values(), self.rgt_arm.get_jnt_values()))
        else:
            return self.delegator.get_jnt_values()

    def rand_conf(self):
        """
        :return:
        author: weiwei
        date: 20210406
        """
        if self.delegator is None:
            return np.concatenate((self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()))
        else:
            return self.delegator.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        if self.delegator is None:
            return self.lft_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof]) and self.rgt_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])
        else:
            return self.delegator.are_jnts_in_ranges(jnt_values=jnt_values)

    def is_hold(self):
        if self.delegator is None:
            return self.lft_arm.is_hold() and self.rgt_arm.is_hold()
        else:
            return self.delegator.is_hold()

    def get_jaw_width(self):
        return self.get_ee_values()

    def change_jaw_width(self, jaw_width):
        self.change_ee_values(ee_values=jaw_width)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20240307
        """
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='nova2_dual_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.body.gen_stickmodel(toggle_root_frame=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.lft_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame,
                                    name=name + "_lft_arm").attach_to(m_col)
        self.rgt_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame,
                                    name=name + "_rgt_arm").attach_to(m_col)
        return m_col
    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='nova2_dual_meshmodel'):

        m_col = mmc.ModelCollection(name=name)

        self.body.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=toggle_flange_frame,
                                toggle_root_frame=toggle_jnt_frames, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, name=name + "_body").attach_to(m_col)

        self.lft_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh,
                                   name=name + "_lft_arm").attach_to(m_col)
        self.rgt_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh,
                                   name=name + "_rgt_arm").attach_to(m_col)
        return m_col


if __name__ == '__main__':

    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import modeling.collision_model as cm
    import wrs.HuGroup_Qin.utils.file_sys as fs

    base = wd.World(cam_pos=[3, 0, 0.35], lookat_pos=[0, 0, 0.35])
    mgm.gen_frame().attach_to(base)

    obj_mdl = cm.CollisionModel(initor="H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl")
    obj_mdl.pos = np.array([0.05, -0.25, 0.25])
    obj_mdl.rotmat = rm.rotmat_from_euler(-np.pi/2, 0, 0)
    obj_mdl.show_local_frame()
    obj_mdl.attach_to(base)

    grasps_collection = fs.load_pickle("/wrs/HuGroup_Qin\data\grasps\pipe_50ml_grasps.pkl")

    robot = hugroup_lab_nova2_dual(enable_cc=True)
    robot.use_rgt()

    robot.lft_arm.goto_given_conf(jnt_values=np.radians(np.array([-12.93, -24.28, -93.69, 3.67, 21.95, -0.22])))
    for ind, grasp in enumerate(grasps_collection):
        obj_pose = obj_mdl.homomat
        grasp_pose = rm.homomat_from_posrot(grasp.ac_pos, grasp.ac_rotmat)
        rbt_ee_pose_init = np.dot(obj_pose, grasp_pose)
        ik_sol = robot.ik(tgt_pos=rbt_ee_pose_init[:3, 3],
                          tgt_rotmat=rbt_ee_pose_init[:3, :3], seed_jnt_values=None)
        #robot.change_jaw_width(.0085)
        if ik_sol is not None:
            for ik_sol_item in ik_sol:
                robot.goto_given_conf(jnt_values=ik_sol_item)
                if robot.is_collided():
                    print(">>> The robot is self-collided")
                    continue
                robot.gen_meshmodel(alpha=0.9, toggle_jnt_frames=True).attach_to(base)
                robot.gen_stickmodel(toggle_flange_frame=True).attach_to(base)
                robot.show_cdprim()
        else:
            print(">>> No IK solution")
            continue
    base.run()
    pass
