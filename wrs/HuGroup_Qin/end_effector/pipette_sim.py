""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240618 Osaka Univ.

"""
import os
import math
import numpy as np
import wrs.modeling.model_collection as mmc
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class pipette(gpi.GripperInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMType.DEFAULT, name="nova2_pipette"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1],
                                 n_dof=1, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes",
                                                                             "pipette_gripper_v1.stl"),
                                                                              cdmesh_type=self.cdmesh_type,
                                                                              cdprim_type=mcm.mc.CDPType.USER_DEFINED,
                                                                              userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.5, .5, 1, 1])

        # reinitialize
        self.jlc.finalize()

        # acting center
        self.loc_acting_center_pos = np.array([0, -0.075, 0.061])

        # collision detection
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0])

        # dummy jaw width
        self.jaw_range = np.array([0.0, 0.05])

    @staticmethod
    def _base_cdprim(ex_radius=None):
        pdcnd = CollisionNode("pipette_cdnode")
        collision_primitive_r0 = CollisionBox(Point3(0, -0.01, -0.012),
                                              x=.129/2 + ex_radius, y=0.060 + ex_radius, z=.055/2 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        collision_primitive_r1 = CollisionBox(Point3(0, -0.075, 0.01),
                                                x=.03/2 + ex_radius, y=0.025 + ex_radius, z=.0515 + ex_radius)
        pdcnd.addSolid(collision_primitive_r1)

        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self.pos
        self.coupling.rotmat = self.rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    # dummy function, for the consistence
    def get_jaw_width(self):
        return self.jaw_range[0]

    # dummy function, for the consistence
    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        self.jaw_width = self.jaw_range[0]

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False, name='nova2_pipette'):
        m_col = mmc.ModelCollection(name=name)

        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='nova2_pipette_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    mgm.gen_frame().attach_to(base)
    end_effort = pipette()
    end_effort.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    end_effort.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    #end_effort.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    end_effort.show_cdprim()
    base.run()
