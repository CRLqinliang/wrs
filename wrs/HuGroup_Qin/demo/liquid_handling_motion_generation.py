""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240629 Osaka Univ.

"""

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as cm
import numpy as np
import json
import wrs.visualization.panda.world as wd
import wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_pipette_dual as nova2_wrsv3gripper_dual
import wrs.motion.optimization_based.incremental_nik as inik
import wrs.motion.probabilistic.rrt_connect as rrtc
from wrs import wd, rm, ym, rrtc, mgm

try:
    import motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

class ObjetType:
    # The manipulated objects should be defined in advance.
    def __init__(self, obj_mdl_path, obj_mdl_pos, obj_mdl_rotmat, grasp_path, name):
        self.obj_mdl_path = obj_mdl_path
        self.obj_mdl = cm.CollisionModel(initor=obj_mdl_path)
        self.obj_mdl.pos = obj_mdl_pos
        self.obj_mdl.rotmat = rm.rotmat_from_euler(obj_mdl_rotmat[0], obj_mdl_rotmat[1], obj_mdl_rotmat[2])
        self.grasp_collection = fs.load_pickle(grasp_path)
        self.name = name

    def get_configurations(self):
        return [self.obj_mdl_path, self.obj_mdl, self.grasp_collection, self.name]

    def set_pos_rotmat(self, pos=None, rotmat=None):
        if pos is not None:
            self.obj_mdl.pos = pos
        if rotmat is not None:
            self.obj_mdl.rotmat = rotmat


class ObjectCollection:
    def __init__(self):
        self.obj_collection = dict()

    def add_object(self, obj: ObjetType):
        self.obj_collection[obj.name] = obj

    def remove_object(self, name: str):
        if name in self.obj_collection.keys():
            self.obj_collection.pop(name)
        else:
            raise ValueError("The object is not in the collection")

    def update_objects_to_base(self, base: wd.World):
        for obj_name in self.obj_collection.keys():
            self.obj_collection[obj_name].obj_mdl.attach_to(base)

    def set_object_config(self, name: str, pos, rotmat):
        if name not in self.obj_collection.keys():
            raise ValueError("The object is not in the collection")
        else:
            self.obj_collection[name].set_pos_rotmat(pos=pos, rotmat=rotmat)

    def get_object_config(self, name: str):
        if name not in self.obj_collection.keys():
            raise ValueError("The object is not in the collection")
        else:
            return self.obj_collection[name].get_configurations()


class LiquidHandlingTask:
    def __init__(self, args, base, dual_robot_sys, obj_collection: ObjectCollection):
        # init the base
        self.args = args
        self.base = base

        # init the dual robot system
        self.dual_robot_sys = dual_robot_sys
        self.init_jnt_values_lft = args.init_jnt_values[0]
        self.init_jnt_values_rgt = args.init_jnt_values[1]
        self.obj_collection = obj_collection

        # init the joint values of the dual robot system
        self.dual_robot_sys.lft_arm.goto_given_conf(jnt_values=np.radians(self.init_jnt_values_lft))
        self.dual_robot_sys.rgt_arm.goto_given_conf(jnt_values=np.radians(self.init_jnt_values_rgt))

        # RRT & linear motion planner init of both arms
        self.rrtc_planner = rrtc.RRTConnect(self.dual_robot_sys)
        self.inink_svlr = inik.IncrementalNIK(self.dual_robot_sys)

        # for animation
        self.robot_mesh_list = []
        self.obj_mesh_list = []

    def update_system(self):
        self.dual_robot_sys.gen_meshmodel(alpha=.8, toggle_jnt_frames=True).attach_to(self.base)
        self.dual_robot_sys.gen_stickmodel(toggle_jnt_frames=True).attach_to(self.base)
        # self.dual_robot_sys.show_cdprim()
        for obj_name in self.obj_collection.obj_collection.keys():
            self.obj_collection.obj_collection[obj_name].obj_mdl.attach_to(self.base)
        self.base.run()

    def show_path(self, path):
        for jnts_s in path:
            self.dual_robot_sys.goto_given_conf(jnt_values=jnts_s)
            self.dual_robot_sys.gen_meshmodel(alpha=.5, toggle_jnt_frames=True).attach_to(self.base)
            self.dual_robot_sys.gen_stickmodel(toggle_jnt_frames=True).attach_to(self.base)
            # self.dual_robot_sys.show_cdprim()
            for obj_name in self.obj_collection.obj_collection.keys():
                self.obj_collection.obj_collection[obj_name].obj_mdl.attach_to(self.base)
        self.base.run()

    def extend_mesh(self, path_jnv):
        for jnts_s in path_jnv:
            self.dual_robot_sys.goto_given_conf(jnt_values=jnts_s)
            self.robot_mesh_list.append(self.dual_robot_sys.gen_meshmodel(alpha=.8, toggle_jnt_frames=True))
            for obj_name in self.obj_collection.obj_collection.keys():
                if obj_name != "pipe_50ml_dummy": # no dummy pipe
                    self.obj_mesh_list.append(self.obj_collection.obj_collection[obj_name].obj_mdl)

    def path_check(self, path):
        return True if path is None else False

    def collision_check(self, jnt_value):
        # check if the robot is self-collided (use_left or use_right)
        current_jnt_values = self.dual_robot_sys.get_jnt_values()
        self.dual_robot_sys.goto_given_conf(jnt_values=jnt_value)
        is_collided = self.dual_robot_sys.is_collided()
        self.dual_robot_sys.goto_given_conf(jnt_values=current_jnt_values)
        return is_collided

    def pipe_cfgrasp_jnv(self, pipe_name, reference_grasp_pose_rotmat=None):
        # return collision-free grasp joint value given current obj_pose.
        collision_free_grasp_jnv = []
        obj_pose = self.obj_collection.obj_collection[pipe_name].obj_mdl.homomat
        grasps_collection = self.obj_collection.obj_collection[pipe_name].grasp_collection
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
            ik_sol_init = self.dual_robot_sys.ik(tgt_pos=rbt_ee_pose_init[:3, 3],
                                                 tgt_rotmat=rbt_ee_pose_init[:3, :3],
                                                 seed_jnt_values=self.dual_robot_sys.get_jnt_values())
            if ik_sol_init is not None:
                if self.collision_check(ik_sol_init) is False:
                    collision_free_grasp_jnv.append(ik_sol_init)
            if reference_grasp_pose_rotmat is not None and \
                    np.allclose(reference_grasp_pose_rotmat, grasp.ac_rotmat, rtol=1e-3):
                break
        if collision_free_grasp_jnv is None:
            print("No collision-free grasp joint value")
        return collision_free_grasp_jnv

    def back_to_init_path(self):
        # back to the initial position
        self.dual_robot_sys.use_lft()
        back_to_init_path_lft = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                       goal_conf=np.radians(self.init_jnt_values_lft),
                                                       obstacle_list=[], other_robot_list=[],
                                                       ext_dist=.4, max_time=300)
        if self.path_check(back_to_init_path_lft):
            print(">>> Cannot generate the path to move the object to the waiting position")
            return
        # else:
        #     self.show_path(back_to_init_path_lft.jv_list)
        self.dual_robot_sys.goto_given_conf(jnt_values=back_to_init_path_lft.jv_list[-1])

        self.dual_robot_sys.use_rgt()
        back_to_init_path_rgt = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                       goal_conf=np.radians(self.init_jnt_values_rgt),
                                                       obstacle_list=[], other_robot_list=[],
                                                       ext_dist=.4, max_time=300)
        if self.path_check(back_to_init_path_rgt):
            print(">>> Cannot generate the path to move the object to the waiting position")
            return
        # else:
        #     self.show_path(back_to_init_path_rgt.jv_list)
        self.dual_robot_sys.goto_given_conf(jnt_values=back_to_init_path_rgt.jv_list[-1])

        return [back_to_init_path_lft.jv_list, back_to_init_path_rgt.jv_list]

    def pipe_load_operation(self, pipe_name, pull_back_vec = np.array([0.061, 0, 0.01]), pipe_up_dist = 0.043,
                            collect_mesh=False):
        '''
        Assume the left arm is waiting for the pipe at the waiting position.
        '''
        # right arm
        self.dual_robot_sys.use_rgt()

        # get collision - free grasp joint value
        pipe_grasp_jnv = self.pipe_cfgrasp_jnv(pipe_name,
                                               reference_grasp_pose_rotmat=np.dot(rm.rotmat_from_euler(np.pi, 0, 0),
                                                                           rm.rotmat_from_euler(0, -np.pi/2, 0)))[0]
        # init -> pipe_grasp (left arm)
        # self.dual_robot_sys.goto_given_conf(jnt_values=pipe_grasp_jnv)
        #self.update_system()
        arm_jnt_values = self.dual_robot_sys.get_jnt_values()
        rgt_to_pipe_path = self.rrtc_planner.plan(start_conf=np.array(arm_jnt_values),
                                                  goal_conf=np.array(pipe_grasp_jnv), # chose one
                                                  obstacle_list=[],
                                                  other_robot_list=[],
                                                  ext_dist=.1,
                                                  max_time=300)
        if self.path_check(rgt_to_pipe_path):
            print(">>> Cannot generate the path to grasp the object")
            return
        else:
           # self.show_path(rgt_to_pipe_path.jv_list)
            if collect_mesh:
                self.extend_mesh(rgt_to_pipe_path.jv_list)

        # hold the pipe
        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_to_pipe_path.jv_list[-1])
        self.dual_robot_sys.hold(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)


        # pull back the pipe
        rgt_pipe_pullback_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                               start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                               goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + pull_back_vec,
                                                               goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                               seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                               granularity=0.01)
        if self.path_check(rgt_pipe_pullback_path):
            print(">>> Cannot generate the path to pull back the object")
            return
        else:
            #self.show_path(rgt_pipe_pullback_path)
            if collect_mesh:
                self.extend_mesh(rgt_pipe_pullback_path)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pipe_pullback_path[-1])

        ''' - WRS中计算
        self.dual_robot_sys.goto_given_conf(jnt_values=lft_to_reload_path.jv_list[-1])
        tgt_rotmat = self.dual_robot_sys.gl_tcp_rotmat @ rm.rotmat_from_euler(0, np.pi, 0)
        tgt_pos = self.dual_robot_sys.gl_tcp_pos - tgt_rotmat @ np.array([0, 0, 0.335 + 0.05])
        self.obj_collection.set_object_config("pipe_50ml_dummy", tgt_pos, tgt_rotmat)

        # caculate the feasible grasp configuration of the target pipe.
        self.dual_robot_sys.use_rgt()
        reload_cfgrasp_jnv = self.pipe_cfgrasp_jnv("pipe_50ml_dummy")
        if len(reload_cfgrasp_jnv) == 0:
            print("No collision-free grasp joint value")
            return
        else:
            reload_cfgrasp_jnv = reload_cfgrasp_jnv[np.argmin(np.linalg.norm(reload_cfgrasp_jnv - self.dual_robot_sys.get_jnt_values(), axis=1))]
            tgt_pos, tgt_rotmat = self.dual_robot_sys.fk(reload_cfgrasp_jnv)
        rgt_pipe_to_reload_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                             start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                             goal_tcp_pos=tgt_pos,
                                                             goal_tcp_rotmat=tgt_rotmat,
                                                             seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                             granularity=0.01)
        '''
        rgt_pipe_to_reload_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                    goal_conf=np.radians(self.args.pipe_reload.rgt_arm_reload_jnv),
                                                    obstacle_list=[], other_robot_list=[],
                                                    ext_dist=.1, max_time=300)

        if self.path_check(rgt_pipe_to_reload_path):
            print(">>> rgt_arm: Cannot generate the path to move the object to the reload position")
            return
        else:
            #self.show_path(rgt_pipe_to_reload_path.jv_list)
            if collect_mesh:
                self.extend_mesh(rgt_pipe_to_reload_path.jv_list)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pipe_to_reload_path.jv_list[-1])

        # caculate the target position and rotation matrix of the pipe, and put the dummy pipe to the target position.
        self.dual_robot_sys.use_lft()
        lft_to_reload_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                    goal_conf=np.radians(self.args.pipe_reload.lft_arm_reload_jnv),
                                                    obstacle_list=[], other_robot_list=[],
                                                    ext_dist=.1, max_time=300)
        if self.path_check(lft_to_reload_path):
            print(">>> Cannot generate the path to the reload position.")
            return
        else:
            #self.show_path(lft_to_reload_path.jv_list)
            if collect_mesh:
                self.extend_mesh(lft_to_reload_path.jv_list)
        self.dual_robot_sys.goto_given_conf(jnt_values=lft_to_reload_path.jv_list[-1])

        self.dual_robot_sys.use_rgt()
        rgt_pipe_up_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                         start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                         goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, pipe_up_dist]),
                                                         goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                         seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                         granularity=0.01)
        if self.path_check(rgt_pipe_up_path):
            print(">>> right_arm: Cannot generate the path to move the object to the reload position")
            return
        else:
            #self.show_path(rgt_pipe_up_path)
            if collect_mesh:
                self.extend_mesh(rgt_pipe_up_path)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pipe_up_path[-1])
        self.dual_robot_sys.release(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)

        # hold the pipe with the left arm
        self.dual_robot_sys.use_lft()
        self.dual_robot_sys.hold(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)

        # right arm -> waiting position
        self.dual_robot_sys.use_rgt()
        rgt_reload_to_waiting_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                            goal_conf=np.radians(self.args.pipe_reload.rgt_arm_waiting_jnv),
                                                            obstacle_list=[], other_robot_list=[],
                                                            ext_dist=.1, max_time=300)
        if self.path_check(rgt_reload_to_waiting_path):
            print(">>> right_arm: Cannot generate the path to move the object to the waiting position")
            return
        else:
            #self.show_path(rgt_reload_to_waiting_path.jv_list)
            if collect_mesh:
                self.extend_mesh(rgt_reload_to_waiting_path.jv_list)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_reload_to_waiting_path.jv_list[-1])

        return [rgt_to_pipe_path.jv_list, rgt_pipe_pullback_path, rgt_pipe_to_reload_path.jv_list,
                lft_to_reload_path.jv_list, rgt_pipe_up_path, rgt_reload_to_waiting_path.jv_list]

    def pipe_deposit_operation(self, pipe_name, pipe_down_dist = 0.043, to_pullback_vec = np.array([-0.061, 0, 0.003]) ,
                               collect_mesh=False):
        '''
        Assume the left arm is holding the pipe at the reload position. rgt arm at the waiting position.
        '''
        # left arm
        self.dual_robot_sys.use_lft()
        lft_to_reload_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                    goal_conf=np.radians(self.args.pipe_reload.lft_arm_reload_jnv),
                                                    obstacle_list=[], other_robot_list=[],
                                                    ext_dist=.1, max_time=300)
        if self.path_check(lft_to_reload_path):
            print(">>> Cannot generate the path to move the object to the reload position")
            return
        else:
            #self.show_path(lft_to_reload_path.jv_list)
            if collect_mesh:
                self.extend_mesh(lft_to_reload_path.jv_list)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_to_reload_path.jv_list[-1])

        # right arm
        self.dual_robot_sys.use_rgt()

        # get collision - free grasp joint value
        pipe_grasp_jnv = self.pipe_cfgrasp_jnv("pipe_50ml_dummy")
        if len(pipe_grasp_jnv) == 0:
            print("No collision-free grasp joint value")
            return

        ''' - WRS中计算
        reload_cfgrasp_jnv = pipe_grasp_jnv[np.argmin(np.linalg.norm(pipe_grasp_jnv - self.dual_robot_sys.get_jnt_values(), axis=1))]
        rgt_wait_to_pipe_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                goal_conf=np.array(reload_cfgrasp_jnv),  # chose one
                                                obstacle_list=[],
                                                other_robot_list=[],
                                                ext_dist=.3,
                                                max_time=300)
        '''
        rgt_wait_to_pipe_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                goal_conf=np.radians(self.args.pipe_reload.rgt_arm_reload_jnv),
                                                obstacle_list=[],
                                                other_robot_list=[],
                                                ext_dist=.3,
                                                max_time=300)

        if self.path_check(rgt_wait_to_pipe_path):
            print(">>> Cannot generate the path to grasp the object")
            return
        else:
            #self.show_path(rgt_wait_to_pipe_path.jv_list)
            if collect_mesh:
                self.extend_mesh(rgt_wait_to_pipe_path.jv_list)

        # hold the pipe right arm
        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_wait_to_pipe_path.jv_list[-1])
        self.dual_robot_sys.hold(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)

        # release the pipe left arm
        self.dual_robot_sys.use_lft()
        self.dual_robot_sys.release(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)

        # pull out the pipe. along with the y-axis
        self.dual_robot_sys.use_rgt()
        rgt_pipe_to_pullout_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                              start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                              goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, -pipe_down_dist]),
                                                              goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                              seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                              granularity=0.01)
        if self.path_check(rgt_pipe_to_pullout_path):
            print(">>> Cannot generate the path to pull out the object")
            return
        else:
            #self.show_path(rgt_pipe_to_pullout_path)
            # self.show_path(troppy_path)

            if collect_mesh:
                self.extend_mesh(rgt_pipe_to_pullout_path)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pipe_to_pullout_path[-1])

        # put the pipe to the waiting position.
        if pipe_name == "pipe_50ml_grasp":
            pipe_grasp_rot = np.radians(args.pipe_grasp.rot)
            self.obj_collection.set_object_config("pipe_50ml_dummy",
                                                  np.array(args.pipe_grasp.pos) + np.array([0.05, 0, 0]),
                                                  rm.rotmat_from_euler(pipe_grasp_rot[0], pipe_grasp_rot[1],
                                                                       pipe_grasp_rot[2]))
        elif pipe_name == "pipe_50ml_replace":
            pipe_grasp_rot = np.radians(args.pipe_replace.rot)
            self.obj_collection.set_object_config("pipe_50ml_dummy", np.array(args.pipe_replace.pos) + np.array([0.05, 0, 0]),
                                                                     rm.rotmat_from_euler(pipe_grasp_rot[0], pipe_grasp_rot[1],
                                                                                          pipe_grasp_rot[2]))
        pipe_grasp_jnv = self.pipe_cfgrasp_jnv("pipe_50ml_dummy", reference_grasp_pose_rotmat=np.dot(rm.rotmat_from_euler(np.pi, 0, 0),
                                                                           rm.rotmat_from_euler(0, -np.pi/2, 0)))
        if len(pipe_grasp_jnv) == 0:
            print("No collision-free grasp joint value")
            return
        else:
            pipe_grasp_jnv = pipe_grasp_jnv[np.argmin(np.linalg.norm(pipe_grasp_jnv - self.dual_robot_sys.get_jnt_values(), axis=1))]
            tgt_pos, tgt_rotmat = self.dual_robot_sys.fk(pipe_grasp_jnv)
        rgt_pullout_to_pre_release_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                                        start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                        goal_tcp_pos=tgt_pos,
                                                                        goal_tcp_rotmat=tgt_rotmat,
                                                                        seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                                        granularity=0.01)
        if self.path_check(rgt_pullout_to_pre_release_path):
            print(">>> Cannot generate the path to move the object to the pre_release position")
            return
        else:
            #self.show_path(rgt_pullout_to_pre_release_path)
            if collect_mesh:
                self.extend_mesh(rgt_pullout_to_pre_release_path)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pullout_to_pre_release_path[-1])
        rgt_pre_release_to_release_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                                     start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                     goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + to_pullback_vec,
                                                                     goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                     seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                                     granularity=0.01)
        if self.path_check(rgt_pre_release_to_release_path):
            print(">>> Cannot generate the path to move the object to the release position")
            return
        else:
            #self.show_path(rgt_pre_release_to_release_path)
            if collect_mesh:
                self.extend_mesh(rgt_pre_release_to_release_path)

        self.dual_robot_sys.goto_given_conf(jnt_values=rgt_pre_release_to_release_path[-1])
        self.dual_robot_sys.release(obj_cmodel=self.obj_collection.obj_collection[pipe_name].obj_mdl)

        self.dual_robot_sys.use_lft()
        lft_reload_to_standby_path = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                                        start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                        goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0.05, 0.03]),
                                                                        goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                        seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                                        granularity=0.01)
        if self.path_check(lft_reload_to_standby_path):
            print(">>> Cannot generate the path to move the object to the waiting position")
            return
        else:
            #self.show_path(lft_reload_to_standby_path)
            if collect_mesh:
                self.extend_mesh(lft_reload_to_standby_path)
        self.dual_robot_sys.goto_given_conf(jnt_values=lft_reload_to_standby_path[-1])

        self.dual_robot_sys.use_rgt()
        rgt_release_to_waiting_path = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                             goal_conf=np.radians(self.args.pipe_reload.rgt_arm_waiting_jnv),  # chose one
                                                             obstacle_list=[],
                                                             other_robot_list=[],
                                                             ext_dist=.1,
                                                             max_time=300)
        if self.path_check(rgt_release_to_waiting_path):
            print(">>> Cannot generate rgt_release_to_waiting_path ")
        else:
            #self.show_path(rgt_release_to_waiting_path.jv_list)
            if collect_mesh:
                self.extend_mesh(rgt_release_to_waiting_path.jv_list)
        self.dual_robot_sys.goto_given_conf(rgt_release_to_waiting_path.jv_list[-1])

        # rgt arm still at the reload position.
        return [lft_to_reload_path.jv_list, rgt_wait_to_pipe_path.jv_list, rgt_pipe_to_pullout_path,
                rgt_pullout_to_pre_release_path, rgt_pre_release_to_release_path, lft_reload_to_standby_path,
                rgt_release_to_waiting_path.jv_list]

    def liquid_moving(self, from_position_jnv, to_position_jnv, dive_depth=.13, collect_mesh=False):
        '''
        parameters: from_position_jnv, to_position_jnv, absorb_time, release_time
            from_position_jnv: the joint values of the left arm to absorb the liquid
            to_position_jnv: the joint values of the left arm to release the liquid
            dive_depth: the depth of the diving.
        '''

        # liquid moving operation only for the left arm
        self.dual_robot_sys.use_lft()
        if not self.dual_robot_sys.is_hold():
            print("The left arm is not holding the object")
            return

        # move to the absorb position - linear motion
        lft_absorb_path_move = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                        goal_conf=np.radians(from_position_jnv),
                                                        obstacle_list=[], other_robot_list=[],
                                                        ext_dist=.1, max_time=300)

        if self.path_check(lft_absorb_path_move):
            print(">>> Cannot generate the path to absorb the liquid")
            return
        else:
            #self.show_path(lft_absorb_path_move.jv_list)
            # toppra_path = move_jntspace_path(path = lft_absorb_path_move.jv_list,
            #                    max_jntacc=[1.2]* lft_absorb_path_move.jv_list[0].shape[0],
            #                    max_jntvel=[1.2]* lft_absorb_path_move.jv_list[0].shape[0],
            #                    control_frequency=1/100, toggle_debug=True)
            if collect_mesh:
                self.extend_mesh(lft_absorb_path_move.jv_list)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_absorb_path_move.jv_list[-1])

        # absorb the liquid
        lft_absorb_path_down = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                          start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                          goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, -dive_depth]),
                                                          goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                          seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                          granularity=0.01)
        if self.path_check(lft_absorb_path_down):
            print(">>> Cannot generate the path to absorb the liquid")
            return
        else:
            #self.show_path(lft_absorb_path_down)
            if collect_mesh:
                self.extend_mesh(lft_absorb_path_down)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_absorb_path_down[-1])

        # move to the release position - linear motion
        lft_absorb_path_up = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                            start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                            goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, dive_depth]),
                                                            goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                            seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                            granularity=0.01)
        if self.path_check(lft_absorb_path_up):
            print(">>> Cannot generate the path to release the liquid")
            return
        else:
            #self.show_path(lft_absorb_path_up)
            if collect_mesh:
                self.extend_mesh(lft_absorb_path_up)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_absorb_path_up[-1])

        lft_release_path_move = self.rrtc_planner.plan(start_conf=self.dual_robot_sys.get_jnt_values(),
                                                       goal_conf=np.radians(to_position_jnv),
                                                       obstacle_list=[], other_robot_list=[],
                                                       ext_dist=.1, max_time=300)
        if self.path_check(lft_release_path_move):
            print(">>> Cannot generate the path to release the liquid")
            return
        else:
            #self.show_path(lft_release_path_move.jv_list)
            if collect_mesh:
                self.extend_mesh(lft_release_path_move.jv_list)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_release_path_move.jv_list[-1])
        lft_release_path_down = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                              start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                              goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, -dive_depth]),
                                                              goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                              seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                              granularity=0.01)
        if self.path_check(lft_release_path_down):
            print(">>> Cannot generate the path to release the liquid")
            return
        else:
            # self.show_path(lft_release_path_down)
            if collect_mesh:
                self.extend_mesh(lft_release_path_down)

        self.dual_robot_sys.goto_given_conf(jnt_values=lft_release_path_down[-1])
        lft_release_path_up = self.inink_svlr.gen_linear_motion(start_tcp_pos=self.dual_robot_sys.gl_tcp_pos,
                                                                  start_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                  goal_tcp_pos=self.dual_robot_sys.gl_tcp_pos + np.array([0, 0, dive_depth]),
                                                                  goal_tcp_rotmat=self.dual_robot_sys.gl_tcp_rotmat,
                                                                  seed_jnt_values=self.dual_robot_sys.get_jnt_values(),
                                                                  granularity=0.01)
        if self.path_check(lft_release_path_up):
            print(">>> Cannot generate the path to release the liquid")
            return
        else:
            #self.show_path(lft_release_path_up)
            if collect_mesh:
                self.extend_mesh(lft_release_path_up)
        self.dual_robot_sys.goto_given_conf(jnt_values=lft_release_path_up[-1])

        return [lft_absorb_path_move.jv_list, lft_absorb_path_down, lft_absorb_path_up,
                lft_release_path_move.jv_list, lft_release_path_down, lft_release_path_up]


class anaimation_data(object):
    def __init__(self, robot_mesh_list, obj_mesh_list):
        self.counter = 0
        self.robot_mesh_list = robot_mesh_list
        self.obj_mesh_list = obj_mesh_list

    def __len__(self):
        return len(self.robot_mesh_list)


def update_animation(anime_data,  task):
    if anime_data.counter > 0:
        anime_data.robot_mesh_list[anime_data.counter - 1].detach()
        anime_data.obj_mesh_list[anime_data.counter - 1].detach()

    if anime_data.counter >= len(anime_data):
        anime_data.counter = 0

    robot_mesh_model = anime_data.robot_mesh_list[anime_data.counter]
    obj_mesh_model = anime_data.obj_mesh_list[anime_data.counter]
    robot_mesh_model.attach_to(base)
    obj_mesh_model.attach_to(base)

    # mesh_model.show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


def move_jntspace_path(path,
                       max_jntvel: list = None,
                       max_jntacc: list = None,
                       start_frame_id=1,
                       control_frequency=1 / 33,
                       toggle_debug=False):
    if TOPPRA_EXIST:
        result = []
        # enter servo mode
        if not path or path is None:
            raise ValueError("The given is incorrect!")
        # Refer to https://www.ufactory.cc/_files/ugd/896670_9ce29284b6474a97b0fc20c221615017.pdf
        # the robotic arm can accept joint position commands sent at a fixed high frequency like 33 Hz
        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                            control_frequency=control_frequency,
                                                            max_vels=max_jntvel,
                                                            max_accs=max_jntacc,
                                                            toggle_debug=toggle_debug)
        interpolated_path = interpolated_path[start_frame_id:]
        for jnt_values in interpolated_path:
            result.append(np.rad2deg(jnt_values))
        return result
    else:
        raise NotImplementedError

# 自定义 JSON 编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



if __name__ == "__main__":
    import json
    import box


    base = wd.World(cam_pos=[3, 0, 0.35], lookat_pos=[0, 0, 0.35])
    mgm.gen_frame().attach_to(base)
    hugroup_lab_nova2_dual = nova2_wrsv3gripper_dual.hugroup_lab_nova2_dual(enable_cc=True)

    with open("/wrs/HuGroup_Qin/demo/config.json", 'r') as file:
        args = json.load(file)
    args=box.Box(args)

    pipe_load_operation_path = dict()
    liquid_dropping_path = dict()
    pipe1_deposit_operation_path = dict()

    pipe_load_replace_path = dict()
    liquid_adding_path = dict()
    pipe2_deposit_operation_path = dict()

    path_dir = "/wrs/HuGroup_Qin/demo/path_jnv.json"
    collect_mesh = True
    # create the object collection
    obj_collection = ObjectCollection()
    pipe_grasp = ObjetType(obj_mdl_path=args.pipe_path, obj_mdl_pos=args.pipe_grasp.pos,
                           obj_mdl_rotmat=np.array(np.radians(args.pipe_grasp.rot)),
                           grasp_path=args.pipe_grasp_path, name="pipe_50ml_grasp")
    obj_collection.add_object(pipe_grasp)

    pipe_release = ObjetType(obj_mdl_path=args.pipe_path, obj_mdl_pos=args.pipe_grasp.pos,
                             obj_mdl_rotmat=np.array(np.radians(args.pipe_grasp.rot)),
                             grasp_path=args.pipe_release_path, name="pipe_50ml_dummy")
    obj_collection.add_object(pipe_release)

    pipe_release = ObjetType(obj_mdl_path=args.pipe_path, obj_mdl_pos=args.pipe_replace.pos,
                             obj_mdl_rotmat=np.array(np.radians(args.pipe_replace.rot)),
                             grasp_path=args.pipe_grasp_path, name="pipe_50ml_replace")
    obj_collection.add_object(pipe_release)

    # create the liquid handling task
    liquid_handling_task = LiquidHandlingTask(args, base, hugroup_lab_nova2_dual, obj_collection)

    [pipe_load_operation_path['rgt_to_pipe_path'], pipe_load_operation_path['rgt_pipe_pullback_path'], pipe_load_operation_path['rgt_pipe_to_reload_path'],
     pipe_load_operation_path['lft_to_reload_path'], pipe_load_operation_path['rgt_pipe_up_path'], pipe_load_operation_path['rgt_reload_to_waiting_path']] = \
        liquid_handling_task.pipe_load_operation("pipe_50ml_grasp", collect_mesh=collect_mesh)

    [liquid_dropping_path['lft_absorb_path_move'], liquid_dropping_path['lft_absorb_path_down'], liquid_dropping_path['lft_absorb_path_up'],
     liquid_dropping_path['lft_release_path_move'], liquid_dropping_path['lft_release_path_down'], liquid_dropping_path['lft_release_path_up']] = \
        liquid_handling_task.liquid_moving(from_position_jnv=args.liquid_moving.lft_arm_petri_position_jnv,
                                       to_position_jnv=args.liquid_moving.lft_arm_waste_release_position_jnv, collect_mesh=collect_mesh)

    [pipe1_deposit_operation_path['lft_to_reload_path'], pipe1_deposit_operation_path['rgt_wait_to_pipe_path'], pipe1_deposit_operation_path['rgt_pipe_to_pullout_path'],
     pipe1_deposit_operation_path['rgt_pullout_to_pre_release_path'], pipe1_deposit_operation_path['rgt_pre_release_to_release_path'],
     pipe1_deposit_operation_path['lft_reload_to_standby_path'], pipe1_deposit_operation_path['rgt_release_to_waiting_path']]=\
        liquid_handling_task.pipe_deposit_operation("pipe_50ml_grasp", collect_mesh=collect_mesh)

    [pipe_load_replace_path['rgt_to_pipe_path'], pipe_load_replace_path['rgt_pipe_pullback_path'], pipe_load_replace_path['rgt_pipe_to_reload_path'],
     pipe_load_replace_path['lft_to_reload_path'], pipe_load_replace_path['rgt_pipe_up_path'], pipe_load_replace_path['rgt_reload_to_waiting_path']] = \
        liquid_handling_task.pipe_load_operation("pipe_50ml_replace", pull_back_vec = np.array([0.065, 0, 0.01]), collect_mesh=collect_mesh)

    [liquid_adding_path['lft_absorb_path_move'], liquid_adding_path['lft_absorb_path_down'], liquid_adding_path['lft_absorb_path_up'],
     liquid_adding_path['lft_release_path_move'], liquid_adding_path['lft_release_path_down'], liquid_adding_path['lft_release_path_up']] = \
        liquid_handling_task.liquid_moving(from_position_jnv=args.liquid_moving.lft_arm_liquid_absort_position_jnv,
                                       to_position_jnv=args.liquid_moving.lft_arm_petri_position_jnv, collect_mesh=collect_mesh)

    [pipe2_deposit_operation_path['lft_to_reload_path'], pipe2_deposit_operation_path['rgt_wait_to_pipe_path'], pipe2_deposit_operation_path['rgt_pipe_to_pullout_path'],
     pipe2_deposit_operation_path['rgt_pullout_to_pre_release_path'], pipe2_deposit_operation_path['rgt_pre_release_to_release_path'],
     pipe2_deposit_operation_path['lft_reload_to_standby_path'], pipe2_deposit_operation_path['rgt_release_to_waiting_path']]=\
        liquid_handling_task.pipe_deposit_operation("pipe_50ml_replace", collect_mesh=collect_mesh)

    data = {
        "pipe_load_operation_path": pipe_load_operation_path,
        "liquid_dropping_path": liquid_dropping_path,
        "pipe1_deposit_operation_path": pipe1_deposit_operation_path,
        "pipe_load_replace_path": pipe_load_replace_path,
        "liquid_adding_path": liquid_adding_path,
        "pipe2_deposit_operation_path": pipe2_deposit_operation_path
    }

    with open(path_dir, 'w') as json_file:
        print(">>> Save the path data to the path_jnv.json file")
        json.dump(data, json_file, cls=NumpyEncoder, indent=4)

    anime_data = anaimation_data(liquid_handling_task.robot_mesh_list, liquid_handling_task.obj_mesh_list)
    # Animation
    taskMgr.doMethodLater(0.01, update_animation, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()