""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240619 Osaka Univ.

"""
import basis.robot_math as rm
import modeling.collision_model as cm
import numpy as np
import visualization.panda.world as wd
import robot_sim.end_effectors.gripper.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from grasping.grasp import GraspCollection


# define the grasp poses for pipe
def pipe_gripper_grasps_with_rotation_along_pipezaxis(gripper, obj_cmodel, jaw_center_pos,
                                      approaching_direction, thumb_opening_direction,
                                      jaw_width, rotation_interval=np.radians(30),
                                      rotation_range=(-np.radians(180), np.radians(180)),
                                      toggle_flip=False, toggle_dbg=False):
    grasp_collection = GraspCollection()
    for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
        rotated_rotmat = rm.rotmat_from_axangle([0, 0, 1], rotate_angle)
        rotated_approaching_direction = np.dot(rotated_rotmat, approaching_direction)
        rotated_thumb_opening_direction = np.dot(rotated_rotmat, thumb_opening_direction)
        grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                   approaching_direction=rotated_approaching_direction,
                                   thumb_opening_direction=rotated_thumb_opening_direction,
                                   jaw_width=jaw_width)
        if not gripper.is_mesh_collided([obj_cmodel]):
            grasp_collection.append(grasp)

    if toggle_flip:
        for rotate_angle in np.arange(rotation_range[0], rotation_range[1], rotation_interval):
            rotated_rotmat = rm.rotmat_from_axangle([0, 0, 1], rotate_angle)
            rotated_approaching_direction = np.dot(rotated_rotmat, approaching_direction)
            rotated_thumb_opening_direction = np.dot(rotated_rotmat, -thumb_opening_direction)
            grasp = gripper.grip_at_by_twovecs(jaw_center_pos=jaw_center_pos,
                                               approaching_direction=rotated_approaching_direction,
                                               thumb_opening_direction=rotated_thumb_opening_direction,
                                               jaw_width=jaw_width)
            if not gripper.is_mesh_collided([obj_cmodel]):
                grasp_collection.append(grasp)

    return grasp_collection

# create the virtual environment
base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
gripper = wrs_gripper_v3.WRSGripper3()

# Generate collision model and attach to the virtual environment
objpath = "H:\Qin\wrs\HuGroup_Qin\objects\meshes\pipe_50ml.stl"
objcm = cm.CollisionModel(objpath)
objcm.attach_to(base)
objcm.show_local_frame()

# define the grasp poses along with the axis of pipe
grasp_collection = pipe_gripper_grasps_with_rotation_along_pipezaxis(gripper=gripper,
                                                      obj_cmodel=objcm,
                                                      jaw_center_pos=np.array([0, 0, 0.25]),
                                                      approaching_direction=np.array([0, 1, 0]),
                                                      thumb_opening_direction=np.array([1, 0, 0]),
                                                      jaw_width=0.0085, rotation_interval=np.radians(15), toggle_flip=True,
                                                      toggle_dbg=False)

print("Number of grasps is", len(grasp_collection))

# save the grasp_info_list
grasp_collection.save_to_disk("../data/grasps/pipe_50ml_grasps_release.pkl")

# show all the grasp poses
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                            jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.3).attach_to(base)

base.run()

