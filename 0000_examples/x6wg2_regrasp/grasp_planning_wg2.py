import os
from wrs import wd, rm, gpa, mcm
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v2 as ee

mesh_name = "bunnysim"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name+".stl")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_grasp.pickle")

base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel(mesh_path)
obj_cmodel.attach_to(base)

gripper = ee.WRSGripper2()
# grippers.gen_meshmodel().attach_to(base)
# base.run()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           obj_cmodel,
                                           angle_between_contact_normals=rm.radians(175),
                                           rotation_interval=rm.radians(15),
                                           max_samples=100,
                                           min_dist_between_sampled_contact_points=.005,
                                           contact_offset=.001,
                                           toggle_dbg=False)
print(grasp_collection)
grasp_collection.save_to_disk(grasp_path)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()