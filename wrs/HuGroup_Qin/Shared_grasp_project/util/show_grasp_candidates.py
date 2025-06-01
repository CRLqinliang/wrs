import sys
sys.path.append("E:/Qin/wrs")
import pickle
from wrs import wd, gpa, mcm, rm, cbt
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as ee
mesh_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl"

base = wd.World(cam_pos=rm.np.array([.5, .5, .5]), lookat_pos=rm.np.array([0, 0, 0]))
obj_cmodel = mcm.CollisionModel(mesh_path, alpha=0.8)
obj_cmodel.show_local_frame()
obj_cmodel.attach_to(base)

gripper = ee.WRSGripper3()
def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
# grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_grasp.pickle")
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           obj_cmodel,
                                           angle_between_contact_normals=rm.np.radians(180),
                                           rotation_interval=rm.np.radians(30),
                                           max_samples=50,
                                           min_dist_between_sampled_contact_points=.003,
                                           contact_offset=.0015,
                                           min_thickness=.0005,
                                           toggle_dbg=False)
print(grasp_collection)
# grasp_collection.save_to_disk(grasp_path)
# grasp_collection = grasp_load(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_57.pickle")
# for grasp in grasp_collection:
#     gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
#     gripper.gen_meshmodel(alpha=.4).attach_to(base)
# base.run()

counter = [0]
on_screen = []


def update(on_screen, counter, gripper, grasp_collection, task):
    for item in on_screen:
        item.detach()
    if counter[0] == len(grasp_collection):
        counter[0] = 0
    grasp = grasp_collection[counter[0]]
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    on_screen.append(gripper.gen_meshmodel(alpha=1))
    on_screen[-1].attach_to(base)
    if base.inputmgr.keymap['space']:
        counter[0] += 1

    return task.cont


taskMgr.doMethodLater(.01, update, "update",
                      extraArgs=[on_screen, counter, gripper, grasp_collection],
                      appendTask=True)
base.run()
