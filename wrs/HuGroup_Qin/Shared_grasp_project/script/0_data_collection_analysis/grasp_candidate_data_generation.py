""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241201 Osaka Univ.

"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append("E:/Qin/wrs")
import pickle
import wrs.basis.robot_math as rm
import wrs.basis.constant as ct
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.grasping.planning.antipodal as gpa
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
from wrs.manipulation.placement.flatsurface import FSReferencePoses
from wrs.grasping.reasoner import GraspReasoner
from tqdm import tqdm

def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def grasp_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def fs_reference_poses_collection(gripper, obj_cmodel):
    fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
    grasp_collection_set = []
    for pose in fs_reference_poses:
        grasp_collection_set.append(gpa.plan_gripper_grasps(
            gripper,
            obj_cmodel,
            angle_between_contact_normals=rm.np.radians(160),
            rotation_interval=rm.np.radians(90),
            max_samples=20
        ))
    return fs_reference_poses, grasp_collection_set


def grasp_collection_show(gripper, grasp_collection_set, cmodel_list):
    for grasp_collection in grasp_collection_set:
        for grasp in grasp_collection:
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            if gripper.is_mesh_collided(cmodel_list=cmodel_list, toggle_dbg=False):
                continue
            gripper.gen_meshmodel(alpha=.5).attach_to(base)
    base.run()

def show_obj(name, obj_path, pos, rotmat, rgb=None, alpha=None):
    # we only consider SE(2) DOF for the object
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha, initor=obj_path)
    obj_cmodel.pos = pos
    obj_cmodel.rotmat = rotmat
    obj_cmodel.show_local_frame()
    obj_cmodel.attach_to(base)

def show_feasiblt_grasp(gripper, grasp_collection):
    # show the feasible grasps
    for grasp in grasp_collection:
        gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                                jaw_width=grasp.ee_values)
        gripper.gen_meshmodel(rgb=rm.const.green, alpha=.3).attach_to(base)
    base.run()

def show_grasp(obj_init_pos, obj_init_rotmat, grasp_id, object_feasible_grasps, gripper):
    if grasp_id is None:
        print("没有找到可行抓取！")
        return None

    # 显示可行抓取
    for index in range(len(grasp_id)):
        grasp = object_feasible_grasps._grasp_list[grasp_id[index]]
        
        # 将局部坐标系下的抓取位置转换到世界坐标系
        jaw_center_pos = obj_init_pos + obj_init_rotmat @ grasp.ac_pos
        # 将局部坐标系下的抓取旋转矩阵转换到世界坐标系
        jaw_center_rotmat = obj_init_rotmat @ grasp.ac_rotmat
        
        # 显示抓取器
        gripper.grip_at_by_pose(jaw_center_pos=jaw_center_pos,
                               jaw_center_rotmat=jaw_center_rotmat,
                               jaw_width=grasp.ee_values)
        gripper.gen_meshmodel(rgb=rm.const.hug_blue, alpha=.2).attach_to(base)
    base.run()


def analyze_object_grasps(obj_path, samples_list=[10, 30, 50, 70]):
    """分析单个物体的抓取数据"""
    results = []
    obj_name = Path(obj_path).stem
    obj_cmodel = mcm.CollisionModel(initor=obj_path)

    # 初始化机器人和抓取器
    robot = nova2_gripper_v3()
    robot.gen_meshmodel().attach_to(base)
    gripper = wrs_gripper_v3.WRSGripper3()
    
    # 获取稳定放置姿态
    fs_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
    stable_poses = list(fs_poses)
    
    for max_samples in samples_list:
        # 计算抓取候选
        grasp_candidates = gpa.plan_gripper_grasps(
            gripper,
            obj_cmodel,
            angle_between_contact_normals=rm.np.radians(160),
            rotation_interval=rm.np.radians(90),
            max_samples=max_samples
        )

        # 创建GraspReasoner实例
        grasp_reasoner = GraspReasoner(robot, grasp_candidates)
        
        # 对每个稳定位姿计算可行抓取
        for pose_idx, stable_pose in enumerate(stable_poses):
            obj_pos = stable_pose[0].copy()
            obj_rotmat = stable_pose[1].copy()
            obj_pos[0], obj_pos[1] = 0.0, 0.5

            # 显示物体
            obj_cmodel.pos = obj_pos
            obj_cmodel.rotmat = obj_rotmat
            obj_cmodel.show_local_frame()
            obj_cmodel.attach_to(base)


            # 使用GraspReasoner判断可行抓取
            # if pose_idx == 1:
            #     toggle_dbg_flag = True
            # else:
            #     toggle_dbg_flag = False
            feasible_gids, _ = grasp_reasoner.find_feasible_gids(
                goal_pose=[obj_pos, obj_rotmat], consider_robot=False,
                obstacle_list = robot.body.lnk_list[0].cmodel,
                toggle_dbg=False
            )

            # 显示可行抓取
            # if pose_idx == 1 and feasible_gids is not None:
            #     show_grasp(obj_pos, obj_rotmat, list(set(feasible_gids)), grasp_candidates, gripper)

            feasible_count = len(list(set(feasible_gids))) if feasible_gids is not None else 0
            if len(grasp_candidates) != 0:
                feasible_ratio = (feasible_count / len(grasp_candidates)) * 100
            else:
                feasible_ratio = 0
            
            results.append({
                '物体名称': obj_name,
                '抓取样本数': max_samples,
                '稳定位姿编号': pose_idx + 1,
                '稳定位姿': f"{obj_pos}, {obj_rotmat}",
                '可行抓取比例': f"{feasible_ratio:.2f}%"
            })
            obj_cmodel.detach()
    
    return results


def analyze_all_objects():
    """分析目录下所有STL文件"""
    ycb_dir = Path(r"H:\Qin\wrs\wrs\bench_mark\ycb")
    all_results = []
    
    # 首先获取所有stl文件列表
    stl_files = list(ycb_dir.glob("*.stl"))
    
    # 使用tqdm包装循环
    for stl_file in tqdm(stl_files, desc="分析物体进度"):
        try:
            results = analyze_object_grasps(str(stl_file))
            all_results.extend(results)
        except Exception as e:
            print(f"处理 {stl_file.name} 时出错: {str(e)}")
    
    # 创建数据表格
    df = pd.DataFrame(all_results)
    # 保存到CSV文件
    df.to_csv("grasp_analysis_results.csv", index=False, encoding='utf-8-sig')
    print("分析结果已保存到 grasp_analysis_results.csv")
    return df


if __name__ == '__main__':
    try:
        # 世界配置
        base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
        # mgm.gen_frame().attach_to(base)

        # 对象配置
        # obj_path = r"E:\Qin\wrs\wrs\bench_mark\ycb\power_drill.stl"  # 使用相对路径
        obj_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl"
        # obj_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\mug.stl"
        # obj_path = r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bracketR1.stl"
        # table_path = r"H:\Qin\wrs\wrs\HuGroup_Qin\robot_sim\meshes\regrasp_table_env.stl"
        # obj_path = r"H:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\tubebig.stl"
        # obj_path = r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bunnysim.stl"
        
        obj_cmodel = mcm.CollisionModel(initor=obj_path, rgb=rm.const.steel_gray, alpha=0.8)
        fs_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
        # for index, pose in enumerate(fs_poses):
        #     pose[0][1] = 0
        #     obj_cmodel.pos = pose[0] + np.array([index ,0 ,0]) * 0.2
        #     obj_cmodel.rotmat = pose[1]
        #     obj_cmodel_copy = obj_cmodel.copy()
        #     obj_cmodel_copy.show_local_frame()
        #     obj_cmodel_copy.attach_to(base)
        # base.run()

        # base.run()
        if obj_cmodel is None:
            raise ValueError("无法加载物体模型")

        # 抓取规划
        gripper = wrs_gripper_v3.WRSGripper3()


        # generate grasp data
        grasp_collection_set = gpa.plan_gripper_grasps(
            gripper,
            obj_cmodel,
            angle_between_contact_normals=rm.np.radians(160),
            rotation_interval=rm.np.radians(60),
            max_samples=260
        )
        
        # 显示抓取集合
        print(len(grasp_collection_set))
        # grasp_collection_set.save_to_disk(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Power_drill\power_drill_grasp_{}.pickle".format(len(grasp_collection_set)))
        # grasp_collection_set.save_to_disk(
        #     r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_{}.pickle".format(
        #         len(grasp_collection_set)))

        grasp_collection_set = grasp_load(r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle\bottle_grasp_109.pickle")
        obj_cmodel.show_local_frame()
        # obj_cmodel.attach_to(base)
        for index, grasp in enumerate(grasp_collection_set):
        # grasp = grasp_collection_set.__getitem__(64)
            gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,
                                    jaw_center_rotmat=grasp.ac_rotmat,
                                    jaw_width=grasp.ee_values)
            gripper.gen_meshmodel(alpha=1, rgb = rm.const.hug_blue).attach_to(base)
            # if index >=80:
            #     break
        base.run()
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
