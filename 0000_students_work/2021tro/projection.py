import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as cm
import math
from scipy.spatial import cKDTree

base = wd.World(cam_pos=np.array([-.3,-.7,.42]), lookat_pos=np.array([0,0,0]))
# mgm.gen_frame().attach_to(base)
bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
bowl_model.set_rgba([.3,.3,.3,1])
bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi,0,0))
bowl_model.attach_to(base)

pn_direction = np.array([0, 0, -1])

bowl_samples, bowl_sample_normals = bowl_model.sample_surface(toggle_option='normals', radius=.002)
selection = bowl_sample_normals.dot(-pn_direction)>.1
bowl_samples = bowl_samples[selection]
bowl_sample_normals=bowl_sample_normals[selection]
tree = cKDTree(bowl_samples)

pt_direction = rm.orthogonal_vector(pn_direction, toggle_unit=True)
tmp_direction = np.cross(pn_direction, pt_direction)
plane_rotmat = np.column_stack((pt_direction, tmp_direction, pn_direction))
homomat=np.eye(4)
homomat[:3,:3] = plane_rotmat
homomat[:3,3] = np.array([-.07,-.03,.1])
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[1,1,1,.3])
twod_plane.attach_to(base)

circle_radius=.05
line_segs = [[homomat[:3,3], homomat[:3,3]+pt_direction*.07], [homomat[:3,3]+pt_direction*.07, homomat[:3,3]+pt_direction*.07+tmp_direction*.07],
             [homomat[:3,3]+pt_direction*.07+tmp_direction*.07, homomat[:3,3]+tmp_direction*.07], [homomat[:3,3]+tmp_direction*.07, homomat[:3,3]]]
# mgm.gen_linesegs(line_segs).attach_to(base)
for sec in line_segs:
    gm.gen_stickmodel(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], radius=.002, type='round').attach_to(base)
epos = (line_segs[0][1]-line_segs[0][0])*.7+line_segs[0][0]
gm.gen_arrow(spos=line_segs[0][0], epos=epos, stick_radius=0.004).attach_to(base)
spt = homomat[:3,3]
# mgm.gen_stick(spt, spt + pn_direction * 10, rgba=[0,1,0,1]).attach_to(base)
# base.run()
gm.gen_dashed_arrow(spt, spt - pn_direction * .07, stick_radius=.004).attach_to(base) # p0
cpt, cnrml = bowl_model.ray_hit(spt, spt + pn_direction * 10000, option='closest')
gm.gen_dashed_stick(spt, cpt, rgba=[.57, .57, .57, .7], radius=0.003).attach_to(base)
gm.gen_sphere(pos=cpt, radius=.005).attach_to(base)
gm.gen_dashed_arrow(cpt, cpt - pn_direction * .07, stick_radius=.004).attach_to(base) # p0
gm.gen_dashed_arrow(cpt, cpt + cnrml * .07, stick_radius=.004).attach_to(base) # p0

angle = rm.angle_between_vectors(-pn_direction, cnrml)
vec = np.cross(-pn_direction, cnrml)
rotmat = rm.rotmat_from_axangle(vec, angle)
new_plane_homomat = np.eye(4)
new_plane_homomat[:3,:3] = rotmat.dot(homomat[:3,:3])
new_plane_homomat[:3,3] = cpt
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=new_plane_homomat, rgba=[1,1,1,.3])
twod_plane.attach_to(base)
new_line_segs = [[cpt, cpt+rotmat.dot(pt_direction)*.07],
                 [cpt+rotmat.dot(pt_direction)*.07, cpt+rotmat.dot(pt_direction)*.07+rotmat.dot(tmp_direction)*.07],
                 [cpt+rotmat.dot(pt_direction)*.07+rotmat.dot(tmp_direction)*.07, cpt+rotmat.dot(tmp_direction)*.07],
                 [cpt+rotmat.dot(tmp_direction)*.07, cpt]]
# mgm.gen_linesegs(new_line_segs).attach_to(base)
for sec in new_line_segs:
    gm.gen_stickmodel(spos=sec[0], epos=sec[1], rgba=[0, 0, 0, 1], radius=.002, type='round').attach_to(base)
epos = (new_line_segs[0][1]-new_line_segs[0][0])*.7+new_line_segs[0][0]
gm.gen_arrow(spos=new_line_segs[0][0], epos=epos, stick_radius=0.004).attach_to(base)

t_cpt = cpt
last_normal = cnrml
direction = rotmat.dot(pt_direction)
tmp_direction = rotmat.dot(tmp_direction)
n=50
for tick in range(1, n+1):
    t_npt = cpt+direction*.07/n
    # mgm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, major_radius=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .005)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # mgm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    homomat = np.eye(4)
    homomat[:3,:3]=plane_rotmat
    homomat[:3,3]=plane_center
    # twod_plane = mgm.gen_box(np.array([.2, .2, .001]), pos=pos, rgba=[.5,.7,1,.3]).attach_to(base)
    projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # mgm.gen_stick(t_npt, projected_point, major_radius=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    # mgm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, major_radius=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(direction)
    tmp_direction = new_rotmat.dot(tmp_direction)
    # new_line_segs = [[cpt, cpt+motion_vec*(.07-tick*.07/n)],
    #                  [cpt+motion_vec*(.07-tick*.07/n), cpt+motion_vec*(.07-tick*.07/n)+tmp_direction*.07]]
    # mgm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # break

t_cpt = cpt
direction = new_rotmat.dot(tmp_direction)
for tick in range(1, n+1):
    t_npt = cpt+direction*.07/n
    # mgm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, major_radius=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .005)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # mgm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    homomat = np.eye(4)
    homomat[:3,:3]=plane_rotmat
    homomat[:3,3]=plane_center
    # if tick == 5:
    #     mgm.gen_box(np.array([.2, .2, .001]), pos=pos, rgba=[.5,.7,1,.1]).attach_to(base)
    projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # mgm.gen_stick(t_npt, projected_point, major_radius=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    # mgm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, major_radius=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    # motion_vec = new_rotmat.dot(motion_vec)
    direction = new_rotmat.dot(tmp_direction)
    # new_line_segs = [[cpt, cpt+motion_vec*(.05-tick*.05/n)],
    #                  [cpt+motion_vec*(.05-tick*.05/n), cpt+motion_vec*(.05-tick*.05/n)+tmp_direction*.05]]
    # mgm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # break
base.run()

t_cpt = cpt
direction = new_rotmat.dot(-pt_direction)
for tick in range(1, n+1):
    t_npt = cpt+direction*.05/n
    # mgm.gen_arrow(spos=cpt, epos=t_npt, major_radius=0.001, rgba=[0,1,1,1]).attach_to(base)
    # mgm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, major_radius=0.001, rgba=[1,1,0,1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .0015)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # mgm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    homomat = np.eye(4)
    homomat[:3,:3]=plane_rotmat
    homomat[:3,3]=plane_center
    # twod_plane = mgm.gen_box(np.array([.2, .2, .001]), pos=pos, rgba=[.5,.7,1,.1]).attach_to(base)
    projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # mgm.gen_stick(t_npt, projected_point, major_radius=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    # mgm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, major_radius=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    # motion_vec = new_rotmat.dot(motion_vec)
    direction = new_rotmat.dot(-pt_direction)
    # new_line_segs = [[cpt, cpt+motion_vec*(.05-tick*.05/n)],
    #                  [cpt+motion_vec*(.05-tick*.05/n), cpt+motion_vec*(.05-tick*.05/n)+tmp_direction*.05]]
    # mgm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # if tick ==2:
    #     break

t_cpt = cpt
direction = new_rotmat.dot(-tmp_direction)
for tick in range(1, n+1):
    t_npt = cpt+direction*.05/n
    # mgm.gen_arrow(spos=t_npt, epos=t_npt+last_normal*.015, major_radius=0.001, rgba=[1, 1, 0, 1]).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .0015)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # mgm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    homomat = np.eye(4)
    homomat[:3,:3]=plane_rotmat
    homomat[:3,3]=plane_center
    # twod_plane = mgm.gen_box(np.array([.2, .2, .001]), pos=pos, rgba=[.5,.7,1,.3]).attach_to(base)
    projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # mgm.gen_stick(t_npt, projected_point, major_radius=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt-projected_point)
    if pn_direction.dot(new_normal) > .1:
        new_normal = -new_normal
    # mgm.gen_arrow(spos=projected_point, epos=projected_point+new_normal*.015, major_radius=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    # motion_vec = new_rotmat.dot(motion_vec)
    direction = new_rotmat.dot(-tmp_direction)
    # new_line_segs = [[cpt, cpt+motion_vec*(.05-tick*.05/n)],
    #                  [cpt+motion_vec*(.05-tick*.05/n), cpt+motion_vec*(.05-tick*.05/n)+tmp_direction*.05]]
    # mgm.gen_linesegs(new_line_segs).attach_to(base)
    gm.gen_stickmodel(spos=cpt, epos=projected_point, rgba=[1, .6, 0, 1], radius=.002, type='round').attach_to(base)
    cpt=projected_point
    last_normal = new_normal
    # break


base.run()
