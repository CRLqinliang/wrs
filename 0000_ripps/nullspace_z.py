import os
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta_ripps as cbt
import visualization.panda.world as wd
import modeling.geometric_model as gm
import robot_con.cobotta.cobotta_x as cbtx


def genSphere(pos, radius=0.02, rgba=None):
    if rgba is None:
        rgba = [1, 0, 0, 1]
    gm.gen_sphere(pos=pos, radius=radius, rgba=rgba).attach_to(base)


if __name__ == '__main__':

    base = wd.World(cam_pos=[1.2, 1.2, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    # robot_s
    component_name = 'arm'
    robot_s = cbt.CobottaRIPPS()
    tgt_pos = np.array([0.25, .0, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2).dot(rm.rotmat_from_axangle([0,0,1], -math.pi/2))
    jnt_values = robot_s.ik(component_name=component_name, tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_values is None:
        gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        base.run()
    robot_s.fk(component_name=component_name,
               jnt_values=jnt_values)
    # robot_s.gen_meshmodel().attach_to(base)
    # base.run()

    gl_tcp = robot_s.get_gl_tcp(manipulator_name="arm")
    # null space planning
    path = []
    ratio = .001
    for t in range(0, 5000, 1):
        print("-------- timestep = ", t, " --------")
        xa_jacob = robot_s.jacobian()
        # xa_ns = rm.null_space(xa_jacob)
        # cur_jnt_values = robot_s.get_jnt_values()
        # cur_jnt_values += np.ravel(xa_ns)*.01
        # nullspace rotate
        xa_ns = rm.null_space(xa_jacob[[0,1,2,3,4], :])
        cur_jnt_values = robot_s.get_jnt_values(component_name=component_name)
        cur_jnt_values -= np.ravel(xa_ns[:, 0]) * ratio
        # gm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
        print(xa_ns)
        print(gl_tcp[1][:3,2])
        # cur_jnt_values += xa_ns.dot(gl_tcp[1][:3,2])*ratio
        # gl_tcp[1][:3, 2].dot(xa_jacob[:3, :])
        # cur_jnt_values += gl_tcp[1][:3,2].dot(xa_jacob[:3, :])*ratio
        status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        # if status == "succ":
        if t % 200 == 0:
            path.append(cur_jnt_values)
            robot_s.gen_meshmodel(rgba=[0, 1, 1, .1]).attach_to(base)
        # elif status == "out_of_rng":
        #     ratio=-ratio
    path = path[::-1]
    robot_s.fk(component_name=component_name,
               jnt_values=jnt_values)
    ratio = -ratio
    for t in range(0, 5000, 1):
        print("-------- timestep = ", t, " --------")
        xa_jacob = robot_s.jacobian()
        # xa_ns = rm.null_space(xa_jacob)
        # cur_jnt_values = robot_s.get_jnt_values()
        # cur_jnt_values += np.ravel(xa_ns)*.01
        # nullspace rotate
        xa_ns = rm.null_space(xa_jacob[[0,1,2,3,4], :])
        cur_jnt_values = robot_s.get_jnt_values(component_name=component_name)
        cur_jnt_values -= np.ravel(xa_ns[:, 0]) * ratio
        # gm.gen_frame(pos=gl_tcp[0], rotmat=gl_tcp[1]).attach_to(base)
        print(xa_ns)
        print(gl_tcp[1][:3,2])
        # cur_jnt_values += xa_ns.dot(gl_tcp[1][:3,2])*ratio
        # gl_tcp[1][:3, 2].dot(xa_jacob[:3, :])
        # cur_jnt_values += gl_tcp[1][:3,2].dot(xa_jacob[:3, :])*ratio
        status = robot_s.fk(component_name=component_name, jnt_values=cur_jnt_values)
        # if status == "succ":
        if t % 200 == 0:
            path.append(cur_jnt_values)
            robot_s.gen_meshmodel(rgba=[0, 1, 1, .1]).attach_to(base)

    # robot_x = cbtx.CobottaX()
    # robot_x.move_jnts_motion(path)

    # uncomment the following part for animation
    def update(rbtmnp, motioncounter, robot, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detach()
            pose = path[motioncounter[0]]
            robot.fk(armname, pose)
            rbtmnp[0] = robot.gen_meshmodel(toggle_tcpcs=True)
            rbtmnp[0].attach_to(base)
            # genSphere(robot.get_gl_tcp(component_name)[0], radius=0.01, rgba=[1, 1, 0, 1])
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again
    rbtmnp = [None]
    motioncounter = [0]
    taskMgr.doMethodLater(0.1, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot_s, path, component_name], appendTask=True)
    base.setFrameRateMeter(True)
    base.run()