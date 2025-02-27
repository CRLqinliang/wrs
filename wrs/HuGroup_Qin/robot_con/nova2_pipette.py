""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240706 Osaka Univ.

"""
import basis.robot_math as rm
from typing import Optional, Literal
import numpy as np
import time
from wrs.HuGroup_Qin.driver.robot_driver.dobot import Dobot
from wrs.HuGroup_Qin.robot_con.pipette_driver import pipette

try:
    import motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

__VERSION__ = (0, 0, 1)


class Nova2Pipette(object):
    @staticmethod
    def pos_arm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in Arm API to the WRS system
        :param arr: Position array obtained from the XArm API
        :return: Converted position array
        """
        return arr / 1000

    @staticmethod
    def pos_wrs2arm(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in WRS system to the Arm API
        :param arr: Position array in the WRS system
        :return: Converted position array
        """
        return arr * 1000

    @staticmethod
    def angle_arm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the angle in Arm API to the WRS system
        :param arr: angle array obtained from the Arm API in degree
        :return: Converted angle array in radian
        """
        return np.deg2rad(arr)

    @staticmethod
    def angle_wrs2arm(arr: np.ndarray) -> np.ndarray or None:
        """
        Convert the angle in WRS system to the Arm API
        :param arr: Position array in the WRS system
        :return: Converted position array
        """
        if arr is None:
            return None
        return np.rad2deg(arr)

    @staticmethod
    def orientation_arm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the orientation (r, p, y) in Arm API to the WRS system
        :param arr: Orientation in the Arm API
        :return: Converted orientation array
        """
        return rm.rotmat_from_euler(*Nova2Pipette.angle_arm2wrs(arr))

    def __init__(self, ip: str = "192.168.5.1", init_enable_rbt=True):

        self._arm_x = Dobot(ip=ip)
        self._arm_x.clear_error()
        if init_enable_rbt and not self._arm_x.is_enable:
            # self._arm_x.power_on()
            self._arm_x.enable_robot()
        print("Robot restart", self._arm_x.robot_mode)
        self.ndof = 6
        self.pipette = pipette(port='COM10', baudrate=115200)

    @property
    def mode(self) -> int:
        """
        Dobot mode
        :return:   If the brake is released, the mode is 2.
                   If the robot is powered on but not enabled, the mode is 4.
                   If the robot is enabled successfully, the mode is 5.
                   If the robot runs, the mode is 7.
                   If the robot pauses, the mode is 10.
                   If the robot enters drag mode (enabled state), the mode is 6.
                   If the robot is dragging and recording, the mode is 8.
                   If the robot is jogging, the mode is 11.
                   Alarm is the top priority. When other modes exist simultaneously, if there is an alarm, the mode is set to 9 first.

        """
        return self._arm_x.robot_mode

    @property
    def terminal_baudrate(self):
        if self._has_gripper:
            return self._arm_x.get_terminal485_baudrate()
        else:
            return

    def clear_error(self):
        self._arm_x.clear_error()

    def set_speed(self, speed: int = 50):
        assert 0 <= speed <= 100
        self._arm_x.set_speed(speed)

    def reset(self):
        self._arm_x.reset_robot()

    def fk(self, joint_values: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Forward _kinematics
        :param joint_values:
        :return: 1. positions (1x3 array) and 2. orientations (3x3 matrix)
        """
        pose = self._arm_x.fk(self.angle_wrs2arm(joint_values))
        pos = self.pos_arm2wrs(pose[:3])
        rot = self.orientation_arm2wrs(pose[3:])
        return pos, rot

    def ik(self, tgt_pos: np.ndarray, tgt_rot: np.ndarray, seed_jnts: np.ndarray = None) -> Optional[np.ndarray]:
        """

        :param tgt_pos: The position under WRS system
        :param tgt_rot: The 3x3 Rotation matrix or 1x3 RPY matrix
        :return: inverse _kinematics solution
        """
        if tgt_rot.shape == (3, 3):
            tgt_rpy = rm.rotmat_to_euler(tgt_rot)
        else:
            tgt_rpy = tgt_rot.flatten()[:3]
        ik_sol = self._arm_x.ik(pos=self.pos_wrs2arm(tgt_pos),
                                rot=self.angle_wrs2arm(tgt_rpy),
                                seed_jnts=self.angle_wrs2arm(seed_jnts))
        if ik_sol is None:
            return
        return self.angle_arm2wrs(ik_sol)

    def get_jnt_values(self) -> np.ndarray:
        """
        Get the joint values of the arm
        :return: Joint values (Array)
        """
        return self.angle_arm2wrs(self._arm_x.get_jnt())

    def get_pose(self) -> (np.ndarray, np.ndarray):
        """
        Get the cartesian position
        :return: tuple(Position(Array), Orientation(Array))
        """
        pose = self._arm_x.get_tcp_cartesian()
        return self.pos_arm2wrs(np.array(pose[:3])), rm.rotmat_from_euler(*self.angle_arm2wrs(pose[3:]))

    def move_j(self, jnt_val: np.ndarray, ):
        """
        Move the robot to a target joint value
        :param jnt_val: Targe joint value (1x6 Array)
        :return: if the path is moved successfully, it will return 0
        """
        assert isinstance(jnt_val, np.ndarray) and len(jnt_val) == self.ndof
        self._arm_x.movej(self.angle_wrs2arm(jnt_val))

    def move_p(self, pos: np.ndarray, rot: np.ndarray, is_linear: bool = True):
        """
        Move to a pose under the robot base coordinate
        :param pos: Position (Array([x,y,z])) of the pose
        :param rot: Orientation (Array([roll,pitch,yaw]) or Array(3x3)) of the pose
        :param is_linear: bool, if True is linear movement
        """
        pos = self.pos_wrs2arm(pos)
        rot = np.array(rot)
        if rot.shape == (3, 3):
            rpy = rm.rotmat_to_euler(rot)
        else:
            rpy = rot.flatten()[:3]
        rpy = self.angle_wrs2arm(rpy)
        if is_linear:
            self._arm_x.movel(pos, rpy)
        else:
            self._arm_x.movep(pos, rpy)

    def move_l(self, pos: np.ndarray, rot: np.ndarray):
        """
        Move the robot to a target pose
        :param pos: Position (Array([x,y,z])) of the pose
        :param rot: Orientation (Array([roll,pitch,yaw]) or Array(3x3)) of the pose
        """
        pos = self.pos_wrs2arm(pos)
        rot = np.array(rot)
        if rot.shape == (3, 3):
            rpy = rm.rotmat_to_euler(rot)
        else:
            rpy = rot.flatten()[:3]
        rpy = self.angle_wrs2arm(rpy)
        self._arm_x.movel(pos, rpy)

    def set_DO_digital_out(self, index: int, val: Literal[0, 1]):
        """
        set digital output for the DO port. value is a {0,1}
        :param index:
        :param val:
        :return:
        """
        self._arm_x.set_digital_out(index, val)

    def move_jntspace_path(self, path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           control_frequency=1 / 33,
                           toggle_debug=False):
        if TOPPRA_EXIST:
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
                                                                toggle_debug=False)
            interpolated_path = interpolated_path[start_frame_id:]
            for jnt_values in interpolated_path:
                self._arm_x.servoj(self.angle_wrs2arm(jnt_values))
                time.sleep(.03)
            return
        else:
            raise NotImplementedError

    def __del__(self):
        self._arm_x.close()
        del self.pipette
        print("The robot is closed")


if __name__ == '__main__':
    pass
