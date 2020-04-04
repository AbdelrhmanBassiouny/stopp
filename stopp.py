import numpy as np
from .data_structs import RobotKinematics
from .trajectory_utils import EnsurePathHasOddSize, FindMaxJoint, SynchronizeJointsToTrajectory
from .joint_profile import ConstructJointProfile


class Robot:
    """ define a robot with certain kinematic limits to be used when constructing a trajectory profile"""
    def __init__(self, n_joints, j_max, a_max, v_max):
        """
        initialize robot kinematic parameters with user specified parameters
        Note::  If the robot joints have different kinematic limits, It is recommended to use the lowest values here
                to ensure safety and correct performance.
        :param n_joints: int -> defining robot's number of joints
        :param j_max:  float -> defining joint maximum jerk to use in trajectory (rad/sec^3)
        :param a_max: float -> defining joint maximum acceleration to use in trajectory (rad/sec^2)
        :param v_max: float -> defining joint maximum velocity to use in trajectory (rad/sec)
        """
        self.rob_k = RobotKinematics(n_joints, j_max, a_max, v_max)

    def TimeParameterizePath(self, rob_path, interp_time_step=None):
        """Construct the trajectory of the robot from a predefined path
        :param rob_path: numpy nd-array -> rows=number of joints, columns= number of path points.
        :param interp_time_step: if not None, interpolation process will be added using this time step.
        :return rob_trajectory: (list) each list entry is a (TrajectoryPoint) describing a joint trajectory.
        """
        rob_path = np.copy(rob_path)

        EnsurePathHasOddSize(rob_path)

        max_j_num = FindMaxJoint(rob_path)

        max_trajectory = ConstructJointProfile(self.rob_k, rob_path[max_j_num], interp_time_step)

        rob_trajectory = SynchronizeJointsToTrajectory(rob_path, max_trajectory)

        return rob_trajectory


if __name__ == "__main__":
    """This is an Example usage of the library, Enjoy!"""
    rob_j_max = 800.0
    rob_a_max = 50
    rob_v_max = 6
    joints = 2
    n_points = 31
    time_step = 0.004
    path = np.array([np.linspace(0, 50*(j+1), n_points) for j in range(joints)]) * (np.pi / 180)

    my_rob = Robot(joints, rob_j_max, rob_a_max, rob_v_max)
    trajectory = my_rob.TimeParameterizePath(path, time_step)

    for j in range(joints):
        print("joint {} trajectory points time = {}".format(j, trajectory[j].t))
        print("joint {} trajectory points position = {}".format(j, trajectory[j].pos))
        print("joint {} trajectory points velocity = {}".format(j, trajectory[j].vel))
        print("joint {} trajectory points acceleration = {}".format(j, trajectory[j].acc))
        print("============================================================")
