import numpy as np

from .data_structs import RobotKinematics
from .joint_profile import ConstructJointProfile
from .trajectory_utils import EnsurePathHasOddSize, FindMaxJoint, SynchronizeJointsToTrajectory, ValidateRobotPath


class Robot:
    """ define a robot with certain kinematic limits to be used when constructing a trajectory profile"""
    def __init__(self, n_joints, j_max, a_max, v_max):
        """
        initialize robot kinematic parameters with user specified parameters
        Note::  If the robot joints have different kinematic limits, It is recommended to use the lowest values here
                to ensure safety and correct performance.
        :param n_joints: defining robot's number of joints
        :type n_joints: int
        :param j_max: defining joint maximum jerk to use in trajectory (rad/sec^3)
        :type j_max: float
        :param a_max: defining joint maximum acceleration to use in trajectory (rad/sec^2)
        :type a_max: float
        :param v_max: defining joint maximum velocity to use in trajectory (rad/sec)
        :type v_max: float
        """
        if n_joints <= 0:
            raise ValueError("Robot number of joints should be greater than zero")
        if j_max <= 0:
            raise ValueError("Robot jerk limit should be greater than zero")
        if a_max <= 0:
            raise ValueError("Robot acceleration limit should be greater than zero")
        if v_max <= 0:
            raise ValueError("Robot velocity limit should be greater than zero")

        self.rob_k = RobotKinematics(n_joints, j_max, a_max, v_max)

    def TimeParameterizePath(self, robot_path, interp_time_step=None):
        """Construct the trajectory of the robot from a predefined path.
        :param robot_path: Union[ndarray, List, Tuple] -> rows=number of joints, columns= number of path points.
        :param interp_time_step: if not None, interpolation process will be added using this time step.
        :return rob_trajectory: (list) each list entry is a (TrajectoryPoint) containing a joint trajectory.
        """
        # Check path dimensions, types, and number of joints.
        ValidateRobotPath(robot_path, self.rob_k.j_num)

        # Copy path to a numpy nd-array object.
        rob_path = np.copy(robot_path).reshape(self.rob_k.j_num, -1)

        # If path points are even, Add a point in the middle to make it Odd.
        rob_path = EnsurePathHasOddSize(rob_path)

        # Find Max Joint, which is the joint that moves the greatest distance.
        max_j_num = FindMaxJoint(rob_path)

        # Construct Max Joint Profile.
        max_trajectory = ConstructJointProfile(self.rob_k, rob_path[max_j_num], interp_time_step)

        # Construct Other Joints' Profiles from the Max Joint Profile, to have them Synchronized.
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
