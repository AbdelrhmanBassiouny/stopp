import numpy as np

from .data_structs import RobotKinematics
from .joint_profile import ConstructJointProfile
from .trajectory_utils import EnsurePathHasOddSize, FindMaxJoint, SynchronizeJointsToTrajectory, ValidateRobotPath, ConcatTraj
from copy import deepcopy


class Robot:
    """ define a robot with certain kinematic limits to be used when constructing a trajectory profile"""
    def __init__(self, n_joints, j_max, a_max, v_max, d_max=None):
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
        if a_max <= 0 or d_max <=0:
            raise ValueError("Robot acceleration limit should be greater than zero")
        if v_max <= 0:
            raise ValueError("Robot velocity limit should be greater than zero")

        self.rob_k = RobotKinematics(n_joints, j_max, a_max, v_max, d_max)

    def TimeParameterizePath(self, robot_path, vi=0, vf=0, interp_time_step=None, velocity_mode=False, robot_kinematics=None):
        """Construct the trajectory of the robot from a predefined path.
        :param robot_path: Union[ndarray, List, Tuple] -> rows=number of joints, columns= number of path points.
        :param interp_time_step: if not None, interpolation process will be added using this time step.
        :return rob_trajectory: (list) each list entry is a (TrajectoryPoint) containing a joint trajectory.
        """
        self.rob_k = robot_kinematics if robot_kinematics is not None else self.rob_k
        if self.rob_k.d_max < self.rob_k.a_max and not velocity_mode:
            path = robot_path.tolist()
            path_1 = np.array([j_path[:len(j_path)//2+1] for j_path in path])
            path_2 = np.array([j_path[len(j_path)//2:] for j_path in path])
            robk_2 = deepcopy(self.rob_k)
            robk_2.a_max = self.rob_k.d_max
            originial_rob_k = deepcopy(self.rob_k)
            traj_2 = self.TimeParameterizePath(
                path_2, vi=self.rob_k.v_max, vf=0, interp_time_step=interp_time_step, velocity_mode=True, robot_kinematics=robk_2)
            self.rob_k = originial_rob_k
            traj_1 = self.TimeParameterizePath(
                path_1, vi=0, vf=traj_2[0].vel[0], interp_time_step=interp_time_step, velocity_mode=True)
            traj = ConcatTraj([traj_1[0], traj_2[0]])
            return [traj]
        # Check path dimensions, types, and number of joints.
        ValidateRobotPath(robot_path, self.rob_k.j_num)
        accelerate = True
        shift_vel = False
        init_vel = 0
        v_diff = abs(vf - vi)
        if vi < 0 or vf < 0:
            shift_vel = True
            if vi < vf:
                vi = 0
                vf = v_diff
                accelerate = True
                robot_path = np.flip(robot_path)
        
        if vi > vf:
            vi = v_diff
            vf = 0
            accelerate = False
        
            
        
        if velocity_mode:
            if vi <= vf:
                self.rob_k.v_max = vf
            else:
                self.rob_k.v_max = vi
                vi = vf
                # accelerate = False
        
        # Copy path to a numpy nd-array object.
        rob_path = np.copy(robot_path)

        if not velocity_mode:
            # If path points are even, Add a point in the middle to make it Odd.
            rob_path = EnsurePathHasOddSize(rob_path)

        # Find Max Joint, which is the joint that moves the greatest distance.
        max_j_num = FindMaxJoint(rob_path)
        # print(max_j_num)

        # Construct Max Joint Profile.
        max_trajectory = ConstructJointProfile(self.rob_k, rob_path[max_j_num],
                                               vi=vi, vf=vf, time_step=interp_time_step,
                                               velocity_mode=velocity_mode, accelerate=accelerate)
        if shift_vel:
            max_trajectory.vel += init_vel
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
    trajectory = my_rob.TimeParameterizePath(path, vi=1, vf=0, interp_time_step=time_step)

    for j in range(joints):
        print("joint {} trajectory points time = {}".format(j, trajectory[j].t))
        print("joint {} trajectory points position = {}".format(j, trajectory[j].pos))
        print("joint {} trajectory points velocity = {}".format(j, trajectory[j].vel))
        print("joint {} trajectory points acceleration = {}".format(j, trajectory[j].acc))
        print("============================================================")
