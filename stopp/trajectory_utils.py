from math import ceil, fabs

import numpy as np

from .data_structs import TrajectoryPoint as Tp


class ShapeError(Exception):
    pass


class LogicError(Exception):
    pass


class NoSolutionError(Exception):
    pass


def IsClose(a, b):
    return fabs(a - b) <= 0.0001


def IsGreater(a, b):
    return a-0.0001 >= b


def IsSmaller(a, b):
    return a+0.0001 <= b

def ConcatTraj(traj_list):
    trajectory = traj_list[0]
    for i in range(1, len(traj_list)):
        second_half = traj_list[i]
        trajectory.t = np.concatenate(
            [trajectory.t, second_half.t+trajectory.t[-1]], axis=0)
        trajectory.pos = np.concatenate(
            [trajectory.pos, second_half.pos], axis=0)
        trajectory.vel = np.concatenate(
            [trajectory.vel, second_half.vel], axis=0)
        trajectory.acc = np.concatenate(
            [trajectory.acc, second_half.acc], axis=0)
    return trajectory


def SolveForPositiveRealRoots(coefficients):
    roots = np.roots(coefficients)
    x_roots = roots[np.isclose(roots.imag, 0.0, 0.001, 0.0001)].real
    indices = np.array(np.where(x_roots >= 0))
    if indices.size < 1:
        raise NoSolutionError("No real positive solution found")
    x = np.min(x_roots[indices]) if indices.size > 0 else -1
    return x


def Interpolate(array_to_interpolate, step):
    if step <= 0:
        raise ValueError("Interpolation step should be greater than zero")
    time_list = []
    for i in range(1, len(array_to_interpolate)):
        size = ceil((array_to_interpolate[i]-array_to_interpolate[i-1]) / step) + 1
        time_list.append(np.linspace(array_to_interpolate[i-1], array_to_interpolate[i], size))
        if i > 1:
            # Remove the duplicate point at the start,
            # since it was included at the end of the previous array.
            time_list[-1] = time_list[-1][1:]
    return np.concatenate(time_list, axis=0)


def ValidateRobotPath(robot_path, n_joints):
    if not isinstance(robot_path, np.ndarray):
        raise TypeError("Expected robot_path to have {}, but got {} instead".format(np.ndarray, type(robot_path)))
    if not len(robot_path.shape) == 2:
        raise ShapeError("Expected robot_path to have 2 dimensions, but got {} instead"
                         .format(len(robot_path.shape)))
    if not robot_path.shape[0] == n_joints:
        raise ShapeError("Expected robot_path to have its first_dimension_size({}) equal to number_of_joints({})"
                         .format(robot_path.shape[0], n_joints))
    if not robot_path.shape[1] >= 2:
        raise LogicError("Expected robot_path to have at least 2 points, got {} instead".format(robot_path.shape[1]))


def EnsurePathHasOddSize(rob_path):
    if (rob_path[0].size % 2) == 0:
        n = (rob_path[0].size // 2) - 1
        mid_point = (rob_path[:, n] + rob_path[:, n + 1]) / 2.0
        rob_path = np.column_stack([rob_path[:, 0:(n + 1)], mid_point, rob_path[:, (n + 1):]])
    return rob_path


def FindMaxJoint(rob_path):
    joint_distances = [np.fabs(rob_path[j, -1] - rob_path[j, 0]) for j in range(len(rob_path))]
    max_dists = np.argmax(joint_distances)
    return max_dists[0] if max_dists.size > 1 else max_dists


def SynchronizeJointsToTrajectory(joints_path, trajectory):
    ratio = (joints_path[:, -1] - joints_path[:, 0]) / (trajectory.pos[-1] - trajectory.pos[0])
    rob_trajectory = []
    for j in range(len(joints_path)):
        new_joint_path = (trajectory.pos - trajectory.pos[0]) * ratio[j] + joints_path[j, 0]
        joint_trajectory = Tp(new_joint_path, trajectory.vel * ratio[j], trajectory.acc * ratio[j], trajectory.t)
        rob_trajectory.append(joint_trajectory)
    return rob_trajectory
