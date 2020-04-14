from math import ceil, fabs

import numpy as np

from .data_structs import TrajectoryPoint as Tp


def Assert(condition, message):
    try:
        assert condition
    except AssertionError:
        print(message)
        raise


def IsClose(a, b):
    return fabs(a - b) <= 0.0001


def IsGreater(a, b):
    return a-0.0001 >= b


def IsSmaller(a, b):
    return a+0.0001 <= b


def SolveForRealRoots(coefficients, assert_condition=True):
    roots = np.roots(coefficients)
    x_roots = roots[np.isclose(roots.imag, 0.0, 0.001, 0.0001)].real
    indices = np.array(np.where(x_roots >= 0))
    if assert_condition:
        Assert(indices.size > 0, "No real positive solution found")
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
    if not hasattr(robot_path, '__getitem__'):
        raise TypeError("Robot Path should be a 2 dimensional sequence [List or Tuple or ndarray]")
    if not hasattr(robot_path[0], '__getitem__') and n_joints > 1:
        raise TypeError("Robot Path should be a 2 dimensional array or a nested sequence"
                        " with shape=(joints_number, points_number)")
    if len(robot_path) != n_joints and n_joints > 1:
        raise ValueError("Robot Path should have its first dimension equal to number of joints")

    n_points = len(robot_path[0]) if n_joints > 1 else len(robot_path)
    if n_points < 2:
        raise ValueError("Need at least 2 path points to construct a trajectory")
    if n_joints > 1:
        for j in range(n_joints):
            if len(robot_path[j]) != n_points:
                raise ValueError("All joints should have the same number of points")


def FindMaxJoint(rob_path):
    joint_distances = [np.fabs(rob_path[j, -1] - rob_path[j, 0]) for j in range(len(rob_path))]
    max_dists = np.argmax(joint_distances)
    return max_dists[0] if max_dists.size > 1 else max_dists


def EnsurePathHasOddSize(rob_path):
    if (rob_path[0].size % 2) == 0:
        n = (rob_path[0].size // 2) - 1
        mid_point = (rob_path[:, n] + rob_path[:, n + 1]) / 2.0
        return np.column_stack([rob_path[:, 0:(n + 1)], mid_point, rob_path[:, (n + 1):]])
    else:
        return rob_path


def SynchronizeJointsToTrajectory(joints_path, trajectory):
    ratio = (joints_path[:, -1] - joints_path[:, 0]) / (trajectory.pos[-1] - trajectory.pos[0])
    rob_trajectory = []
    for j in range(len(joints_path)):
        new_joint_path = (trajectory.pos - trajectory.pos[0]) * ratio[j] + joints_path[j, 0]
        joint_trajectory = Tp(new_joint_path, trajectory.vel * ratio[j], trajectory.acc * ratio[j], trajectory.t)
        rob_trajectory.append(joint_trajectory)
    return rob_trajectory
