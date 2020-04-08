from .data_structs import TrajectoryPoint as Tp
from math import ceil, fabs
import numpy as np


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
    time_list = []
    total_size = 0
    for i in range(1, len(array_to_interpolate)):
        size = ceil((array_to_interpolate[i]-array_to_interpolate[i-1]) / step) + 1
        time_list.append(np.linspace(array_to_interpolate[i-1], array_to_interpolate[i], size))
        if i > 1:
            # Remove the duplicate point at the start,
            # since it was included at the end of the previous array.
            time_list[-1] = time_list[-1][1:]
        total_size += size
    return np.concatenate(time_list, axis=0)


def FindMaxJoint(rob_path):
    joint_distances = [np.fabs(rob_path[j, -1] - rob_path[j, 0]) for j in range(len(rob_path))]
    max_dists = np.argmax(joint_distances)
    return max_dists[0] if max_dists.size > 1 else max_dists


def EnsurePathHasOddSize(rob_path):
    if (rob_path[0].size % 2) == 0:
        n = (rob_path[0].size // 2) - 1
        mid_point = (rob_path[:, n] + rob_path[:, n + 1]) / 2.0
        return np.column_stack([rob_path[:, 0:(n + 1)], mid_point, rob_path[:, (n + 1):]])


def SynchronizeJointsToTrajectory(joints_path, trajectory):
    ratio = (joints_path[:, -1] - joints_path[:, 0]) / (trajectory.pos[-1] - trajectory.pos[0])
    rob_trajectory = []
    for j in range(len(joints_path)):
        new_joint_path = (trajectory.pos - trajectory.pos[0]) * ratio[j] + joints_path[j, 0]
        joint_trajectory = Tp(new_joint_path, trajectory.vel * ratio[j], trajectory.acc * ratio[j], trajectory.t)
        rob_trajectory.append(joint_trajectory)
    return rob_trajectory


def AdjustProfileToRequirements(profile):
    # Calculate the profile phases' end points at robot kinematic limits (max parameters).
    profile.CalculateAtMaxParameters()

    # Check if the required distance has been surpassed when moving at max parameters.
    if IsGreater(profile.reached_point.pos, profile.end.pos):
        # Adjust the profile shape to reach end position instead:
        #   In SustainedPulse, this is done by reducing acceleration cruise length.
        #   In Pulse, this is done by reducing pulse peak (i.e. peak acceleration).
        profile.ReduceToReachEndPosition()


def GetFullTrajectory(first_half, flip):
    second_half = Tp(0, 0, 0, 0)
    pos_lift_up = 2 * first_half.pos[-1]
    second_half.t = np.cumsum(np.flip(np.diff(first_half.t))) + first_half.t[-1]
    second_half.pos = -np.flip(first_half.pos[:-1]) + pos_lift_up
    second_half.vel = np.flip(first_half.vel[:-1])
    second_half.acc = -np.flip(first_half.acc[:-1])

    # Concatenate the first and second trajectory halves to get the full trajectory.
    trajectory = Tp(0, 0, 0, 0)
    trajectory.t = np.concatenate([first_half.t, second_half.t], axis=0)
    trajectory.pos = np.concatenate([first_half.pos, second_half.pos], axis=0)
    trajectory.vel = np.concatenate([first_half.vel, second_half.vel], axis=0)
    trajectory.acc = np.concatenate([first_half.acc, second_half.acc], axis=0)

    # re-flip the data if the path was originally decreasing thus was originally flipped.
    if flip:
        trajectory.pos = -trajectory.pos + pos_lift_up
        trajectory.vel = -trajectory.vel
        trajectory.acc = -trajectory.acc

    return trajectory
