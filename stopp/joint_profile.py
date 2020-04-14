import numpy as np

from .data_structs import TrajectoryPoint as Tp
from .trajectory_profiles import SustainedPulse, Pulse
from .trajectory_utils import IsGreater


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


def ConstructJointProfile(robot_kinematics, joint_path, time_step=None):
    """Construct hte trajectory of a joint given its path..
        :param robot_kinematics: a RobotKinematics object for which the trajectory profile is constructed
        :param joint_path: (numpy array) path points container
        :param time_step: (optional)(float) interpolation time step
        :return The joint trajectory.
    """
    rob_k = robot_kinematics
    path = np.copy(joint_path)
    # The equations work for increasing position values, thus flip the path if otherwise.
    flip = False
    if path[1] < path[0]:
        path = np.flip(path)
        flip = True

    # make use of trajectory symmetry, thus only need to compute half the trajectory.
    mid_idx = len(path) // 2

    # Initialize the profile as a Sustained Acceleration Pulse (SAP) profile.
    profile = SustainedPulse(Tp(path[0], 0, 0, 0), Tp(path[mid_idx], 0, 0, 0), rob_k)

    # Adjust profile shape to accommodate for robot limits, the start point, and goal point requirements.
    AdjustProfileToRequirements(profile)

    # Make sure these requirements are met after the adjustments.
    if IsGreater(profile.reached_point.vel, rob_k.v_max) \
            or IsGreater(profile.reached_point.pos, profile.end.pos):
        # Change profile shape from SustainedPulse to Pulse if it cannot accommodate for the requirements.
        profile = Pulse(profile.start, profile.end, rob_k)
        # Adjust the pulse profile to the requirements.
        AdjustProfileToRequirements(profile)

    # Sample from the constructed profile by the path positions, and an interpolation time step if given.
    first_half = profile.SampleByPosition(path[:mid_idx + 1], time_step)

    # Construct and Concatenate the second trajectory half using symmetry to get the full trajectory.
    # also, if path was flipped in the beginning then flip the trajectory.
    trajectory = GetFullTrajectory(first_half, flip)

    # Return a TrajectoryPoint object containing the trajectory data.
    return trajectory
