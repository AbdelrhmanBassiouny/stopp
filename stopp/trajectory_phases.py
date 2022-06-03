from math import pi

import numpy as np

from .data_structs import TrajectoryPoint as Tp
from .trajectory_utils import SolveForPositiveRealRoots, Interpolate, IsClose


class Quintic:
    def __init__(self, robot_kinematics):
        """ Quintic Trajectory Equations:
            pos = a*(t**5) + b*(t**4) + c*(t**3) + d*(t**2) + e*t + f
            vel = 5*a*(t**4) + 4*b*(t**3) + 3*c*(t**2) + 2*d*t + e
            vel = 20*a*(t**3) + 12*b*(t**2) + 6*c*t + 2*d
        """
        self.start = Tp(0, 0, 0, 0)
        self.end = Tp(0, 0, 0, 0)
        self.j_max = robot_kinematics.j_max
        self.J = self.j_max * 2 / np.pi  # self.J = 2 * self.j_max / 3 is possible for no sine template
        self.a_max = robot_kinematics.a_max
        self.v_max = robot_kinematics.v_max
        self.coefficients_matrix = np.array([])

    def CalculateQuinticCoefficients(self):

        t0 = 0.0
        i0 = [t0 ** i for i in reversed(range(6))]
        i1 = [i * t0 ** max(i - 1, 0) for i in reversed(range(6))]
        i2 = [i * (i - 1) * t0 ** max(i - 2, 0) for i in reversed(range(6))]

        tf = self.end.t
        i3 = [tf ** i for i in reversed(range(6))]
        i4 = [i * tf ** max(i - 1, 0) for i in reversed(range(6))]
        i5 = [i * (i - 1) * tf ** max(i - 2, 0) for i in reversed(range(6))]

        a = np.array([i0, i1, i2, i3, i4, i5])
        b = np.array([self.start.pos, self.start.vel, self.start.acc,
                      self.end.pos, self.end.vel, self.end.acc])

        x = np.linalg.solve(a, b)
        self.coefficients_matrix = np.array([[x[0],  x[1],    x[2],    x[3],   x[4],   x[5]],
                                             [0.0, 5*x[0],  4*x[1],  3*x[2], 2*x[3],   x[4]],
                                             [0.0,    0.0, 20*x[0], 12*x[1], 6*x[2], 2*x[3]]])

    def SampleByTime(self, times_array):
        time_matrix = np.array([times_array**i for i in reversed(range(6))])
        return np.dot(self.coefficients_matrix, time_matrix)

    def SampleByPosition(self, positions, time_step=None):
        """Calculate the accelerations, velocities, and times for these positions.

            This should be Used only after the phase Control Point has been calculated.
            :param positions: 1d-list/array of positions
            :param time_step: (float) if not None, then an interpolation step will be added using this time step.
            :return:(numpy nd-array) shape=(4, number of positions)
                    1st row = times, 2nd row = positions, 3rd row = velocities, 4th row = accelerations
        """

        pos_coefficients = np.copy(self.coefficients_matrix[0])
        # print(positions)
        # print(self.__class__)
        # print("start_vel = ", self.start.vel)
        # print("end_vel = ", self.end.vel)
        # print("start_pos = ", self.start.pos)
        # print("end_pos = ", self.end.pos)
        # print("start_t = ", self.start.t)
        # print("end_t = ", self.end.t)
        path_times = []
        # Add the phase starting time if time_step is specified (i.e. if interpolation is needed)
        if (time_step is not None) and not (isinstance(self, RampPhase) and self.J > 0):
            path_times.append(0.0)

        # Solving for points times from points positions
        for i in range(len(positions)):
            pos_coefficients[-1] -= positions[i]
            path_times.append(SolveForPositiveRealRoots(pos_coefficients))
            pos_coefficients[-1] += positions[i]

        if time_step is not None:
            # Add the Phase's end point.
            if len(path_times) < 1 or (len(path_times) >= 1 and not IsClose(path_times[-1], self.end.t)):
                path_times.append(self.end.t)

            if len(path_times) == 1:
                time_array = np.array(path_times)
            else:
                # Interpolating
                time_array = Interpolate(path_times, time_step)
            if not (isinstance(self, RampPhase) and self.J > 0) and (time_array.shape[0] > 1):
                time_array = time_array[1:]
        else:
            time_array = np.array(path_times)

        data = np.empty((4, time_array.shape[0]))
        data[0, :] = time_array

        # Sampling
        data[1:, :] = self.SampleByTime(data[0])

        return data


class RampPhase(Quintic):
    def __init__(self, robot_kinematics, ramp_down=False):
        Quintic.__init__(self, robot_kinematics)
        if ramp_down:
            self.J = -1 * self.J
            self.j_max = -1 * self.j_max

    def Calculate(self, end_acceleration=0.0):
        self.end.acc = end_acceleration
        self.end.t = max((self.end.acc - self.start.acc) / self.J, 0.0)
        self.end.vel = 0.5 * (self.start.acc + self.end.acc) * self.end.t + self.start.vel
        k = (0.25 - 1 / (pi ** 2))  # k = 3/20 is possible for no sine template.
        self.end.pos = k * self.J * (self.end.t**3) \
            + 0.5 * self.start.acc * (self.end.t**2) \
            + self.start.vel * self.end.t + self.start.pos


class CruisePhase(Quintic):

    def Calculate(self, cruise_end_vel):
        self.end.acc = self.start.acc
        self.end.vel = cruise_end_vel
        self.end.t = (self.end.vel - self.start.vel) / self.start.acc
        self.end.pos = 0.5 * self.start.acc * (self.end.t**2) + self.start.vel * self.end.t + self.start.pos

    def CalculateFromPeriod(self, cruising_period):
        self.end.acc = self.start.acc
        self.end.t = cruising_period
        self.end.vel = self.start.acc * self.end.t + self.start.vel
        self.end.pos = 0.5 * self.start.acc * (self.end.t**2) + self.start.vel * self.end.t + self.start.pos


class DwellPhase(Quintic):

    def Calculate(self, end_position):
        self.end.vel = self.start.vel
        self.end.acc = self.start.acc
        if IsClose(self.start.vel, 0.0):
            self.end.t  = self.start.t
            self.end.pos = self.start.pos
        else:
            self.end.t = max((end_position - self.start.pos) / self.start.vel, 0)
            self.end.pos = self.end.t * self.start.vel + self.start.pos
