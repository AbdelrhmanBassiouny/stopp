from bisect import bisect_right
from math import pi

import numpy as np

from .data_structs import TrajectoryPoint as Tp
from .trajectory_phases import RampPhase, CruisePhase, DwellPhase
from .trajectory_utils import IsGreater, SolveForPositiveRealRoots, NoSolutionError


class Profile:
    def __init__(self, initial_profile_point, final_profile_point, robot_kinematics):
        """Initialize the common phases in any profile shape, and the start and end conditions.
            This acts as a Base/Parent class for two shapes Acceleration profile shapes (Pulse, SustainedPulse)
        """
        # Initialize robot kinematic limits to be used in profile calculations.
        self.j_max = robot_kinematics.j_max
        self.J = self.j_max * 2 / np.pi  # self.J = 2 * self.j_max / 3 is possible for no sine template
        self.a_max = robot_kinematics.a_max
        self.v_max = robot_kinematics.v_max

        # Initialize Profile Start and End points.
        self._start = Tp.FromTrajectoryPoint(initial_profile_point)
        self._end = Tp.FromTrajectoryPoint(final_profile_point)

        # Initialize Common Profile Phases.
        self._ramp_up = RampPhase(robot_kinematics)
        self._ramp_down = RampPhase(robot_kinematics, ramp_down=True)
        self._dwell = DwellPhase(robot_kinematics)
        self._phase_list = []

        # Phase Linking
        self._ramp_up.start = self._start
        self._dwell.start = self._ramp_down.end
        self.reached_point = self._dwell.end

    def UpdatePhaseSequence(self):
        final_phase_list = []
        for phase in range(len(self._phase_list)):
            if IsGreater(self._phase_list[phase].end.t, 0.0):
                final_phase_list.append(self._phase_list[phase])
        self._phase_list = final_phase_list

    def CalculateQuinticCoefficients(self):
        for phase in self._phase_list:
            phase.CalculateQuinticCoefficients()

    def SampleByPosition(self, positions, time_step=None):
        prev_idx = 0
        data_list = []
        accumulated_time = 0.0
        for i in range(len(self._phase_list)):
            # print(positions[prev_idx])
            # print(positions[prev_idx:])
            # print(self._phase_list[i].end.pos)
            curr_idx = bisect_right(positions, self._phase_list[i].end.pos, lo=prev_idx)

            phase_positions = positions[prev_idx:curr_idx]

            if len(phase_positions) > 0 or time_step is not None:
                data_list.append(self._phase_list[i].SampleByPosition(phase_positions, time_step=time_step))
                data_list[-1][0] += accumulated_time

            accumulated_time += self._phase_list[i].end.t
            prev_idx = curr_idx

        if len(data_list) > 0:
            data_array = np.column_stack(data_list)
        else:
            data_array = np.array([[self.start.t  , self.end.t  ],
                                   [self.start.pos, self.end.pos],
                                   [self.start.vel, self.end.vel],
                                   [self.start.acc, self.end.acc]])

        return Tp(data_array[1], data_array[2], data_array[3], data_array[0])

    # Its mandatory that we copy values here and not re-initialize to not mess up the linked phase objects
    # that's why i made sure any read or write to req_start or req_end will be a copy to or from operations.

    def _Set_Start(self, trajectory_point): self._start.CopyFrom(trajectory_point)
    def _Get_Start(self): return Tp.FromTrajectoryPoint(self._start)
    def _Set_End(self, trajectory_point): self._end.CopyFrom(trajectory_point)
    def _Get_End(self): return Tp.FromTrajectoryPoint(self._end)

    start = property(_Get_Start, _Set_Start)
    end = property(_Get_End, _Set_End)


class SustainedPulse(Profile):
    """An acceleration profile composed of 4 quintic trajectories/phases linked to each other.

        Acceleration RampUp -> Acceleration Cruise -> Acceleration RampDown -> Acceleration Dwell
    """
    def __init__(self, initial_profile_point, final_profile_point, robot_kinematics):
        Profile.__init__(self, initial_profile_point, final_profile_point, robot_kinematics)
        # Initialize Cruise Phase.
        self._cruise = CruisePhase(robot_kinematics)

        # Phase Linking
        self._cruise.start = self._ramp_up.end
        self._ramp_down.start = self._cruise.end

        # Phase Sequence
        self._phase_list.extend([self._ramp_up, self._cruise, self._ramp_down, self._dwell])

    def CalculateAtMaxParameters(self):
        self._ramp_up.Calculate(self.a_max)

        # Adjust cruise to reach v_max at end of ramp_down.
        cruise_end_vel = max(self.v_max - 2 * self._ramp_up.end.vel, 0.0) + self._ramp_up.end.vel

        self._cruise.Calculate(cruise_end_vel)
        self._ramp_down.Calculate()
        self._dwell.Calculate(self.end.pos)
        self.UpdatePhaseSequence()

    def ReduceToReachEndPosition(self):
        self._ramp_up.Calculate(self.a_max)

        # Reduce cruise period to reach end_position at end of ramp_down instead of reaching v_max.
        k = (0.25 - 1 / (pi ** 2))  # k = 3/20 is another possible value (no sine template).
        a = 0.5 * self._cruise.start.acc
        b = self._cruise.start.vel + self._cruise.start.acc * self._cruise.start.t
        c = self._cruise.start.pos - self.end.pos + self._cruise.start.vel * self._cruise.start.t\
            - k * self.J * (self._cruise.start.t**3) \
            + 0.5 * self._cruise.start.acc * (self._cruise.start.t**2)
        try:
            cruising_period = SolveForPositiveRealRoots([a, b, c])
        except NoSolutionError:
            # Means there's no positive real roots found, thus no cruising period required.
            cruising_period = 0.0

        self._cruise.CalculateFromPeriod(cruising_period)
        self._ramp_down.Calculate()
        self._dwell.Calculate(self.end.pos)
        self.UpdatePhaseSequence()


class Pulse(Profile):
    """An acceleration profile composed of 3 quintic trajectories/phases linked to each other.

                Acceleration RampUp -> Acceleration RampDown -> Acceleration Dwell

        This is considered a special case of Sustained pulse where Cruise Phase is removed.
        And the adjustments are done to the Ramp Phases instead of always ramp-ing to a_max
        as in sustained acceleration pulse where adjustments are only on cruising period.
    """
    def __init__(self, initial_profile_point, final_profile_point, robot_kinematics):
        Profile.__init__(self, initial_profile_point, final_profile_point, robot_kinematics)

        # Phase Linking
        self._ramp_down.start = self._ramp_up.end

        # Phase Sequence
        self._phase_list.extend([self._ramp_up, self._ramp_down, self._dwell])

    def CalculateAtMaxParameters(self):
        # Calculate peak acceleration that will ensure final_velocity = v_max(robot_max_velocity)
        acc_peak = np.sqrt(0.5 * (2 * self.J * (self.v_max - self.start.vel) + (self.start.acc ** 2)))
        # If acc_peak to reach v_max is > a_max, then we cannot go to v_max, Thus use a_max as acc_peak.
        acc_peak = min(acc_peak, self.a_max)

        self._ramp_up.Calculate(acc_peak)
        self._ramp_down.Calculate()
        self._dwell.Calculate(self.end.pos)
        self.UpdatePhaseSequence()

    def ReduceToReachEndPosition(self):
        # Adjust Acceleration peak to reach end_position at end of ramp down.
        coefficients = [self.J, 2*self.start.acc, 2*self.start.vel, self.start.pos-self.end.pos]
        acc_peak = SolveForPositiveRealRoots(coefficients) * self.J + self._start.acc

        self._ramp_up.Calculate(acc_peak)
        self._ramp_down.Calculate()
        self._dwell.Calculate(self.end.pos)
        self.UpdatePhaseSequence()
