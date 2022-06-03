class JointState:
    def __init__(self, position, velocity, acceleration):
        """initialize with user specified parameters"""
        self.pos = position
        self.vel = velocity
        self.acc = acceleration

    @classmethod
    def FromJointState(cls, joint_state):
        """copy/initialize this JointState from another JointState object"""
        return cls(joint_state.pos, joint_state.vel, joint_state.acc)


class TrajectoryPoint(JointState):
    """ for defining trajectory point(s) that has position, velocity, acceleration and time"""
    def __init__(self, position, velocity, acceleration, time):
        JointState.__init__(self, position, velocity, acceleration)
        self.t = time

    @classmethod
    def FromTrajectoryPoint(cls, trajectory_point):
        """initialize/construct this trajectory point from another trajectory point object"""
        return cls(trajectory_point.pos, trajectory_point.vel, trajectory_point.acc, trajectory_point.t)

    @classmethod
    def FromJointStateAndTime(cls, joint_state, time):
        """initialize this trajectory point from a JointState object, and a specified time"""
        cls.FromJointState(joint_state)
        cls.t = time
        return cls

    def CopyFrom(self, trajectory_point):
        """Copy this trajectory point data from another trajectory point object"""
        self.t = trajectory_point.t
        self.pos = trajectory_point.pos
        self.vel = trajectory_point.vel
        self.acc = trajectory_point.acc


class RobotKinematics:
    def __init__(self, number_of_joints, max_jerk, max_acceleration, max_velocity, max_decceleration=None):
        self.j_num = number_of_joints
        self.j_max = max_jerk
        self.a_max = max_acceleration
        self.v_max = max_velocity
        self.d_max = max_decceleration if max_decceleration is not None else self.a_max
    @classmethod
    def FromRobotKinematics(cls, robot_kinematics):
        cls.j_num = robot_kinematics.j_num
        cls.j_max = robot_kinematics.j_max
        cls.a_max = robot_kinematics.a_max
        cls.v_max = robot_kinematics.v_max
        return cls

    def CopyFrom(self, robot_kinematics):
        self.j_num = robot_kinematics.j_num
        self.j_max = robot_kinematics.j_max
        self.a_max = robot_kinematics.a_max
        self.v_max = robot_kinematics.v_max
