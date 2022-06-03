import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from stopp import stopp
from stopp.trajectory_utils import ConcatTraj
from math import pi
import numpy as np
from random import randint, seed
import time as t
from trajectory_plot import PlotTrajectory

j_max = 0.05
a_max = 0.05
d_max = 0.03
v_max = 0.3
n_points = 31


def test(s, e, vs=0, ve=0, fig_num=0, vel_mode=True, plot=False):

    n_joints = len(s)
    time_step = 0.004
    path = np.array([np.linspace(s[j], e[j], n_points) for j in range(n_joints)])#*(pi/180)

    st = t.time()
    my_rob = stopp.Robot(n_joints, j_max, a_max, v_max, d_max=d_max)
    trajectory = my_rob.TimeParameterizePath(
        path, vi=vs, vf=ve, interp_time_step=time_step, velocity_mode=vel_mode)
    et = t.time()

    # print("Processing Time = {}".format(et-st))
    # print(min(np.diff(trajectory[0].t)), max(np.diff(trajectory[0].t)))
    # print(np.diff(trajectory[0].t))
    if plot:
        PlotTrajectory(trajectory, n_joints, fig_num)

    return et-st, trajectory


def stress_test(n_tests):
    seed(2)
    global j_max, a_max, v_max
    n_joints = 3
    total_time = 0.0
    for i in range(n_tests):
        j_max = randint(1, 1000)
        a_max = randint(1, 100)
        v_max = randint(1, 10)
        s = [randint(-400, 400) for _ in range(n_joints)]
        e = [randint(-400, 400) for _ in range(n_joints)]
        total_time += test(s, e, fig_num=i)[0]
    print("avg_processing_time = {}".format(total_time/n_tests))


if __name__ == "__main__":
    # stress_test(1000)
    # test([0, 0, 0], [100, 50, 25], vs=1, ve=0)
    # a_max = 0.03
    # _, traj_2 = test([0.5], [1], vs=v_max, ve=0)
    # print("vel = ", traj_1[0].vel[-1])
    # a_max = 0.05
    # _, traj_1 = test([0], [0.5], vs=0, ve=traj_2[0].vel[0])
    # traj = ConcatTraj([traj_1[0], traj_2[0]])
    _, traj = test([0], [1], vs=0, ve=0, vel_mode=False)
    PlotTrajectory(traj,1)
    # PlotTrajectory(traj_2, 1)
