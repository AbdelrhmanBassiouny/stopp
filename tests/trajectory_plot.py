import matplotlib.pyplot as plt


def PlotTrajectory(traj, n_joints, fig_num=0):
    title_font = 30
    annot_font = 25
    labels_font = 20
    legend_font = 13
    line_size = 5
    scatter_size = 200
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', num=fig_num)
    f.set_size_inches(20, 10)
    ax1.set_ylabel("Position\n(rad)", fontsize=title_font)
    ax2.set_ylabel("Velocity\n(rad/sec)", fontsize=title_font)
    ax3.set_ylabel("Acceleration\n(rad/sec^2)", fontsize=title_font)
    ax3.set_xlabel("Time(secs)",fontsize=title_font)
    for ax in [ax1, ax2, ax3]:
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['left'].set_linewidth(2)
        ax.tick_params(labelsize=labels_font, length=6, width=2)
        ax.grid()
    for j in range(n_joints):
        ax3.plot(traj[j].t, traj[j].acc, label='joint {}'.format(j), linewidth=line_size)
        ax2.plot(traj[j].t, traj[j].vel, label='joint {}'.format(j), linewidth=line_size)
        ax1.plot(traj[j].t, traj[j].pos, label='joint {}'.format(j), linewidth=line_size)
    for ax in [ax1, ax2, ax3]:
        ax.legend(fontsize=legend_font)
    ax1.set_title("Synchronized Joint Trajectories", fontsize=title_font, fontweight="bold")
    plt.show()
