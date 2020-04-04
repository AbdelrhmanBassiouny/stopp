# STOPP

Safe Time Optimal Path Parameterization (STOPP) for serial robots, this produces
a jerk limited, joint synchronized trajectories for smooth and safe robot motion for a given predefined path.

## Dependencies
python3.5

numpy

## Installation
```
pip install stopp
```

## Sample Usage
```Python
import stopp

robot_path = numpy_array 
my_robot = stopp.Robot(n_joints, max_jerk, max_acc, max_vel)
trajectory = my_robot(robot_path, interp_time_step=0.004)
first_joint_time = trajectory[0].t
firs_joint_pos = trajectory[0].pos
first_joint_vel = trajectory[0].vel
first_joint_acc = trajectory[0].acc
```

## Sample Result

![](images/sample.png)
