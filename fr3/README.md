# fr3: Franka FR3 from plain Python

Copy this `fr3/` folder into a new project and program the robot without touching ROS:

```python
from fr3 import connect

with connect() as robot:                 # joins ROS domain 42, waits for the robot + MoveIt
    robot.arm.ready()
    robot.arm.move_to(0.5, 0.0, 0.3)     # metres, base frame; keeps the current orientation
    robot.gripper.grasp(0.06)
    print(robot.arm.joints(), robot.arm.pose())
```

## What must be running

**Control PC (connected to the robot):** the franka_ros2 + MoveIt launch, with the same
`ROS_DOMAIN_ID` (42 by default):

```bash
export ROS_DOMAIN_ID=42
# TODO: paste the exact launch command used on the control PC here, e.g.
# ros2 launch <moveit_config_package> <launch_file> robot_ip:=<robot ip>
```

`fr3` uses only these standard interfaces:

| Interface | Used by |
|---|---|
| `/joint_states`, TF `fr3_link0` -> `fr3_hand_tcp` | `arm.joints()`, `arm.pose()` |
| `move_action` (move_group) | `move_to`, `move_joints`, `ready` |
| `compute_cartesian_path`, `execute_trajectory` (move_group) | `move_linear`, `execute_joint_trajectory` |
| `/franka_gripper/{homing,move,grasp}` (optional) | `gripper.*` |
| `/action_server/error_recovery` (optional) | `arm.recover()` |
| `/controller_manager/list_controllers`, `/controller_manager/switch_controller` (optional) | `arm.recover()`, `arm.controller_state()` |

**This PC:** ROS 2 Jazzy with `moveit_msgs` and `tf2_ros`; numpy and scipy. For the gripper and
`recover()`, also build `franka_msgs` and source it:

```bash
source /opt/ros/jazzy/setup.bash
source ~/robot_workspace/franka_msgs_ws/install/setup.bash   # optional: gripper / recover
python3 your_script.py
```

You don't need to export `ROS_DOMAIN_ID`: `connect()` sets it from `RobotConfig.domain_id` when
the terminal hasn't (an exported value wins, with a warning if it differs). If the robot can't be
seen, `connect()` raises `ConnectionError` and says whether *nothing* is visible (network/domain
problem) or the robot launch is missing.

## API

`connect(config=None, wait=True, wait_timeout=10, need_moveit=True)` -> `Robot` with:

**`robot.arm`**
- `joints()` -> 7 angles (rad) or None. `pose()` -> `((x, y, z), (qx, qy, qz, qw))` or None.
- `move_to(x, y, z, orientation=None)`: collision-free MoveIt move; orientation as a quaternion.
- `move_linear([(x, y, z), ...], orientation=None)`: straight-line path through waypoints.
- `move_joints([q1..q7])`, `ready(open_gripper=False)`. `open_gripper=True` opens the
  fingers fully before moving; keep it False while holding something.
- `execute_joint_trajectory(times, positions, velocities=None, accelerations=None)`: run your
  own timed joint path exactly (no planning/collision check). Get to `positions[0]` first with
  `move_joints`. Time a path with `fr3.trajectory.time_trajectory(qs, seconds)`.
- `recover(restart_controller=False)`: after a reflex/error (red light, MoveIt error -4) runs
  Franka error recovery, re-activates the Franka hardware in ros2_control if it was switched
  off, then re-activates `fr3_arm_controller` if it went inactive. No need
  to restart the control PC launch. Needs `franka_msgs` and
  `ros-jazzy-controller-manager-msgs` on this PC.
- `controller_state(name=None)`: `'active'`, `'inactive'`, ... (default: the arm controller).
  `connect()` warns at startup if the arm controller is not active.
- `check_workspace(points)`: raises ValueError outside `config.workspace`.

Motion methods return True/False (reason in the log, e.g. `-4 CONTROL_FAILED: ... try
arm.recover()`) and raise ValueError (unsafe input) or RuntimeError (robot/MoveIt unreachable).

**`robot.gripper`**: `home()`, `open(width=0.08)`, `grasp(width, force=20)`, `available`.

**`fr3.kinematics`** (no ROS): `fk(q)`, `ik(p, R, seed)`, `jacobian(frames(q))`, `LOWER`/`UPPER`
limits, `READY_JOINTS`, `wrist_to_upper_arm(q)` (keep >= 0.35 m or the FR3 stops with a
self-collision error), rotation helpers.

**`fr3.trajectory`** (no ROS): `time_trajectory(qs, duration)` -> `(duration, t, pos, vel, acc)`.

## Settings

```python
from fr3 import RobotConfig, connect
cfg = RobotConfig(velocity_scaling=0.2, workspace={'x': (0.3, 0.7), 'y': (-0.3, 0.3), 'z': (0.1, 0.6)})
robot = connect(cfg)
```

Fields: `domain_id`, `group`, `ee_link`, `base_frame`, `velocity_scaling`, `acceleration_scaling`,
`planning_time`, `gripper_ns`, `error_recovery_action`, `controller_manager`, `arm_controller`,
`workspace`, `ready_joints`.

## Safety

MoveIt only knows the robot itself (not your table, objects or tools), and
`execute_joint_trajectory` checks nothing but joint limits. Start slow, with a hand on the user
stop.
