"""Arm motion through MoveIt (move_group on the control PC). Plain Python in and out.

Conventions: positions in metres and joint angles in radians, in the base frame (fr3_link0).
Orientations are (x, y, z, w) quaternions. Motion methods return True on success and False if
MoveIt tried and failed (reason in the log); they raise ValueError for unsafe/invalid input and
RuntimeError if the robot or MoveIt can't be reached.
"""
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint

from . import kinematics
from .trajectory import blend_start

try:  # needs franka_msgs on this PC
    from franka_msgs.action import ErrorRecovery
except ImportError:
    ErrorRecovery = None

try:  # needs controller_manager_msgs on this PC (sudo apt install ros-jazzy-controller-manager-msgs)
    from controller_manager_msgs.srv import (ListControllers, ListHardwareComponents,
                                             SetHardwareComponentState, SwitchController)
    from lifecycle_msgs.msg import State as LifecycleState
except ImportError:
    ListControllers = ListHardwareComponents = SetHardwareComponentState = None
    SwitchController = LifecycleState = None

_CODE_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items()
               if k.isupper() and isinstance(v, int)}
_CODE_HINTS = {
    MoveItErrorCodes.CONTROL_FAILED: 'the controller stopped (robot in reflex/error?): '
                                     "try arm.recover() ('recover' in the console), then move "
                                     'again',
    MoveItErrorCodes.START_STATE_IN_COLLISION: 'the current state counts as a collision',
    MoveItErrorCodes.PLANNING_FAILED: 'no path found; try a different target or ready() first',
    MoveItErrorCodes.TIMED_OUT: 'planning took too long; try a closer target',
}


def describe_error(code):
    """MoveIt error code -> 'name (hint)' for log messages."""
    text = f'{code} {_CODE_NAMES.get(code, "UNKNOWN")}'
    return f'{text}: {_CODE_HINTS[code]}' if code in _CODE_HINTS else text


class Arm:
    def __init__(self, conn, config, gripper=None):
        self.conn, self.config = conn, config
        self.gripper = gripper
        self.log = conn.log
        self._move = conn.action_client(MoveGroup, 'move_action')
        self._cartesian = conn.service_client(GetCartesianPath, 'compute_cartesian_path')
        self._execute = conn.action_client(ExecuteTrajectory, 'execute_trajectory')
        self._recovery = (conn.action_client(ErrorRecovery, config.error_recovery_action)
                          if ErrorRecovery is not None else None)
        cm = config.controller_manager.rstrip('/')
        if ListControllers is not None:
            self._list_controllers = conn.service_client(ListControllers, f'{cm}/list_controllers')
            self._switch_controller = conn.service_client(SwitchController,
                                                          f'{cm}/switch_controller')
            self._list_hardware = conn.service_client(ListHardwareComponents,
                                                      f'{cm}/list_hardware_components')
            self._set_hardware = conn.service_client(SetHardwareComponentState,
                                                     f'{cm}/set_hardware_component_state')

    # ---------------- State ----------------

    def joints(self):
        """Current 7 joint angles (rad), or None if /joint_states hasn't arrived."""
        return self.conn.joints()

    def pose(self):
        """Current ee_link pose: ((x, y, z), (qx, qy, qz, qw)), or None if no TF yet."""
        return self.conn.pose()

    @staticmethod
    def fk(q):
        """TCP (position (3,), rotation 3x3) for joint angles q, computed locally."""
        return kinematics.fk(q)

    def check_workspace(self, points):
        """Raise ValueError if any (x, y, z) is outside config.workspace."""
        for x, y, z in points:
            for name, value in zip('xyz', (x, y, z)):
                lo, hi = self.config.workspace[name]
                if not lo <= value <= hi:
                    raise ValueError(f'{name}={value:.3f} is outside the safe range [{lo}, {hi}]')

    # ---------------- Motion ----------------

    def _new_move_goal(self):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.config.group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.config.planning_time
        req.max_velocity_scaling_factor = self.config.velocity_scaling
        req.max_acceleration_scaling_factor = self.config.acceleration_scaling
        req.start_state.is_diff = True  # plan from the robot's current state
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _finish(self, result, what):
        if result is None:
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.log.error(f'{what} failed: {describe_error(result.error_code.val)}')
            return False
        return True

    def _current_orientation(self):
        pose = self.pose()
        if pose is None:
            raise RuntimeError(f'No TF from {self.config.base_frame} to {self.config.ee_link} yet')
        return pose[1]

    def move_to(self, x, y, z, orientation=None, timeout=60.0):
        """Plan and execute a collision-free move of ee_link to (x, y, z).

        orientation: (qx, qy, qz, qw), or None to keep the current one.
        """
        x, y, z = float(x), float(y), float(z)
        self.check_workspace([(x, y, z)])
        q = orientation if orientation is not None else self._current_orientation()
        quat = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

        goal = self._new_move_goal()
        target = Pose()
        target.position.x, target.position.y, target.position.z = x, y, z
        target.orientation = quat

        # Position: end-effector must land inside a 5 mm sphere around the target.
        region = BoundingVolume()
        region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005]))
        region.primitive_poses.append(target)
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.config.base_frame
        pos_c.link_name = self.config.ee_link
        pos_c.constraint_region = region
        pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.config.base_frame
        ori_c.link_name = self.config.ee_link
        ori_c.orientation = quat
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        goal.request.goal_constraints.append(constraints)
        return self._finish(self.conn.run_action(self._move, goal, 'move_action', timeout), 'Move')

    def move_joints(self, positions, timeout=60.0):
        """Plan and move in joint space to 7 joint angles (rad). Good for escaping singularities."""
        if len(positions) != 7:
            raise ValueError('need exactly 7 joint values')
        goal = self._new_move_goal()
        constraints = Constraints()
        for name, q in zip(self.config.joint_names, positions):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(q),
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        goal.request.goal_constraints.append(constraints)
        return self._finish(self.conn.run_action(self._move, goal, 'move_action', timeout),
                            'Joint move')

    def ready(self, open_gripper=False):
        """Move to config.ready_joints (gripper down, arm bent): a safe starting pose.

        open_gripper=True first opens the fingers fully. Leave it False while holding
        something you want to keep (e.g. the camera).
        """
        if open_gripper:
            if self.gripper is None or not self.gripper.available:
                raise RuntimeError('open_gripper=True needs the gripper (franka_msgs on this PC)')
            if not self.gripper.open():
                return False
        return self.move_joints(self.config.ready_joints)

    def move_linear(self, points, orientation=None, timeout=120.0):
        """Move through (x, y, z) waypoints along straight lines, as ONE continuous trajectory.

        orientation: held throughout; None keeps the current one.
        """
        points = [(float(x), float(y), float(z)) for x, y, z in points]
        self.check_workspace(points)
        q = orientation if orientation is not None else self._current_orientation()
        quat = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))

        req = GetCartesianPath.Request()
        req.header.frame_id = self.config.base_frame
        req.start_state.is_diff = True
        req.group_name = self.config.group
        req.link_name = self.config.ee_link
        req.max_step = 0.005          # interpolate every 5 mm along each edge
        req.avoid_collisions = True
        # Newer MoveIt versions accept speed scaling here; older ones return full speed,
        # in which case the trajectory is slowed down below before execution.
        server_scales_speed = hasattr(req, 'max_velocity_scaling_factor')
        if server_scales_speed:
            req.max_velocity_scaling_factor = self.config.velocity_scaling
            req.max_acceleration_scaling_factor = self.config.acceleration_scaling
        for x, y, z in points:
            p = Pose()
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation = quat
            req.waypoints.append(p)

        res = self.conn.call_service(self._cartesian, req, 'compute_cartesian_path')
        if res.fraction < 0.999:
            self.log.error(f'Only {res.fraction * 100:.0f}% of the path could be planned; '
                           'not executing')
            return False
        trajectory = res.solution
        if not server_scales_speed:
            _slow_down(trajectory, self.config.velocity_scaling)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        return self._finish(self.conn.run_action(self._execute, goal, 'execute_trajectory',
                                                 timeout), 'Path execution')

    def execute_joint_trajectory(self, times, positions, velocities=None, accelerations=None,
                                 max_start_offset=0.05, timeout=None):
        """Execute your own timed joint trajectory exactly (no MoveIt planning or collision check).

        times: (N,) seconds from start; positions: (N, 7) rad; velocities/accelerations optional.
        Use fr3.trajectory.time_trajectory() to time a joint path. The arm must already be within
        max_start_offset rad of positions[0] (get there with move_joints); the small remaining
        offset is faded out over the first seconds so the motion starts exactly where the arm is.
        """
        import numpy as np
        q_now = self.joints()
        if q_now is None:
            raise RuntimeError(f'No /joint_states received (ROS_DOMAIN_ID={self.conn.domain_id})')
        times = np.asarray(times, dtype=float)
        pos = np.array(positions, dtype=float)
        vel = None if velocities is None else np.array(velocities, dtype=float)
        acc = None if accelerations is None else np.array(accelerations, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 7 or len(times) != len(pos):
            raise ValueError('positions must be (N, 7) with one time per row')
        if np.any(np.diff(times) <= 0):
            raise ValueError('times must be strictly increasing')
        if np.any(pos < kinematics.LOWER) or np.any(pos > kinematics.UPPER):
            raise ValueError('trajectory leaves the FR3 joint limits')
        off = float(np.max(np.abs(np.asarray(q_now) - pos[0])))
        if off > max_start_offset:
            raise RuntimeError(f'Arm is {off:.3f} rad from the trajectory start; '
                               'move there first with move_joints(positions[0])')
        blend_start(times, pos, vel, acc, q_now)

        goal = ExecuteTrajectory.Goal()
        traj = goal.trajectory.joint_trajectory
        traj.joint_names = self.config.joint_names
        for i, t in enumerate(times):
            pt = JointTrajectoryPoint()
            pt.positions = pos[i].tolist()
            if vel is not None:
                pt.velocities = vel[i].tolist()
            if acc is not None:
                pt.accelerations = acc[i].tolist()
            sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
            pt.time_from_start.sec, pt.time_from_start.nanosec = sec, nsec
            traj.points.append(pt)
        timeout = times[-1] + 30.0 if timeout is None else timeout
        return self._finish(self.conn.run_action(self._execute, goal, 'execute_trajectory',
                                                 timeout), 'Trajectory execution')

    # ---------------- Recovery ----------------

    def controller_state(self, name=None):
        """ros2_control state of a controller ('active', 'inactive', ...; default: the arm
        controller), or None if it can't be queried (controller_manager_msgs missing here)."""
        if ListControllers is None:
            return None
        res = self.conn.call_service(self._list_controllers, ListControllers.Request(),
                                     'list_controllers', 5.0)
        name = name or self.config.arm_controller
        return next((c.state for c in res.controller if c.name == name), 'not loaded')

    def hardware_states(self):
        """{component name: lifecycle label} of the ros2_control hardware (normally
        {'FrankaHardwareInterface': 'active'}), or None if it can't be queried."""
        if ListHardwareComponents is None:
            return None
        res = self.conn.call_service(self._list_hardware, ListHardwareComponents.Request(),
                                     'list_hardware_components', 5.0)
        return {c.name: c.state.label for c in res.component}

    def _activate_hardware(self):
        """Re-activate hardware components that ros2_control switched off (after a reflex it
        deactivates the Franka hardware, and then every controller switch is rejected)."""
        for name, label in self.hardware_states().items():
            if label == 'active':
                continue
            self.log.info(f'hardware {name} is {label}; activating it')
            req = SetHardwareComponentState.Request()
            req.name = name
            req.target_state = LifecycleState(id=LifecycleState.PRIMARY_STATE_ACTIVE,
                                              label='active')
            res = self.conn.call_service(self._set_hardware, req,
                                         'set_hardware_component_state', 15.0)
            if not res.ok:
                self.log.error(f'could not activate hardware {name} (now {res.state.label})')
                return False
        return True

    def _switch(self, activate=(), deactivate=()):
        req = SwitchController.Request()
        req.activate_controllers, req.deactivate_controllers = list(activate), list(deactivate)
        req.strictness = SwitchController.Request.STRICT
        req.timeout.sec = 5
        res = self.conn.call_service(self._switch_controller, req, 'switch_controller', 10.0)
        if not res.ok:
            self.log.error(f'switch_controller failed: {getattr(res, "message", "")}')
        return res.ok

    def recover(self, restart_controller=False):
        """Get the arm moving again after a reflex/error, without touching the control PC.

        1. Franka error recovery (clears the reflex; needs franka_msgs here).
        2. Re-activates the Franka hardware in ros2_control if it was switched off.
        3. Re-activates the arm controller if it is no longer active (2 and 3 need
           controller_manager_msgs here). restart_controller=True restarts it even if active.
        Returns True when the robot is cleared and the arm controller is active.
        """
        from action_msgs.msg import GoalStatus
        if self._recovery is None:
            self.log.warning('franka_msgs not available here: skipping error recovery')
        else:
            _, status = self.conn.run_action(self._recovery, ErrorRecovery.Goal(),
                                             'error_recovery', 20.0, with_status=True)
            if status != GoalStatus.STATUS_SUCCEEDED:
                self.log.error('Error recovery failed; check the robot (user stop released? '
                               'lights blue/white, not red?)')
                return False

        if ListControllers is None:
            self.log.warning('controller_manager_msgs not available here: cannot check or '
                             're-activate the arm controller (sudo apt install '
                             'ros-jazzy-controller-manager-msgs)')
            return True
        if not self._activate_hardware():
            self.log.error('restart the launch on the control PC')
            return False
        name = self.config.arm_controller
        state = self.controller_state()
        if state == 'active' and restart_controller:
            self._switch(deactivate=[name])
            state = self.controller_state()
        if state != 'active':
            self.log.info(f'{name} is {state}; activating it')
            self._switch(activate=[name])
            state = self.controller_state()
        if state != 'active':
            self.log.error(f'{name} is still {state}; restart the launch on the control PC')
            return False
        return True


def _slow_down(trajectory, scale):
    """Stretch a trajectory's timing by 1/scale (scale in 0..1)."""
    for pt in trajectory.joint_trajectory.points:
        t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) / scale
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        pt.velocities = [v * scale for v in pt.velocities]
        pt.accelerations = [a * scale * scale for a in pt.accelerations]
