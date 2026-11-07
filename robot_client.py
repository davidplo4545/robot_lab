#!/usr/bin/env python3
"""Send Cartesian pose goals to MoveIt (move_group) for a Franka FR3, from any PC on the ROS network.

MoveIt plans a collision-free path and executes it with the joint trajectory controller.
Only needs moveit_msgs and tf2_ros on this machine; no MoveIt install required here.

Usage:
  python3 franka_moveit_client.py
  python3 franka_moveit_client.py --ros-args -p group:=fr3_arm -p ee_link:=fr3_hand_tcp
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetCartesianPath
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from orbit_planner import AXES, plan_orbit, time_trajectory

try:  # gripper actions; the arm still works without them
    from franka_msgs.action import Grasp, Homing
    from franka_msgs.action import Move as GripperMove
except ImportError:
    Grasp = Homing = GripperMove = None

try:
    import torch
except ImportError:
    torch = None

# Safe workspace box in metres, relative to the robot base. Tighten as needed.
WORKSPACE = {'x': (0.30, 0.75), 'y': (-0.30, 0.30), 'z': (0.10, 0.70)}

# Franka's standard "ready" pose (radians, joints 1-7): arm bent, gripper down, far from singularities.
READY_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# Camera orbit defaults (metres / seconds), used by the 'orbit' command.
ORBIT_CENTER = (0.509, -0.005, 0.1)   # object being filmed
ORBIT_RADIUS = 0.18                   # horizontal camera distance from the object (far side)
ORBIT_NEAR_RADIUS = 0.12              # ...on the robot's side: the path is an oval, since a wide
                                      # circle there makes the elbow fold up (self-collision stop)
ORBIT_HEIGHT = 0.20                   # camera height above the object -> 48-59 deg look-down
ORBIT_SECONDS = 40.0                  # one full lap; slower = sharper frames


class FrankaMoveItClient(Node):
    def __init__(self):
        super().__init__('franka_moveit_client')
        self.declare_parameter('group', 'fr3_arm')            # planning group from the SRDF
        self.declare_parameter('ee_link', 'fr3_hand_tcp')     # link that should reach the target
        self.declare_parameter('base_frame', 'fr3_link0')
        self.declare_parameter('velocity_scaling', 0.1)       # 0..1, start slow
        self.declare_parameter('acceleration_scaling', 0.1)
        self.declare_parameter('planning_time', 5.0)
        # Camera held in the gripper: which end-effector axis the lens looks along, and where
        # the lens is in the end-effector frame (metres). Change live with the 'cam' command.
        # Default: iPhone 17 held vertically and upside down, gripped across its width at the
        # bottom end, so the lens hangs ~12 cm beyond the fingertips (TCP +z), on the back face,
        # looking away from the robot in the ready pose. Measure and adjust with 'cam'.
        # (Flat phone with the lens facing down = '+z', but it folds the elbow.)
        self.declare_parameter('camera_axis', '+x')
        self.declare_parameter('camera_offset', [0.004, 0.0, 0.12])
        self.camera_axis = self.get_parameter('camera_axis').value
        self.camera_offset = [float(v) for v in self.get_parameter('camera_offset').value]
        if self.camera_axis not in AXES:
            raise ValueError(f'camera_axis must be one of {list(AXES)}')

        self.group = self.get_parameter('group').value
        self.ee_link = self.get_parameter('ee_link').value
        self.base_frame = self.get_parameter('base_frame').value
        self.vel = float(self.get_parameter('velocity_scaling').value)
        self.acc = float(self.get_parameter('acceleration_scaling').value)
        self.planning_time = float(self.get_parameter('planning_time').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, MoveGroup, 'move_action')
        self.cartesian_client = self.create_client(GetCartesianPath, 'compute_cartesian_path')
        self.execute_client = ActionClient(self, ExecuteTrajectory, 'execute_trajectory')

        # Latest arm joint angles, needed to start a pre-planned joint trajectory exactly here.
        self.joint_names = [self.base_frame.replace('link0', f'joint{i}') for i in range(1, 8)]
        self.joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

        self.declare_parameter('gripper_ns', '/franka_gripper')
        gripper_ns = self.get_parameter('gripper_ns').value.rstrip('/')
        if Grasp is not None:
            self.grasp_client = ActionClient(self, Grasp, f'{gripper_ns}/grasp')
            self.gripper_move_client = ActionClient(self, GripperMove, f'{gripper_ns}/move')
            self.homing_client = ActionClient(self, Homing, f'{gripper_ns}/homing')

    def _on_joint_state(self, msg):
        self.joint_positions.update(zip(msg.name, msg.position))

    def current_joints(self):
        """Current 7 arm joint angles, or None if /joint_states has not arrived yet."""
        try:
            return [self.joint_positions[n] for n in self.joint_names]
        except KeyError:
            return None

    def current_pose(self):
        """Current ee_link pose in base_frame from TF, or None if not available yet."""
        try:
            t = self.tf_buffer.lookup_transform(self.base_frame, self.ee_link, Time())
        except Exception:
            return None
        pose = Pose()
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation
        return pose

    def move_to(self, x, y, z, orientation=None, timeout=60.0):
        """Plan and execute a move. Call this from your PyTorch logic too (plain floats).

        If orientation is None, the current end-effector orientation is kept.
        Returns True on success.
        """
        x, y, z = float(x), float(y), float(z)
        for name, value in zip('xyz', (x, y, z)):
            lo, hi = WORKSPACE[name]
            if not lo <= value <= hi:
                raise ValueError(f'{name}={value:.3f} is outside the safe range [{lo}, {hi}]')

        if orientation is None:
            current = self.current_pose()
            if current is None:
                raise RuntimeError(f'No TF from {self.base_frame} to {self.ee_link} yet')
            orientation = current.orientation

        if not self.client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("MoveIt action server 'move_action' not found; is move_group running?")

        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.vel
        req.max_acceleration_scaling_factor = self.acc
        req.start_state.is_diff = True  # plan from the robot's current state

        target = Pose()
        target.position.x, target.position.y, target.position.z = x, y, z
        target.orientation = orientation

        # Position: end-effector must land inside a 5 mm sphere around the target.
        region = BoundingVolume()
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005])
        region.primitives.append(sphere)
        region.primitive_poses.append(target)
        pos_c = PositionConstraint()
        pos_c.header.frame_id = self.base_frame
        pos_c.link_name = self.ee_link
        pos_c.constraint_region = region
        pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header.frame_id = self.base_frame
        ori_c.link_name = self.ee_link
        ori_c.orientation = orientation
        ori_c.absolute_x_axis_tolerance = 0.05
        ori_c.absolute_y_axis_tolerance = 0.05
        ori_c.absolute_z_axis_tolerance = 0.05
        ori_c.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        req.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        send_future = self.client.send_goal_async(goal)
        if not self._wait(send_future, 10.0):
            raise RuntimeError('No response from MoveIt when sending goal')
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected by MoveIt')
            return False

        result_future = handle.get_result_async()
        if not self._wait(result_future, timeout):
            self.get_logger().error('Timed out waiting for the move to finish; cancelling')
            handle.cancel_goal_async()
            return False

        code = result_future.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return True
        self.get_logger().error(f'MoveIt failed with error code {code}')
        return False

    def move_cartesian_path(self, points, timeout=120.0):
        """Move through (x, y, z) waypoints along straight lines, as ONE continuous trajectory.

        The current end-effector orientation is held throughout. Returns True on success.
        """
        points = [(float(x), float(y), float(z)) for x, y, z in points]
        for x, y, z in points:
            for name, value in zip('xyz', (x, y, z)):
                lo, hi = WORKSPACE[name]
                if not lo <= value <= hi:
                    raise ValueError(f'waypoint {name}={value:.3f} is outside the safe range [{lo}, {hi}]')

        current = self.current_pose()
        if current is None:
            raise RuntimeError(f'No TF from {self.base_frame} to {self.ee_link} yet')
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("Service 'compute_cartesian_path' not found; is move_group running?")
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Action 'execute_trajectory' not found; is move_group running?")

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.start_state.is_diff = True
        req.group_name = self.group
        req.link_name = self.ee_link
        req.max_step = 0.005          # interpolate every 5 mm along each edge
        req.avoid_collisions = True
        # Newer MoveIt versions accept speed scaling here; older ones return full speed,
        # in which case the trajectory is slowed down below before execution.
        server_scales_speed = hasattr(req, 'max_velocity_scaling_factor')
        if server_scales_speed:
            req.max_velocity_scaling_factor = self.vel
            req.max_acceleration_scaling_factor = self.acc
        for x, y, z in points:
            p = Pose()
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation = current.orientation
            req.waypoints.append(p)

        future = self.cartesian_client.call_async(req)
        if not self._wait(future, 15.0):
            raise RuntimeError('No response from compute_cartesian_path')
        res = future.result()
        if res.fraction < 0.999:
            self.get_logger().error(
                f'Only {res.fraction * 100:.0f}% of the path could be planned; not executing')
            return False

        trajectory = res.solution
        if not server_scales_speed:
            self._slow_down(trajectory, self.vel)

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self.execute_client.send_goal_async(goal)
        if not self._wait(send_future, 10.0):
            raise RuntimeError('No response from MoveIt when sending trajectory')
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error('Trajectory rejected by MoveIt')
            return False
        result_future = handle.get_result_async()
        if not self._wait(result_future, timeout):
            self.get_logger().error('Timed out waiting for the path to finish; cancelling')
            handle.cancel_goal_async()
            return False
        code = result_future.result().result.error_code.val
        if code == MoveItErrorCodes.SUCCESS:
            return True
        self.get_logger().error(f'Execution failed with error code {code}')
        return False

    def move_to_joints(self, positions, timeout=60.0):
        """Plan and move in joint space to 7 joint angles (radians). Good for escaping singularities."""
        if len(positions) != 7:
            raise ValueError('need exactly 7 joint values')
        joint_prefix = self.base_frame.replace('link0', 'joint')  # fr3_link0 -> fr3_joint
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.vel
        req.max_acceleration_scaling_factor = self.acc
        req.start_state.is_diff = True
        constraints = Constraints()
        for i, q in enumerate(positions, start=1):
            constraints.joint_constraints.append(JointConstraint(
                joint_name=f'{joint_prefix}{i}', position=float(q),
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        req.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        result = self._run_action(self.client, goal, 'move_action', timeout)
        if result is None:
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'MoveIt failed with error code {result.error_code.val}')
            return False
        return True

    def move_to_ready(self):
        return self.move_to_joints(READY_JOINTS)

    # ---------------- Gripper ----------------

    def _require_gripper(self):
        if Grasp is None:
            raise RuntimeError('franka_msgs is not available on this PC; build it (see cheat sheet)')

    def _run_action(self, client, goal, name, timeout=30.0):
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action '{name}' not found; check the gripper_ns parameter")
        send_future = client.send_goal_async(goal)
        if not self._wait(send_future, 10.0):
            raise RuntimeError(f'No response from {name}')
        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error(f'{name} goal rejected')
            return None
        result_future = handle.get_result_async()
        if not self._wait(result_future, timeout):
            handle.cancel_goal_async()
            self.get_logger().error(f'{name} timed out')
            return None
        return result_future.result().result

    def gripper_home(self):
        """Calibrate the fingers (opens fully). Do this once after power-on."""
        self._require_gripper()
        result = self._run_action(self.homing_client, Homing.Goal(), 'gripper homing')
        if result is None or not result.success:
            self.get_logger().error(f'Gripper homing failed: {getattr(result, "error", "")}')
            return False
        return True

    def gripper_open(self, width=0.08, speed=0.1):
        """Move the fingers to `width` metres apart (0.08 = fully open)."""
        self._require_gripper()
        goal = GripperMove.Goal()
        goal.width = float(width)
        goal.speed = float(speed)
        result = self._run_action(self.gripper_move_client, goal, 'gripper move')
        if result is None or not result.success:
            self.get_logger().error(f'Gripper move failed: {getattr(result, "error", "")}')
            return False
        return True

    def gripper_grasp(self, width, force=20.0, speed=0.05, epsilon=0.01):
        """Close on an object about `width` metres wide with `force` newtons.

        Succeeds only if the fingers stop within `epsilon` of `width`, i.e. something was gripped.
        """
        self._require_gripper()
        goal = Grasp.Goal()
        goal.width = float(width)
        goal.force = float(force)
        goal.speed = float(speed)
        goal.epsilon.inner = float(epsilon)
        goal.epsilon.outer = float(epsilon)
        result = self._run_action(self.grasp_client, goal, 'gripper grasp')
        if result is None or not result.success:
            self.get_logger().error(
                'Grasp failed: nothing gripped, or the object width is more than '
                f'{epsilon * 1000:.0f} mm off. {getattr(result, "error", "")}')
            return False
        return True

    # ---------------- Pick and place ----------------

    def _check_points(self, points):
        for x, y, z in points:
            for name, value in zip('xyz', (x, y, z)):
                lo, hi = WORKSPACE[name]
                if not lo <= value <= hi:
                    raise ValueError(f'{name}={value:.3f} is outside the safe range [{lo}, {hi}]')

    def _run_steps(self, steps):
        for label, step in steps:
            print(f'  {label} ...')
            if not step():
                print(f'  Stopped: "{label}" failed')
                return False
        return True

    def pick(self, x, y, z, width, force=20.0, approach=0.10):
        """Open, move above (x, y, z), lower straight down, grasp, lift straight up."""
        x, y, z = float(x), float(y), float(z)
        above = (x, y, z + approach)
        self._check_points([(x, y, z), above])
        return self._run_steps([
            ('Opening gripper', lambda: self.gripper_open()),
            ('Moving above object', lambda: self.move_to(*above)),
            ('Lowering', lambda: self.move_cartesian_path([(x, y, z)])),
            ('Grasping', lambda: self.gripper_grasp(width, force)),
            ('Lifting', lambda: self.move_cartesian_path([above])),
        ])

    def place(self, x, y, z, approach=0.10):
        """Move above (x, y, z), lower straight down, open, lift straight up."""
        x, y, z = float(x), float(y), float(z)
        above = (x, y, z + approach)
        self._check_points([(x, y, z), above])
        return self._run_steps([
            ('Moving above target', lambda: self.move_to(*above)),
            ('Lowering', lambda: self.move_cartesian_path([(x, y, z)])),
            ('Releasing', lambda: self.gripper_open()),
            ('Lifting', lambda: self.move_cartesian_path([above])),
        ])

    # ---------------- Camera orbit ----------------

    def orbit_plan(self, center=ORBIT_CENTER, radius=ORBIT_RADIUS, height=ORBIT_HEIGHT,
                   near_radius=ORBIT_NEAR_RADIUS):
        """Plan the whole orbit in joint space (see orbit_planner.py). Raises ValueError if unsafe.

        near_radius < radius gives an oval that stays closer on the robot's side.
        """
        near_radius = None if near_radius is None or near_radius >= radius else near_radius
        plan = plan_orbit(center, radius, height, self.camera_axis, self.camera_offset,
                          q_now=self.current_joints(), near_radius=near_radius)
        self._check_points([tuple(p) for p, _ in plan['poses']])
        return plan

    def orbit_go_to_start(self, plan):
        """Collision-checked MoveIt move to the joint configuration where the loop starts."""
        return self.move_to_joints(plan['q'][0].tolist())

    def orbit_run(self, plan, duration=ORBIT_SECONDS):
        """Execute the pre-planned loop as one smooth joint trajectory (about `duration` s)."""
        q_now = self.current_joints()
        if q_now is None:
            raise RuntimeError('No /joint_states received; check ROS_DOMAIN_ID')
        if max(abs(a - b) for a, b in zip(q_now, plan['q'][0])) > 0.05:
            raise RuntimeError('Arm is not at the orbit start; run the orbit command again')
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Action 'execute_trajectory' not found; is move_group running?")

        duration, times, pos, vel, acc = time_trajectory(plan['q'], duration)
        # Start exactly where the arm is (MoveIt checks this): fade the small leftover offset
        # from the start move out smoothly over the first seconds instead of in one jump.
        delta = [a - b for a, b in zip(q_now, pos[0])]
        fade = min(3.0, duration / 4)
        for i, t in enumerate(times):
            if t < fade:
                w = 0.5 * (1 + math.cos(math.pi * t / fade))
                dw = -0.5 * math.pi / fade * math.sin(math.pi * t / fade)
                ddw = -0.5 * (math.pi / fade) ** 2 * math.cos(math.pi * t / fade)
                pos[i] += [d * w for d in delta]
                vel[i] += [d * dw for d in delta]
                acc[i] += [d * ddw for d in delta]
        goal = ExecuteTrajectory.Goal()
        traj = goal.trajectory.joint_trajectory
        traj.joint_names = self.joint_names
        for t, p, v, a in zip(times, pos, vel, acc):
            pt = JointTrajectoryPoint()
            pt.positions, pt.velocities, pt.accelerations = p.tolist(), v.tolist(), a.tolist()
            sec, nsec = divmod(int(round(t * 1e9)), 1_000_000_000)
            pt.time_from_start.sec, pt.time_from_start.nanosec = sec, nsec
            traj.points.append(pt)
        print(f'  Lap time {duration:.0f} s')

        result = self._run_action(self.execute_client, goal, 'execute_trajectory', duration + 30.0)
        if result is None:
            return False
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'Orbit execution failed with error code {result.error_code.val}')
            return False
        return True

    @staticmethod
    def _slow_down(trajectory, scale):
        """Stretch a trajectory's timing by 1/scale (scale in 0..1)."""
        for pt in trajectory.joint_trajectory.points:
            t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) / scale
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            pt.velocities = [v * scale for v in pt.velocities]
            pt.accelerations = [a * scale * scale for a in pt.accelerations]

    @staticmethod
    def _wait(future, timeout):
        end = time.time() + timeout
        while not future.done():
            if time.time() > end:
                return False
            time.sleep(0.02)
        return True


def main(args=None):
    rclpy.init(args=args)
    node = FrankaMoveItClient()

    if torch is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'PyTorch active, computing on: {device}')

    # Spin ROS in the background so TF and action responses keep flowing.
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print(f'Group: {node.group}, end-effector: {node.ee_link}, frame: {node.base_frame}')
    print(f'Safe workspace: {WORKSPACE}')
    print('Commands (metres):\n'
          '  x y z                 move to a point\n'
          '  p                     print current pose\n'
          '  ready                 go to the safe ready pose (joint space)\n'
          '  t [size]              draw a triangle (default 0.10)\n'
          '  home                  calibrate the gripper (once after power-on)\n'
          '  open [width]          open the gripper (default 0.08 = fully)\n'
          '  grasp width [force]   close on an object of that width (default 20 N)\n'
          '  pick x y z width      open, move above, lower, grasp, lift\n'
          '  place x y z           move above, lower, release, lift\n'
          '  orbit [cx cy cz [r h [sec [near]]]]  move the camera around an object, aimed at it\n'
          f'                        (default centre {ORBIT_CENTER}, r={ORBIT_RADIUS}, '
          f'h={ORBIT_HEIGHT}, {ORBIT_SECONDS:.0f} s,\n'
          f'                        near={ORBIT_NEAR_RADIUS} = radius on the robot side; '
          'near=r for a circle)\n'
          '  cam [axis [ox oy oz]] show/set the gripper axis the lens faces (+x,-x,+y,-y,+z,-z)\n'
          '                        and the lens offset from the gripper TCP (metres)\n'
          '  q                     quit')
    print(f'Camera: lens looks along {node.camera_axis}, offset {node.camera_offset}')
    print('NOTE: the planner does not see the object, the table or the phone; first runs slow, '
          'hand on the user stop.')
    gripper_cmds = ('home', 'open', 'grasp', 'pick', 'place')
    try:
        while rclpy.ok():
            line = input('> ').strip().lower()
            if line == 'q':
                break
            if line == 'ready':
                try:
                    print('Moving to ready pose ...')
                    print('Done.' if node.move_to_ready() else 'Move failed (see log above).')
                except (ValueError, RuntimeError) as e:
                    print(f'Rejected: {e}')
                continue
            parts = line.split()
            if parts and parts[0] == 'cam':
                try:
                    if len(parts) not in (1, 2, 5) or (len(parts) > 1 and parts[1] not in AXES):
                        raise ValueError(f'usage: cam [axis [ox oy oz]], axis one of {list(AXES)}')
                    if len(parts) > 1:
                        node.camera_axis = parts[1]
                    if len(parts) == 5:
                        node.camera_offset = [float(v) for v in parts[2:5]]
                    print(f'Camera: lens looks along {node.camera_axis}, '
                          f'offset {node.camera_offset}')
                except ValueError as e:
                    print(f'Rejected: {e}')
                continue
            if parts and parts[0] == 'orbit':
                try:
                    vals = [float(a) for a in parts[1:]]
                    if len(vals) not in (0, 3, 5, 6, 7):
                        raise ValueError('usage: orbit [cx cy cz [radius height [seconds '
                                         '[near_radius]]]]')
                    center = tuple(vals[0:3]) if vals else ORBIT_CENTER
                    radius, height = vals[3:5] if len(vals) >= 5 else (ORBIT_RADIUS, ORBIT_HEIGHT)
                    seconds = vals[5] if len(vals) >= 6 else ORBIT_SECONDS
                    near = min(vals[6] if len(vals) == 7 else ORBIT_NEAR_RADIUS, radius)
                    shape = (f'radius {radius:.3f} m' if near >= radius else
                             f'oval {near:.3f} m (robot side) to {radius:.3f} m')
                    tilt = (f'{math.degrees(math.atan2(height, near)):.0f} deg' if near >= radius
                            else f'{math.degrees(math.atan2(height, radius)):.0f}-'
                                 f'{math.degrees(math.atan2(height, near)):.0f} deg')
                    print(f'Orbit around {center}: {shape}, {height:.3f} m above, '
                          f'looking down {tilt}, camera {node.camera_axis}')
                    print('  Planning (checking camera rolls and arm configurations) ...')
                    plan = node.orbit_plan(center, radius, height, near)
                    print(f'  Found a safe loop: closest joint limit {plan["margin"]:.2f} rad, '
                          f'largest joint turn to reach the start '
                          f'{math.degrees(plan["start_move"]):.0f} deg')
                    print('  Moving to start pose ...')
                    if not node.orbit_go_to_start(plan):
                        print('  Could not reach the start pose (see log above).')
                        continue
                    answer = input('  At start. Check the camera points at the object, start '
                                   'recording, then press Enter to orbit (q to cancel): ')
                    if answer.strip().lower() == 'q':
                        print('  Orbit cancelled.')
                        continue
                    print('  Orbiting ...')
                    ok = node.orbit_run(plan, seconds)
                    print('Done.' if ok else 'Orbit failed (see log above). Try a larger height, '
                          'a smaller radius, or check the camera axis.')
                except (ValueError, RuntimeError) as e:
                    print(f'Rejected: {e}')
                continue
            if parts and parts[0] in gripper_cmds:
                cmd, args = parts[0], parts[1:]
                try:
                    vals = [float(a) for a in args]
                    if cmd == 'home':
                        ok = node.gripper_home()
                    elif cmd == 'open':
                        ok = node.gripper_open(*vals[:1])
                    elif cmd == 'grasp':
                        if not 1 <= len(vals) <= 2:
                            raise ValueError('usage: grasp width [force]')
                        ok = node.gripper_grasp(*vals)
                    elif cmd == 'pick':
                        if len(vals) != 4:
                            raise ValueError('usage: pick x y z width')
                        ok = node.pick(*vals)
                    else:  # place
                        if len(vals) != 3:
                            raise ValueError('usage: place x y z')
                        ok = node.place(*vals)
                    print('Done.' if ok else f'{cmd} failed (see log above).')
                except (ValueError, RuntimeError) as e:
                    print(f'Rejected: {e}')
                continue
            if line.startswith('t'):
                parts = line.split()
                try:
                    size = float(parts[1]) if len(parts) > 1 else 0.10
                except ValueError:
                    print("Usage: 't' or 't 0.08' (side length in metres)")
                    continue
                pose = node.current_pose()
                if pose is None:
                    print('No TF yet; is the robot/MoveIt launch running?')
                    continue
                x0, y0, z0 = pose.position.x, pose.position.y, pose.position.z
                # Equilateral triangle in the horizontal plane, starting and ending here.
                corners = [(x0 + size, y0, z0),
                           (x0 + size / 2, y0 + size * 0.866, z0),
                           (x0, y0, z0)]
                try:
                    print(f'Drawing a {size:.2f} m triangle at z={z0:.3f} ...')
                    ok = node.move_cartesian_path(corners)
                    print('Done.' if ok else 'Triangle failed (see log above).')
                except (ValueError, RuntimeError) as e:
                    print(f'Rejected: {e}')
                continue
            if line == 'p':
                pose = node.current_pose()
                if pose is None:
                    print('No TF yet; is the robot/MoveIt launch running?')
                else:
                    p, o = pose.position, pose.orientation
                    print(f'Position: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  '
                          f'Orientation xyzw: ({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f})')
                continue
            try:
                x, y, z = (float(v) for v in line.split())
            except ValueError:
                print("Enter three numbers, e.g. '0.5 0.0 0.4'")
                continue
            try:
                print(f'Planning to ({x:.3f}, {y:.3f}, {z:.3f}) ...')
                ok = node.move_to(x, y, z)
                print('Done.' if ok else 'Move failed (see log above).')
            except (ValueError, RuntimeError) as e:
                print(f'Rejected: {e}')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# pick 0.509 -0.005 0.1 0.06
# place 0.509 -0.005 0.1