"""The ROS 2 layer: everything about reaching the control PC lives here.

Arm and Gripper use this; project code should not need to import rclpy at all.
"""
import os
import threading
import time
import warnings

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


class Connection:
    """Joins the ROS network, keeps joint states and TF fresh in a background thread, and offers
    blocking helpers for actions and services."""

    def __init__(self, config, ros_args=None):
        self.config = config
        self._set_domain_id(config.domain_id)
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=ros_args)
        self.node = Node(config.node_name)
        self.log = self.node.get_logger()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self._joint_positions = {}
        self._joint_stamp = None
        self._joint_lock = threading.Lock()
        self.node.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

        # Spin in the background so TF, joint states and action responses keep flowing while
        # the caller blocks.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    # ---------------- Setup / teardown ----------------

    @staticmethod
    def _set_domain_id(domain_id):
        env = os.environ.get('ROS_DOMAIN_ID')
        if env is None or env == '':
            if domain_id is not None:
                os.environ['ROS_DOMAIN_ID'] = str(domain_id)
        elif domain_id is not None and env != str(domain_id):
            warnings.warn(f'ROS_DOMAIN_ID={env} from the environment overrides the configured '
                          f'{domain_id}; the robot will not be seen unless the control PC uses {env}')

    @property
    def domain_id(self):
        return os.environ.get('ROS_DOMAIN_ID', '0')

    def close(self):
        self._executor.shutdown(timeout_sec=1.0)
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    # ---------------- Robot state ----------------

    def _on_joint_state(self, msg):
        with self._joint_lock:
            self._joint_positions.update(zip(msg.name, msg.position))
            self._joint_stamp = time.time()

    def joints(self):
        """Latest 7 arm joint angles (rad) as a list, or None before /joint_states arrives."""
        with self._joint_lock:
            try:
                return [float(self._joint_positions[n]) for n in self.config.joint_names]
            except KeyError:
                return None

    def joint_dict(self):
        """Copy of every joint position seen on /joint_states, by name (arm and fingers)."""
        with self._joint_lock:
            return dict(self._joint_positions)

    def joints_age(self):
        """Seconds since the last /joint_states message (None if never)."""
        return None if self._joint_stamp is None else time.time() - self._joint_stamp

    def pose(self, link=None):
        """((x, y, z), (qx, qy, qz, qw)) of `link` (default ee_link) in base_frame, or None."""
        try:
            t = self.tf_buffer.lookup_transform(self.config.base_frame,
                                                link or self.config.ee_link, Time())
        except Exception:
            return None
        p, q = t.transform.translation, t.transform.rotation
        return (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)

    def wait_until_ready(self, timeout=10.0, need_moveit=True):
        """Block until joint states, TF and (optionally) MoveIt are visible; else explain why."""
        end = time.time() + timeout
        while time.time() < end and (self.joints() is None or self.pose() is None):
            time.sleep(0.05)
        if self.joints() is None:
            # Topics someone else publishes (our own subscriptions show up in the list too).
            topics = [n for n, _ in self.node.get_topic_names_and_types()
                      if n not in ('/rosout', '/parameter_events')
                      and self.node.count_publishers(n) > 0]
            hint = ('no other ROS topics are visible at all: the control PC is not reachable on '
                    'this domain/network' if not topics else
                    f'{len(topics)} topics are visible but not /joint_states with '
                    f'{self.config.joint_names[0]}..: is the robot launch running?')
            raise ConnectionError(
                f'No /joint_states after {timeout:.0f} s (ROS_DOMAIN_ID={self.domain_id}); {hint}. '
                "Check on both PCs: 'echo $ROS_DOMAIN_ID', 'ros2 topic list'.")
        if self.pose() is None:
            raise ConnectionError(f'No TF from {self.config.base_frame} to '
                                  f'{self.config.ee_link}; is robot_state_publisher running?')
        if need_moveit:
            probe = self.action_client(_move_group_action(), 'move_action')
            ok = probe.wait_for_server(timeout_sec=max(1.0, end - time.time()))
            probe.destroy()
            if not ok:
                raise ConnectionError(
                    "MoveIt action 'move_action' not found: is the MoveIt launch running on the "
                    f'control PC (ROS_DOMAIN_ID={self.domain_id})?')

    # ---------------- Blocking helpers ----------------

    def action_client(self, action_type, name):
        from rclpy.action import ActionClient
        return ActionClient(self.node, action_type, name)

    def service_client(self, srv_type, name):
        return self.node.create_client(srv_type, name)

    @staticmethod
    def wait(future, timeout):
        """Wait for an rclpy future (filled by the background thread). True if done in time."""
        end = time.time() + timeout
        while not future.done():
            if time.time() > end:
                return False
            time.sleep(0.02)
        return True

    def run_action(self, client, goal, name, timeout=30.0, with_status=False):
        """Send an action goal and wait for its result.

        Returns the result message (None if rejected or timed out, the goal is then cancelled),
        or (result, status) with with_status=True. Raises RuntimeError if the server is missing.
        """
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action '{name}' not found; is it running on the control PC "
                               f'(ROS_DOMAIN_ID={self.domain_id})?')
        send_future = client.send_goal_async(goal)
        if not self.wait(send_future, 10.0):
            raise RuntimeError(f'No response from {name}')
        handle = send_future.result()
        if not handle.accepted:
            self.log.error(f'{name} goal rejected')
            return (None, None) if with_status else None
        result_future = handle.get_result_async()
        if not self.wait(result_future, timeout):
            handle.cancel_goal_async()
            self.log.error(f'{name} timed out; cancelled')
            return (None, None) if with_status else None
        res = result_future.result()
        return (res.result, res.status) if with_status else res.result

    def call_service(self, client, request, name, timeout=15.0):
        """Call a service and return its response. Raises RuntimeError if missing or silent."""
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Service '{name}' not found; is it running on the control PC "
                               f'(ROS_DOMAIN_ID={self.domain_id})?')
        future = client.call_async(request)
        if not self.wait(future, timeout):
            raise RuntimeError(f'No response from {name}')
        return future.result()


def _move_group_action():
    from moveit_msgs.action import MoveGroup
    return MoveGroup
