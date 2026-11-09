"""fr3: drive a Franka FR3 over ROS 2 / MoveIt from plain Python.

    from fr3 import connect
    with connect() as robot:
        robot.arm.ready()
        robot.arm.move_to(0.5, 0.0, 0.3)
        robot.gripper.grasp(0.06)
        print(robot.arm.joints(), robot.arm.pose())

Copy this folder into a new project; see README.md for what must run on the control PC.
kinematics, trajectory and config import without ROS (offline planning and tests).
"""
from .config import RobotConfig

__all__ = ['RobotConfig', 'Robot', 'connect']


class Robot:
    """The connected robot: .arm, .gripper, and .conn (the ROS layer, rarely needed)."""

    def __init__(self, config=None, ros_args=None, wait=True, wait_timeout=10.0,
                 need_moveit=True):
        from .arm import Arm
        from .connection import Connection
        from .gripper import Gripper
        self.config = config or RobotConfig()
        self.conn = Connection(self.config, ros_args)
        try:
            self.gripper = Gripper(self.conn, self.config)
            self.arm = Arm(self.conn, self.config, self.gripper)
            if wait:
                self.conn.wait_until_ready(wait_timeout, need_moveit)
                if need_moveit:
                    self._warn_if_controller_inactive()
        except BaseException:
            self.conn.close()
            raise

    def _warn_if_controller_inactive(self):
        try:
            state = self.arm.controller_state()
            hardware = self.arm.hardware_states() or {}
        except RuntimeError:
            return  # controller_manager not reachable: moves will report the problem
        off = [f'{n} is {s}' for n, s in hardware.items() if s != 'active']
        if off:
            self.conn.log.warning(f'robot hardware switched off ({", ".join(off)}): moves will '
                                  "fail (-4); call robot.arm.recover() ('recover' in the console)")
        elif state is not None and state != 'active':
            self.conn.log.warning(f'{self.config.arm_controller} is {state}: moves will fail '
                                  "(-4); call robot.arm.recover() ('recover' in the console)")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def connect(config=None, ros_args=None, wait=True, wait_timeout=10.0, need_moveit=True):
    """Join the ROS network and return a Robot (use as a context manager to clean up).

    wait=True blocks until joint states, TF and MoveIt are visible, raising ConnectionError
    with a hint if not. need_moveit=False skips the MoveIt check (e.g. for a viewer).
    """
    return Robot(config, ros_args, wait, wait_timeout, need_moveit)
