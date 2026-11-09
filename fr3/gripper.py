"""Franka Hand gripper (franka_gripper node on the control PC). Widths in metres, force in newtons.

Needs franka_msgs on this PC; without it `available` is False and the arm still works.
"""
try:
    from franka_msgs.action import Grasp, Homing
    from franka_msgs.action import Move as GripperMove
except ImportError:
    Grasp = Homing = GripperMove = None


class Gripper:
    def __init__(self, conn, config):
        self.conn, self.log = conn, conn.log
        self.available = Grasp is not None
        ns = config.gripper_ns.rstrip('/')
        if self.available:
            self._grasp = conn.action_client(Grasp, f'{ns}/grasp')
            self._move = conn.action_client(GripperMove, f'{ns}/move')
            self._homing = conn.action_client(Homing, f'{ns}/homing')

    def _require(self):
        if not self.available:
            raise RuntimeError('franka_msgs is not available on this PC; build it and source '
                               'its install/setup.bash')

    def _ok(self, result, what):
        if result is None or not result.success:
            self.log.error(f'{what} failed: {getattr(result, "error", "")}')
            return False
        return True

    def home(self):
        """Calibrate the fingers (opens fully). Do this once after power-on."""
        self._require()
        return self._ok(self.conn.run_action(self._homing, Homing.Goal(), 'gripper homing'),
                        'Gripper homing')

    def open(self, width=0.08, speed=0.1):
        """Move the fingers to `width` metres apart (0.08 = fully open)."""
        self._require()
        goal = GripperMove.Goal()
        goal.width, goal.speed = float(width), float(speed)
        return self._ok(self.conn.run_action(self._move, goal, 'gripper move'), 'Gripper move')

    def grasp(self, width, force=20.0, speed=0.05, epsilon=0.01):
        """Close on an object about `width` metres wide with `force` newtons.

        Succeeds only if the fingers stop within `epsilon` of `width`, i.e. something was gripped.
        """
        self._require()
        goal = Grasp.Goal()
        goal.width, goal.force, goal.speed = float(width), float(force), float(speed)
        goal.epsilon.inner = goal.epsilon.outer = float(epsilon)
        result = self.conn.run_action(self._grasp, goal, 'gripper grasp')
        if result is None or not result.success:
            self.log.error('Grasp failed: nothing gripped, or the object width is more than '
                           f'{epsilon * 1000:.0f} mm off. {getattr(result, "error", "")}')
            return False
        return True
