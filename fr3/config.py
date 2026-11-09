"""All tunables for talking to the robot, in one place. Override fields per project."""
from dataclasses import dataclass, field

from .kinematics import READY_JOINTS


@dataclass
class RobotConfig:
    # ROS network: must match the control PC. If ROS_DOMAIN_ID is already set in the
    # environment, that value is used (with a warning if it differs from this one).
    domain_id: int = 42
    node_name: str = 'fr3_client'

    # MoveIt (names from the franka_ros2 / MoveIt config running on the control PC)
    group: str = 'fr3_arm'             # planning group from the SRDF
    ee_link: str = 'fr3_hand_tcp'      # link that move_to() puts at the target
    base_frame: str = 'fr3_link0'      # all positions are relative to this frame
    velocity_scaling: float = 0.1      # 0..1 of the robot's limits; start slow
    acceleration_scaling: float = 0.1
    planning_time: float = 5.0         # seconds MoveIt may spend planning one move

    # Optional Franka actions (need franka_msgs on this PC)
    gripper_ns: str = '/franka_gripper'
    error_recovery_action: str = '/action_server/error_recovery'

    # ros2_control on the control PC (needs controller_manager_msgs on this PC). After a reflex
    # the arm controller often ends up inactive; arm.recover() re-activates it.
    controller_manager: str = '/controller_manager'
    arm_controller: str = 'fr3_arm_controller'

    # Safe box for target positions (metres, base frame). Tighten for your setup.
    workspace: dict = field(default_factory=lambda: {
        'x': (0.30, 0.75), 'y': (-0.30, 0.30), 'z': (0.05, 0.70)})

    ready_joints: list = field(default_factory=lambda: [float(v) for v in READY_JOINTS])

    @property
    def joint_names(self):
        """fr3_joint1..fr3_joint7 (derived from base_frame's prefix)."""
        return [self.base_frame.replace('link0', f'joint{i}') for i in range(1, 8)]
