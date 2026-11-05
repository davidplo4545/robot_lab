import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer

class MuJoCoSyncNode(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_sync_node')
        self.model = model
        self.data = data
        
        # Subscribe to the robot's live joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Cache the MuJoCo joint qpos indices for fast lookup
        self.joint_map = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_map[name] = self.model.jnt_qposadr[i]

    def joint_callback(self, msg):
        # Print the first message to see what ROS is sending
        self.get_logger().info(f'Received ROS joints: {msg.name[:3]}...', once=True)
        self.get_logger().info(f'MuJoCo expects: {list(self.joint_map.keys())[:3]}...', once=True)

        for name, pos in zip(msg.name, msg.position):
            # ADAPT THIS LINE based on what ros2 topic echo showed you
            mujoco_name = name.replace('fr3_', '') 
            
            if mujoco_name in self.joint_map:
                idx = self.joint_map[mujoco_name]
                self.data.qpos[idx] = pos
            else:
                # This will warn you if a joint name is being ignored
                # self.get_logger().warn(f'Ignored joint: {name} (tried: {mujoco_name})')
                pass

def main(args=None):
    rclpy.init(args=args)
    
    # 1. Load the model and data
    model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    
    # 2. Create the ROS 2 node to listen for joint updates
    node = MuJoCoSyncNode(model, data)
    
    # 3. Launch the non-blocking "passive" viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and rclpy.ok():
            # Process ROS callbacks to update data.qpos
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # Compute forward kinematics based on the new joint positions
            mujoco.mj_kinematics(model, data)
            
            # Sync the visualizer to the updated data
            viewer.sync()
            
            # Render at roughly 60 FPS
            time.sleep(0.016)
            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
