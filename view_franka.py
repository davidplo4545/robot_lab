import mujoco
import mujoco.viewer

# Load the scene XML, which includes the Panda arm, a floor, and lighting
model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")

# Launch the interactive viewer (this blocks until the window is closed)
mujoco.viewer.launch(model)

