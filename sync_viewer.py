"""Mirror the real robot's joint angles in a MuJoCo viewer (read-only, never moves the robot).

    python sync_viewer.py                                    # robot only
    python sync_viewer.py --cylinder 0.5 0 0 0.065 0.11      # + a see-through marker
    python sync_viewer.py --rigid 0.5 0 0 0.065 0.11         # + a solid cylinder the sim arm
                                                             #   can push and pick up

Positions are metres in the robot base frame (same as move_to/orbit); Z_BOTTOM is where the
cylinder stands (0 = table). With --rigid the viewer runs physics: the sim arm follows the
real joint angles through its motors, and when the real gripper closes on the cylinder the
sim fingers squeeze it, so a real pick lifts it in the viewer too. The sim cylinder only
matches the real one if you give its real position.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from fr3 import connect
from fr3.kinematics import LOWER, UPPER

SCENE = "mujoco_menagerie/franka_emika_panda/scene.xml"

# Shown every run (same format as the flags): (x, y, z_bottom, diameter, height)
CYLINDERS = []   # see-through markers
RIGID = []       # solid, liftable

FINGER_OPEN = 0.04       # per-finger travel (m)
GRIP_CTRL_MAX = 255.0    # menagerie gripper actuator: 0 = closed, 255 = open
SQUEEZE = 0.003          # a closing sim finger goes this much past the real one (m)


def build_model(markers, rigid, mass, friction):
    spec = mujoco.MjSpec.from_file(SCENE)
    # The menagerie model is a Panda: give joints 4 and 6 (and the motors) the FR3's ranges
    for i in range(7):
        spec.joint(f'joint{i + 1}').range = [LOWER[i], UPPER[i]]
        spec.actuator(f'actuator{i + 1}').ctrlrange = [LOWER[i], UPPER[i]]
    if rigid:
        for body in spec.bodies:      # the real arm doesn't sag: neither should the sim one
            if body.name != 'world':
                body.gravcomp = 1

    for i, (x, y, z, d, h) in enumerate(markers):
        spec.worldbody.add_geom(
            name=f'marker{i}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[d / 2, h / 2, 0], pos=[x, y, z + h / 2],
            rgba=[0.9, 0.5, 0.1, 0.4], contype=0, conaffinity=0)
        print(f'marker {i}: ({x:.3f}, {y:.3f}), z {z:.3f}..{z + h:.3f}, diameter {d:.3f}')

    for i, (x, y, z, d, h) in enumerate(rigid):
        body = spec.worldbody.add_body(name=f'rigid{i}', pos=[x, y, z + h / 2])
        body.add_freejoint()
        body.add_geom(
            name=f'rigid{i}', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[d / 2, h / 2, 0],
            mass=mass, friction=[friction, 0.01, 0.0001], condim=4,   # 4: no spinning in grip
            rgba=[0.2, 0.6, 0.9, 1])
        print(f'rigid {i}: ({x:.3f}, {y:.3f}), z {z:.3f}..{z + h:.3f}, diameter {d:.3f}, '
              f'{mass * 1000:.0f} g')
    return spec.compile()


def joint_map(model):
    """MuJoCo joint name -> qpos index."""
    out = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            out[name] = model.jnt_qposadr[i]
    return out


def real_to_sim(joints):
    """{'fr3_joint1': q, 'fr3_finger_joint1': w, ...} -> {'joint1': q, 'finger_joint1': w}"""
    return {name.replace('fr3_', ''): pos for name, pos in joints.items()}


class Gripper:
    """Drive the sim fingers from the real finger width.

    Copying the real width exactly would leave zero squeeze on the object (the real fingers
    stop at its surface, and so would the targets), so it would slide out. While the sim
    fingers touch a rigid object and the real gripper isn't opening, command closed instead;
    release once the real fingers open by 2 mm.
    """

    def __init__(self, model):
        self.fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                        for n in ('left_finger', 'right_finger')}
        self.rigid_bodies = {b for b in range(model.nbody)
                             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '')
                             .startswith('rigid')}
        self.holding = False
        self.hold_width = None
        self.armed = True          # may grab: False after a release until the real gripper closes
        self.open_peak = 0.0       # widest real opening since the last release

    def touching(self, model, data):
        for c in data.contact[:data.ncon]:
            pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if pair & self.fingers and pair & self.rigid_bodies:
                return True
        return False

    def ctrl(self, model, data, width):
        """width: real per-finger opening (m) -> actuator ctrl."""
        if self.holding:
            self.hold_width = min(self.hold_width, width)
            if width > self.hold_width + 0.002:          # real gripper opened: let go, and
                self.holding, self.armed = False, False  # don't re-grab while it keeps opening
                self.open_peak = width
                print(f'sim gripper: released (real width {2 * width * 100:.1f} cm)')
        else:
            if not self.armed:
                self.open_peak = max(self.open_peak, width)
                if width < self.open_peak - 0.002:       # real gripper closing again
                    self.armed = True
            if (self.armed and width < FINGER_OPEN - 0.001
                    and self.touching(model, data)):
                self.holding, self.hold_width = True, width   # real gripper closing on it
                print(f'sim gripper: holding (real width {2 * width * 100:.1f} cm)')
        if self.holding:
            return 0.0
        if width < FINGER_OPEN - 0.001:
            width -= SQUEEZE   # reach the surface: the sim pads stop just short at the real width
        return float(np.clip(width / FINGER_OPEN, 0, 1) * GRIP_CTRL_MAX)


def run_kinematic(model, data, robot, qadr):
    """No rigid objects: just copy the joint angles (exact, no physics)."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for name, pos in real_to_sim(robot.conn.joint_dict()).items():
                if name in qadr:
                    data.qpos[qadr[name]] = pos
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            time.sleep(0.016)  # ~60 FPS


def run_physics(model, data, robot, qadr):
    """Rigid objects: the arm tracks the real joints through its position motors."""
    arm = [f'joint{i}' for i in range(1, 8)]
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'actuator{i}')
           for i in range(1, 9)]
    gripper = Gripper(model)

    # Start the sim where the real robot is, so the arm doesn't swing through the objects
    print('waiting for the real joint angles ...')
    t0 = time.time()
    joints = {}
    while time.time() - t0 < 5 and not all(j in joints for j in arm):
        joints = real_to_sim(robot.conn.joint_dict())
        time.sleep(0.05)
    if all(j in joints for j in arm):
        for name, pos in joints.items():
            if name in qadr:
                data.qpos[qadr[name]] = pos
        for i, j in enumerate(arm):
            data.ctrl[act[i]] = joints[j]
    else:
        print('no joint states yet: starting from the home pose')
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if 'finger_joint1' not in joints:
        print('note: no finger joints in /joint_states, the sim gripper stays open')
    data.ctrl[act[7]] = GRIP_CTRL_MAX
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.time()
        while viewer.is_running():
            joints = real_to_sim(robot.conn.joint_dict())
            for i, j in enumerate(arm):
                if j in joints:
                    data.ctrl[act[i]] = joints[j]
            if 'finger_joint1' in joints:
                data.ctrl[act[7]] = gripper.ctrl(model, data, joints['finger_joint1'])

            now = time.time()   # step physics to keep up with the wall clock
            steps = min(int((now - last) / model.opt.timestep), 50)
            for _ in range(steps):
                mujoco.mj_step(model, data)
            last += steps * model.opt.timestep
            if now - last > 0.1:
                last = now      # fell behind (e.g. window dragged): don't try to catch up
            viewer.sync()
            time.sleep(0.005)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    meta = ('X', 'Y', 'Z_BOTTOM', 'DIAMETER', 'HEIGHT')
    ap.add_argument('--cylinder', nargs=5, type=float, action='append', default=[],
                    metavar=meta, help='see-through marker cylinder; repeat for more')
    ap.add_argument('--rigid', nargs=5, type=float, action='append', default=[],
                    metavar=meta, help='solid cylinder the sim arm can lift; repeat for more')
    ap.add_argument('--mass', type=float, default=0.1, help='rigid cylinder mass, kg (0.1)')
    ap.add_argument('--friction', type=float, default=1.5,
                    help='rigid cylinder friction (1.5; lower = slips out more easily)')
    args = ap.parse_args()

    rigid = RIGID + args.rigid
    model = build_model(CYLINDERS + args.cylinder, rigid, args.mass, args.friction)
    data = mujoco.MjData(model)
    qadr = joint_map(model)

    # Joint states only: no MoveIt needed. Sets ROS_DOMAIN_ID if the terminal didn't.
    with connect(wait=False, need_moveit=False) as robot:
        (run_physics if rigid else run_kinematic)(model, data, robot, qadr)


if __name__ == '__main__':
    main()
