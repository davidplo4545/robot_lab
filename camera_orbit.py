"""Camera orbit for 3D reconstruction: plan with orbit_planner, execute through fr3.

Project-specific (uses the generic fr3 API only).
"""
from fr3.trajectory import time_trajectory
from orbit_planner import AXES, plan_orbit, plan_upright_halves  # noqa: F401  (AXES: CLI)

# Defaults (metres / seconds)
CENTER = (0.509, -0.005, 0.1)   # object being filmed
RADIUS = 0.18                   # horizontal camera distance from the object (far side)
NEAR_RADIUS = 0.12              # ...on the robot's side: the path is an oval, since a wide
                                # circle there makes the elbow fold up (self-collision stop)
HEIGHT = 0.20                   # camera height above the object -> 48-59 deg look-down
SECONDS = 40.0                  # one full lap; slower = sharper frames

# Camera held in the gripper: which gripper (TCP) axis the lens looks along, and where the lens
# is in the TCP frame (metres). Default: iPhone 17 held vertically and upside down, gripped
# across its width at the bottom end, so the lens hangs ~12 cm beyond the fingertips (TCP +z),
# on the back face, looking away from the robot in the ready pose.
# (Flat phone with the lens facing down = '+z', but it folds the elbow.)
CAMERA_AXIS = '+x'
CAMERA_OFFSET = (0.004, 0.0, 0.12)


def plan(robot, center=CENTER, radius=RADIUS, height=HEIGHT, near_radius=NEAR_RADIUS,
         camera_axis=CAMERA_AXIS, camera_offset=CAMERA_OFFSET):
    """Plan the whole orbit in joint space from the arm's current pose. Raises ValueError if unsafe.

    near_radius < radius gives an oval that stays closer on the robot's side.
    """
    near_radius = None if near_radius is None or near_radius >= radius else near_radius
    result = plan_orbit(center, radius, height, camera_axis, camera_offset,
                        q_now=robot.arm.joints(), near_radius=near_radius)
    robot.arm.check_workspace([tuple(p) for p, _ in result['poses']])
    return result


def go_to_start(robot, orbit):
    """Collision-checked MoveIt move to the joint configuration where the loop starts."""
    return robot.arm.move_joints(orbit['q'][0].tolist())


def run(robot, orbit, duration=SECONDS):
    """Execute the pre-planned loop as one smooth joint trajectory (about `duration` s)."""
    q_now = robot.arm.joints()
    if q_now is not None and max(abs(a - b) for a, b in zip(q_now, orbit['q'][0])) > 0.05:
        raise RuntimeError('Arm is not at the orbit start; run the orbit command again')
    duration, times, pos, vel, acc = time_trajectory(orbit['q'], duration)
    print(f'  Lap time {duration:.0f} s')
    return robot.arm.execute_joint_trajectory(times, pos, vel, acc)


# ---------------- Upright picture: two half-laps ----------------

def plan_halves(robot, center=CENTER, radius=RADIUS, height=HEIGHT, near_radius=NEAR_RADIUS,
                camera_axis=CAMERA_AXIS, camera_offset=CAMERA_OFFSET):
    """Two overlapping ~190 deg arcs with a level horizon (see plan_upright_halves).

    Returns its dict: 'arcs' (each usable as an orbit for go_to_start/run), 'margin', ...
    """
    near_radius = None if near_radius is None or near_radius >= radius else near_radius
    result = plan_upright_halves(center, radius, height, camera_axis, camera_offset,
                                 q_now=robot.arm.joints(), near_radius=near_radius)
    robot.arm.check_workspace([tuple(p) for arc in result['arcs'] for p, _ in arc['poses']])
    return result


def run_halves(robot, halves, lap_seconds=SECONDS):
    """Half 1, MoveIt move back to half 2's start, half 2. The arm must be at half 1's start.

    Each half takes its share of lap_seconds (~53% each: they overlap)."""
    for i, arc in enumerate(halves['arcs']):
        if i == 1:
            print('  Moving to the start of half 2 ...')
            if not go_to_start(robot, arc):
                print('  Could not reach the start of half 2 (see log above).')
                return False
        print(f'  Half {i + 1} ...')
        if not run(robot, arc, lap_seconds * arc['span_deg'] / 360):
            return False
    return True
