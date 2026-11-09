#!/usr/bin/env python3
"""Preview an orbit in 3D before running it on the robot (never moves the robot).

Same arguments as the console's orbit command:

    python orbit_preview.py                                   # defaults from camera_orbit.py
    python orbit_preview.py 0.509 -0.005 0.29 0.24 0.11 60 0.18
    python orbit_preview.py 0.509 -0.005 0.055 0.18 0.20 --object 0.065 0.11
    python orbit_preview.py ... --robot                       # plan from the real arm's pose
    python orbit_preview.py ... --halves                      # two upright half-laps

Plans with orbit_planner exactly like 'orbit' does (from 'ready', or from the real arm's
current joints with --robot), then loops the lap in MuJoCo:
    orange line   the lens path (purple: 2nd half)   green line  lens -> aim point (centre)
    grey box      the phone                blue        the object (--object)
Press ']' in the viewer to look through the phone's camera (Esc: back to the free camera).

What you see during the lap is what the arm does: the planned joint trajectory is executed as
is (MoveIt doesn't re-plan it). Only the move from the current pose to the lap's start is
planned by MoveIt at run time, so it's not shown.
"""
import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

import camera_orbit
from fr3.config import RobotConfig
from fr3.kinematics import LOWER, READY_JOINTS, UPPER, fk
from fr3.trajectory import time_trajectory
from orbit_planner import AXES, horizon_tilt_deg, plan_orbit, plan_upright_halves
from sync_viewer import SCENE, joint_map   # read-only reuse: the scene path + joint lookup

# Phone drawn in the gripper (TCP frame, metres): matches camera_orbit's default mounting
# (held upright and upside down, gripped across its 7.2 cm width at the bottom end).
PHONE_CENTER = (0.0, 0.0, 0.06)
PHONE_HALF = (0.004, 0.036, 0.075)
FINGER_ON_PHONE = 0.036   # per-finger opening while holding it

ORANGE, GREEN, RED = [1, 0.5, 0, 1], [0.1, 0.9, 0.2, 1], [1, 0.1, 0.1, 1]
PURPLE = [0.6, 0.3, 1, 1]
GREY = [0.25, 0.25, 0.28, 1]


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('values', nargs='*', type=float,
                    help='cx cy cz [radius height [seconds [near_radius]]], like orbit')
    ap.add_argument('--camera-axis', default=camera_orbit.CAMERA_AXIS, choices=sorted(AXES))
    ap.add_argument('--camera-offset', nargs=3, type=float, default=camera_orbit.CAMERA_OFFSET,
                    metavar=('X', 'Y', 'Z'), help='lens position in the TCP frame (m)')
    ap.add_argument('--object', nargs=2, type=float, metavar=('DIAMETER', 'HEIGHT'),
                    help='draw a cylinder centred on the orbit centre (centre = mid-height)')
    ap.add_argument('--robot', action='store_true',
                    help="plan from the real arm's current joints instead of 'ready'")
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed (default 1 = real time)')
    ap.add_argument('--halves', action='store_true',
                    help='two overlapping half-laps with an upright picture instead of one lap')
    args = ap.parse_args()
    v = args.values
    if len(v) not in (0, 3, 5, 6, 7):
        ap.error('values: cx cy cz [radius height [seconds [near_radius]]]')
    args.center = tuple(v[0:3]) if v else camera_orbit.CENTER
    args.radius, args.height = v[3:5] if len(v) >= 5 else (camera_orbit.RADIUS, camera_orbit.HEIGHT)
    args.seconds = v[5] if len(v) >= 6 else camera_orbit.SECONDS
    near = v[6] if len(v) == 7 else camera_orbit.NEAR_RADIUS
    args.near = None if near is None or near >= args.radius else near   # as camera_orbit.plan
    return args


def start_joints(use_robot):
    if not use_robot:
        return READY_JOINTS.copy(), "'ready'"
    from fr3 import connect
    with connect(wait=False, need_moveit=False) as robot:
        end = time.time() + 5
        while robot.arm.joints() is None and time.time() < end:
            time.sleep(0.05)
        q = robot.arm.joints()
    if q is None:
        raise SystemExit('no joint states from the robot (is it on and the driver running?)')
    return np.array(q), 'the current arm pose'


def lens_of(q, offset, axis):
    p, R = fk(q)
    return p + R @ offset, R @ axis, p, R


def add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float),
                        np.asarray(pos, float), np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_line(scn, a, b, rgba, width=3):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width,
                         np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def build_preview_model(center, obj=None):
    """Robot scene (FR3 joint ranges) + optional object + a camera that follows the lens.

    Switch to the phone camera in the viewer with ']' (Esc goes back to the free camera)."""
    spec = mujoco.MjSpec.from_file(SCENE)
    for i in range(7):
        spec.joint(f'joint{i + 1}').range = [LOWER[i], UPPER[i]]
    if obj:
        d, h = obj
        spec.worldbody.add_geom(name='object', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[d / 2, h / 2, 0], pos=list(center),
                                rgba=[0.2, 0.5, 0.9, 0.8], contype=0, conaffinity=0)
    body = spec.worldbody.add_body(name='phone_cam', mocap=True)
    body.add_camera(name='phone', fovy=65)
    return spec.compile()


def camera_quat(view, up):
    """MuJoCo camera orientation: looks along -z, picture up = +y."""
    z = -view / np.linalg.norm(view)
    y = up - (up @ z) * z
    y /= np.linalg.norm(y)
    m = np.column_stack([np.cross(y, z), y, z])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, m.flatten())
    return quat


def plan_segments(args, q_start, offset):
    """[(name, times, positions, planned joints or None, colour)] played one after another."""
    if not args.halves:
        orbit = plan_orbit(args.center, args.radius, args.height, args.camera_axis, offset,
                           q_now=q_start, near_radius=args.near)
        duration, times, pos, _, _ = time_trajectory(orbit['q'], args.seconds)
        print(f'  closest joint limit {orbit["margin"]:.2f} rad, largest joint turn to the start '
              f'{math.degrees(orbit["start_move"]):.0f} deg, lap {duration:.0f} s')
        tilt = max(horizon_tilt_deg(fk(q)[1]) for q in orbit['q'])
        print(f'  the picture rotates during the lap (horizon tilts up to {tilt:.0f} deg)')
        return [('lap', times, pos, orbit['q'], ORANGE)], orbit['poses']

    res = plan_upright_halves(args.center, args.radius, args.height, args.camera_axis, offset,
                              q_now=q_start, near_radius=args.near)
    segs, poses = [], []
    for i, (arc, colour) in enumerate(zip(res['arcs'], (ORANGE, PURPLE))):
        if i == 1:   # MoveIt plans this move at run time: shown as a straight joint move
            qa, qb = res['arcs'][0]['q'][-1], arc['q'][0]
            _, times, pos, _, _ = time_trajectory(
                [qa + s * (qb - qa) for s in np.linspace(0, 1, 10)], 0.1)
            segs.append(('move back to the start of half 2 (MoveIt picks the real path)',
                         times, pos, None, None))
        duration, times, pos, _, _ = time_trajectory(arc['q'],
                                                     args.seconds * arc['span_deg'] / 360)
        tilt = max(horizon_tilt_deg(fk(q)[1]) for q in arc['q'])
        end = arc['start_deg'] + arc['span_deg']
        print(f'  half {i + 1}: {arc["start_deg"] % 360:.0f} -> {end % 360:.0f} deg around the '
              f'object, {duration:.0f} s, closest joint limit {arc["margin"]:.2f} rad, horizon '
              f'level within {tilt:.2f} deg')
        segs.append((f'half {i + 1}', times, pos, arc['q'], colour))
        poses += arc['poses']
    print(f'  largest joint turn to the start {math.degrees(res["start_move"]):.0f} deg'
          + ('' if res['transition_ok'] else
             '; WARNING: the move between the halves folds the elbow or dips the camera'))
    if res['flip']:
        print('  note: this solution holds the gripper pointing upward')
    return segs, poses


def main():
    args = parse()
    offset = np.asarray(args.camera_offset, float)
    axis = np.asarray(AXES[args.camera_axis], float)
    q_start, start_name = start_joints(args.robot)

    shape = (f'oval {args.near:.3f}-{args.radius:.3f} m' if args.near
             else f'radius {args.radius:.3f} m')
    print(f'Orbit around {args.center}: {shape}, {args.height:.3f} m above, '
          f'camera {args.camera_axis} ({", ".join(f"{v:g}" for v in offset)}), from {start_name}'
          + (', two upright half-laps' if args.halves else ''))
    print('Planning ...')
    try:
        segments, poses = plan_segments(args, q_start, offset)
    except ValueError as e:
        raise SystemExit(f'  {e}')

    c = np.asarray(args.center, float)
    paths = [(np.array([lens_of(q, offset, axis)[0] for q in qs]), col)
             for _, _, _, qs, col in segments if qs is not None]
    lenses = np.vstack([p for p, _ in paths])
    tcps = np.array([fk(q)[0] for _, _, pos, _, _ in segments for q in pos])
    look = [math.degrees(math.atan2(lz - c[2], math.hypot(lx - c[0], ly - c[1])))
            for lx, ly, lz in lenses]
    print(f'  lens height {lenses[:, 2].min():.3f}-{lenses[:, 2].max():.3f} m, looking down '
          f'{min(look):.0f}-{max(look):.0f} deg, gripper lowest z {tcps[:, 2].min():.3f} m')
    ws = RobotConfig().workspace
    outside = [p for p, _ in poses if not all(ws[k][0] <= v <= ws[k][1] for k, v in zip('xyz', p))]
    if outside:
        print(f'  WARNING: {len(outside)} gripper poses are outside the workspace box '
              f'{ws}: the console would refuse this')

    model = build_preview_model(c, args.object)
    data = mujoco.MjData(model)
    qadr = joint_map(model)
    arm = [qadr[f'joint{i}'] for i in range(1, 8)]
    for f in ('finger_joint1', 'finger_joint2'):
        data.qpos[qadr[f]] = FINGER_ON_PHONE

    # Picture "up" in the phone view: the TCP -z side (or +z), whichever starts out upward
    R0 = fk(segments[0][2][0])[1]
    up_sign = 1.0 if -R0[2, 2] >= 0 else -1.0

    PAUSE = 1.0
    ends = np.cumsum([seg[1][-1] + PAUSE for seg in segments])
    print("Playing in a loop. Press ']' in the viewer for the phone's view (Esc: back to the "
          'free camera). Close the window to quit.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = c
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 200, -25
        t0 = time.time()
        shown = None
        while viewer.is_running():
            t = ((time.time() - t0) * args.speed) % ends[-1]
            i = int(np.searchsorted(ends, t, side='right'))
            name, times, pos, _, _ = segments[i]
            if name != shown:
                print(f'  ... {name}')
                shown = name
            ts = min(t - (ends[i - 1] if i else 0.0), times[-1])
            q = np.array([np.interp(ts, times, pos[:, j]) for j in range(7)])
            data.qpos[arm] = q
            lens, view, tcp, R = lens_of(q, offset, axis)
            data.mocap_pos[0] = lens
            data.mocap_quat[0] = camera_quat(view, up_sign * -R[:, 2])
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)   # camera poses aren't part of mj_kinematics

            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                for path, col in paths:
                    for a, b in zip(path[:-1], path[1:]):
                        add_line(scn, a, b, col, 4)
                add_line(scn, lens, c, GREEN, 2)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008, 0, 0], c, np.eye(3), RED)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.007, 0, 0], lens, np.eye(3), GREEN)
                add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, PHONE_HALF,
                         tcp + R @ np.asarray(PHONE_CENTER), R, GREY)
            viewer.sync()
            time.sleep(0.016)


if __name__ == '__main__':
    main()
