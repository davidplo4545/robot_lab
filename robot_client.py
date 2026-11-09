#!/usr/bin/env python3
"""Interactive console for the Franka FR3: moves, gripper, pick/place and the camera orbit.

All robot communication goes through the reusable fr3 package; this file is only the CLI.

Usage:
  python3 robot_client.py
  python3 robot_client.py --vel 0.2 --camera-axis +x --camera-offset 0.004 0 0.12
"""
import argparse
import math

import camera_orbit
import tasks
from fr3 import RobotConfig, connect

HELP = f'''Commands (metres):
  x y z                 move to a point
  p                     print current pose
  ready                 go to the safe ready pose (joint space)
  recover               clear a robot error/reflex (after a red light or error -4)
  t [size]              draw a triangle (default 0.10)
  home                  calibrate the gripper (once after power-on)
  open [width]          open the gripper (default 0.08 = fully)
  grasp width [force]   close on an object of that width (default 20 N)
  pick x y z width      open, move above, lower, grasp, lift
  place x y z           move above, lower, release, lift
  orbit [cx cy cz [r h [sec [near]]]]  move the camera around an object, aimed at it
                        (default centre {camera_orbit.CENTER}, r={camera_orbit.RADIUS}, \
h={camera_orbit.HEIGHT}, {camera_orbit.SECONDS:.0f} s,
                        near={camera_orbit.NEAR_RADIUS} = radius on the robot side; near=r for a circle)
  cam [axis [ox oy oz]] show/set the gripper axis the lens faces (+x,-x,+y,-y,+z,-z)
                        and the lens offset from the gripper TCP (metres)
  q                     quit'''


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--vel', type=float, default=0.1, help='velocity scaling 0..1 (default 0.1)')
    ap.add_argument('--acc', type=float, default=0.1, help='acceleration scaling 0..1')
    ap.add_argument('--domain-id', type=int, default=42, help='ROS_DOMAIN_ID (default 42)')
    ap.add_argument('--camera-axis', default=camera_orbit.CAMERA_AXIS,
                    choices=list(camera_orbit.AXES))
    ap.add_argument('--camera-offset', type=float, nargs=3, default=camera_orbit.CAMERA_OFFSET,
                    metavar=('OX', 'OY', 'OZ'))
    ap.add_argument('--no-wait', action='store_true',
                    help="start even if the robot/MoveIt isn't visible yet")
    args, ros_args = ap.parse_known_args()
    return args, ros_args


class Console:
    def __init__(self, robot, camera_axis, camera_offset):
        self.robot = robot
        self.camera_axis = camera_axis
        self.camera_offset = list(camera_offset)

    def report(self, ok, what='Move'):
        print('Done.' if ok else f'{what} failed (see log above).')

    # ---- commands: each takes the list of words after the command name ----

    def cmd_ready(self, _):
        print('Moving to ready pose ...')
        self.report(self.robot.arm.ready())

    def cmd_recover(self, _):
        print('Recovering from error ...')
        self.report(self.robot.arm.recover(), 'Recovery')

    def cmd_p(self, _):
        pose = self.robot.arm.pose()
        if pose is None:
            print('No TF yet; is the robot/MoveIt launch running?')
            return
        (x, y, z), (qx, qy, qz, qw) = pose
        print(f'Position: ({x:.3f}, {y:.3f}, {z:.3f})  '
              f'Orientation xyzw: ({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})')

    def cmd_t(self, words):
        try:
            size = float(words[0]) if words else 0.10
        except ValueError:
            print("Usage: 't' or 't 0.08' (side length in metres)")
            return
        self.report(tasks.triangle(self.robot, size), 'Triangle')

    def cmd_home(self, _):
        self.report(self.robot.gripper.home(), 'home')

    def cmd_open(self, words):
        self.report(self.robot.gripper.open(*[float(w) for w in words[:1]]), 'open')

    def cmd_grasp(self, words):
        vals = [float(w) for w in words]
        if not 1 <= len(vals) <= 2:
            raise ValueError('usage: grasp width [force]')
        self.report(self.robot.gripper.grasp(*vals), 'grasp')

    def cmd_pick(self, words):
        vals = [float(w) for w in words]
        if len(vals) != 4:
            raise ValueError('usage: pick x y z width')
        self.report(tasks.pick(self.robot, *vals), 'pick')

    def cmd_place(self, words):
        vals = [float(w) for w in words]
        if len(vals) != 3:
            raise ValueError('usage: place x y z')
        self.report(tasks.place(self.robot, *vals), 'place')

    def cmd_cam(self, words):
        if len(words) not in (0, 1, 4) or (words and words[0] not in camera_orbit.AXES):
            raise ValueError(f'usage: cam [axis [ox oy oz]], axis one of {list(camera_orbit.AXES)}')
        if words:
            self.camera_axis = words[0]
        if len(words) == 4:
            self.camera_offset = [float(v) for v in words[1:4]]
        print(f'Camera: lens looks along {self.camera_axis}, offset {self.camera_offset}')

    def cmd_orbit(self, words):
        vals = [float(w) for w in words]
        if len(vals) not in (0, 3, 5, 6, 7):
            raise ValueError('usage: orbit [cx cy cz [radius height [seconds [near_radius]]]]')
        center = tuple(vals[0:3]) if vals else camera_orbit.CENTER
        radius, height = vals[3:5] if len(vals) >= 5 else (camera_orbit.RADIUS, camera_orbit.HEIGHT)
        seconds = vals[5] if len(vals) >= 6 else camera_orbit.SECONDS
        near = min(vals[6] if len(vals) == 7 else camera_orbit.NEAR_RADIUS, radius)
        shape = (f'radius {radius:.3f} m' if near >= radius else
                 f'oval {near:.3f} m (robot side) to {radius:.3f} m')
        tilt = (f'{math.degrees(math.atan2(height, near)):.0f} deg' if near >= radius
                else f'{math.degrees(math.atan2(height, radius)):.0f}-'
                     f'{math.degrees(math.atan2(height, near)):.0f} deg')
        print(f'Orbit around {center}: {shape}, {height:.3f} m above, '
              f'looking down {tilt}, camera {self.camera_axis}')
        print('  Planning (checking camera rolls and arm configurations) ...')
        orbit = camera_orbit.plan(self.robot, center, radius, height, near,
                                  self.camera_axis, self.camera_offset)
        print(f'  Found a safe loop: closest joint limit {orbit["margin"]:.2f} rad, '
              f'largest joint turn to reach the start {math.degrees(orbit["start_move"]):.0f} deg')
        print('  Moving to start pose ...')
        if not camera_orbit.go_to_start(self.robot, orbit):
            print('  Could not reach the start pose (see log above).')
            return
        answer = input('  At start. Check the camera points at the object, start '
                       'recording, then press Enter to orbit (q to cancel): ')
        if answer.strip().lower() == 'q':
            print('  Orbit cancelled.')
            return
        print('  Orbiting ...')
        ok = camera_orbit.run(self.robot, orbit, seconds)
        print('Done.' if ok else 'Orbit failed (see log above). Try a larger height, '
              'a smaller radius, or check the camera axis.')

    def move_xyz(self, words):
        try:
            x, y, z = (float(v) for v in words)
        except ValueError:
            print("Enter three numbers, e.g. '0.5 0.0 0.4'")
            return
        print(f'Planning to ({x:.3f}, {y:.3f}, {z:.3f}) ...')
        self.report(self.robot.arm.move_to(x, y, z))

    def handle(self, line):
        words = line.split()
        if not words:
            return
        command = getattr(self, f'cmd_{words[0]}', None)
        try:
            if command is None:
                self.move_xyz(words)
            else:
                command(words[1:])
        except (ValueError, RuntimeError) as e:
            print(f'Rejected: {e}')


def main():
    args, ros_args = parse_args()
    config = RobotConfig(domain_id=args.domain_id, velocity_scaling=args.vel,
                         acceleration_scaling=args.acc)
    print(f'Connecting (ROS_DOMAIN_ID {args.domain_id} unless set in the environment) ...')
    try:
        robot = connect(config, ros_args=ros_args or None, wait=not args.no_wait)
    except ConnectionError as e:
        print(f'Cannot reach the robot: {e}')
        return
    with robot:
        console = Console(robot, args.camera_axis, args.camera_offset)
        print(f'Group: {config.group}, end-effector: {config.ee_link}, frame: {config.base_frame}')
        print(f'Safe workspace: {config.workspace}')
        print(HELP)
        console.cmd_cam([])
        if not robot.gripper.available:
            print('Gripper commands unavailable: franka_msgs not found (source its workspace).')
        print('NOTE: the planner does not see the object, the table or the phone; first runs '
              'slow, hand on the user stop.')
        try:
            while True:
                line = input('> ').strip().lower()
                if line == 'q':
                    break
                console.handle(line)
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == '__main__':
    main()


# pick 0.509 -0.005 0.1 0.06
# place 0.509 -0.005 0.1
