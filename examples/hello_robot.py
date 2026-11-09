#!/usr/bin/env python3
"""Smallest fr3 program: connect, go to ready, nudge up 5 cm and back, print the pose.

Run from the folder that contains fr3/:  python3 examples/hello_robot.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from fr3 import connect  # noqa: E402
from tasks import pick, place, triangle  # noqa: E402
from camera_orbit import plan, go_to_start, run

with connect() as robot:
    # print('Joints:', [round(q, 3) for q in robot.arm.joints()])
    # print('Going to ready ...', 'ok' if robot.arm.ready() else 'FAILED')
    # (x, y, z), orientation = robot.arm.pose()
    # print(f'Pose: ({x:.3f}, {y:.3f}, {z:.3f})')
    # print('Up 5 cm ...', 'ok' if robot.arm.move_linear([(x, y, z + 0.05)]) else 'FAILED')
    # print('Back ...', 'ok' if robot.arm.move_linear([(x, y, z)]) else 'FAILED')
    # print('Pose:', robot.arm.pose()[0])
    # robot.arm.recover(restart_controller=True)
    robot.arm.ready(open_gripper=True)
    
    # pick(robot, 0.509, -0.005, 0.08, 0.06)
    # place(robot, 0.509, -0.005, 0.08)
    # robot.arm.ready()
    # camera_orbit = plan(robot)
    # go_to_start(robot, camera_orbit)
    # run(robot, camera_orbit)
    
    # pick 0.509 -0.005 0.1 0.06
    # place 0.509 -0.005 0.1
