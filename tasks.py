"""Small multi-step tasks built on the fr3 API: examples of project logic.

Each step returns True/False; a sequence stops at the first failure.
"""


def run_steps(steps):
    for label, step in steps:
        print(f'  {label} ...')
        if not step():
            print(f'  Stopped: "{label}" failed')
            return False
    return True


def pick(robot, x, y, z, width, force=20.0, approach=0.10):
    """Open, move above (x, y, z), lower straight down, grasp, lift straight up."""
    x, y, z = float(x), float(y), float(z)
    above = (x, y, z + approach)
    robot.arm.check_workspace([(x, y, z), above])
    return run_steps([
        ('Opening gripper', lambda: robot.gripper.open()),
        ('Moving above object', lambda: robot.arm.move_to(*above)),
        ('Lowering', lambda: robot.arm.move_linear([(x, y, z)])),
        ('Grasping', lambda: robot.gripper.grasp(width, force)),
        ('Lifting', lambda: robot.arm.move_linear([above])),
    ])


def place(robot, x, y, z, approach=0.10):
    """Move above (x, y, z), lower straight down, open, lift straight up."""
    x, y, z = float(x), float(y), float(z)
    above = (x, y, z + approach)
    robot.arm.check_workspace([(x, y, z), above])
    return run_steps([
        ('Moving above target', lambda: robot.arm.move_to(*above)),
        ('Lowering', lambda: robot.arm.move_linear([(x, y, z)])),
        ('Releasing', lambda: robot.gripper.open()),
        ('Lifting', lambda: robot.arm.move_linear([above])),
    ])


def triangle(robot, size=0.10):
    """Draw an equilateral triangle in the horizontal plane, starting and ending here."""
    pose = robot.arm.pose()
    if pose is None:
        raise RuntimeError('No TF yet; is the robot/MoveIt launch running?')
    x0, y0, z0 = pose[0]
    corners = [(x0 + size, y0, z0), (x0 + size / 2, y0 + size * 0.866, z0), (x0, y0, z0)]
    print(f'Drawing a {size:.2f} m triangle at z={z0:.3f} ...')
    return robot.arm.move_linear(corners)
