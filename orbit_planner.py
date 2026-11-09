"""Plan a camera orbit for a Franka FR3 entirely in joint space, before the robot moves.

A camera held in the gripper circles an object while always looking at it. MoveIt's Cartesian
planner follows such a loop greedily from wherever the arm starts and often runs into a joint
limit (or flips the wrist) part-way round. Instead, this module:
  1. builds the circle of camera poses for several camera roll angles,
  2. solves inverse kinematics continuously around the loop from several start configurations,
  3. keeps the loop that stays furthest from the joint limits,
  4. turns it into a smooth, timed joint trajectory (fr3.trajectory.time_trajectory).

Project-specific. The robot model (FK/IK/limits) lives in the reusable fr3 package.
Needs numpy and scipy. No ROS imports, so it can be tested offline.
"""
import math

import numpy as np

from fr3.kinematics import (LIMIT_PAD, LOWER as FR3_LOWER, MIN_WRIST_TO_UPPER_ARM,
                            READY_JOINTS, UPPER as FR3_UPPER, clear_of_table as _clear_of_table,
                            fk, ik, joint_margin, rot_between, rot_z, wrist_to_upper_arm)

MIN_MARGIN = 0.10      # reject loops that come closer than this to a joint limit (rad)
GOOD_MARGIN = 0.15     # among loops at least this safe, prefer the one closest to the current arm pose
START_MOVE_MIN_Z = 0.25  # gripper and lens stay at least this high (m) while moving to the start

AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0),
        '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}


# ---------------- Orbit geometry ----------------

def orbit_radius(theta, center, radius, near_radius=None):
    """Horizontal camera distance at angle theta (rad) around the object.

    With near_radius set, the path is an oval: near_radius on the side facing the robot base
    (where a wide circle would make the arm fold up), widening smoothly to `radius` elsewhere
    (75% of the way already at the sides, full width on the far side).
    """
    if near_radius is None:
        return radius
    toward_base = math.atan2(-center[1], -center[0])
    s = 1 - ((1 + math.cos(theta - toward_base)) / 2) ** 2
    return near_radius + (radius - near_radius) * s


def orbit_poses(center, radius, height, camera_axis='+x', camera_offset=(0, 0, 0), roll=0.0,
                n=120, start_deg=180.0, span_deg=360.0, near_radius=None):
    """TCP poses (position, 3x3 rotation) that move a camera around `center`, looking at it.

    radius:        horizontal camera distance (far side, if near_radius is given).
    near_radius:   optional smaller distance on the robot's side -> oval path.
    camera_axis:   TCP axis the lens looks along, e.g. '+x'.
    camera_offset: lens position in the TCP frame (metres).
    roll:          camera roll about its own axis (rad): one value, or one per pose (n+1).
    The camera tilts toward the object but never spins about the vertical, so the wrist only
    wobbles (a full 360 deg spin is impossible for an arm with joint limits). The video picture
    therefore rotates (once per lap, plus any roll changes), which 3D reconstruction handles fine.
    Starts at start_deg (180 = robot's side of the object) and goes counter-clockwise.
    """
    c = np.asarray(center, dtype=float)
    offset = np.asarray(camera_offset, dtype=float)
    down = np.array([0.0, 0.0, -1.0])
    to_down = rot_between(AXES[camera_axis], down)
    rolls = np.broadcast_to(np.asarray(roll, dtype=float), (n + 1,))
    poses = []
    for k in range(n + 1):
        th = math.radians(start_deg + span_deg * k / n)
        rad = orbit_radius(th, c, radius, near_radius)
        lens = c + np.array([rad * math.cos(th), rad * math.sin(th), height])
        base = rot_z(rolls[k]) @ to_down                 # camera looking straight down, rolled
        R = rot_between(down, c - lens) @ base           # tilt to aim at the object
        poses.append((lens - R @ offset, R))
    return poses


def _not_folded(qs):
    return all(wrist_to_upper_arm(q) >= MIN_WRIST_TO_UPPER_ARM for q in qs)


def _start_move_ok(here, q0, samples=30):
    return _not_folded([here + s * (q0 - here) for s in np.linspace(0.0, 1.0, samples)])


def _start_move_low_point(here, q0, camera_offset, samples=40):
    """Lowest height of the gripper TCP and the lens on a straight joint move here -> q0."""
    offset = np.asarray(camera_offset, dtype=float)
    low = np.inf
    for s in np.linspace(0.0, 1.0, samples):
        p, R = fk(here + s * (q0 - here))
        low = min(low, p[2], (p + R @ offset)[2])
    return low


def _margin(q):
    return joint_margin(q)


def _margin_all(qs):
    return joint_margin(qs)


def _config_ok(q):
    return (_margin(q) >= MIN_MARGIN and wrist_to_upper_arm(q) >= MIN_WRIST_TO_UPPER_ARM
            and _clear_of_table([q]))


class _Node:
    __slots__ = ('q', 'j', 'value', 'start_move', 'parent')

    def __init__(self, q, j, value, start_move, parent):
        self.q, self.j, self.value, self.start_move, self.parent = q, j, value, start_move, parent


def _prune(nodes, cap):
    """Keep up to `cap` distinct nodes: safest first, then shortest move to the start."""
    nodes = sorted(nodes, key=lambda nd: (-min(nd.value, GOOD_MARGIN + 0.1), nd.start_move))
    kept = []
    for nd in nodes:
        if all(nd.j != k.j or np.max(np.abs(nd.q - k.q)) > 0.05 for k in kept):
            kept.append(nd)
            if len(kept) >= cap:
                break
    return kept


def plan_orbit(center, radius, height, camera_axis='+x', camera_offset=(0, 0, 0),
               q_now=None, n_rolls=12, seeds_per_roll=10, n=144, rng_seed=0,
               near_radius=None, start_angles=(180.0, 270.0, 0.0, 90.0), beam=20,
               n_coarse=48, roll_reach=1):
    """Find a safe continuous joint-space lap that starts close to the current arm pose.

    Search: the camera roll may change gradually along the path (it does not matter for the
    video), which gives the arm much more freedom. For each start angle, arm configurations are
    found at the start for every roll, then propagated step by step around the path (same or
    neighbouring roll), keeping the best `beam` branches by their worst joint-limit margin.
    Every configuration keeps the elbow unfolded (FR3 self-collision stop) and clears the table;
    the move to the start keeps the camera high and the elbow unfolded too.
    Among laps with at least GOOD_MARGIN, picks the shortest move to the start.
    Returns a dict with 'q' (n+1 x 7), 'poses', 'roll' (per pose), 'margin', 'start_move' (rad).
    """
    rng = np.random.default_rng(rng_seed)
    lo, hi = FR3_LOWER + LIMIT_PAD, FR3_UPPER - LIMIT_PAD
    here = np.asarray(q_now if q_now is not None else READY_JOINTS, dtype=float)
    rolls = np.linspace(0, 2 * math.pi, n_rolls, endpoint=False)
    min_start_z = min(START_MOVE_MIN_Z, center[2] + height - 0.05)
    args = (center, radius, height, camera_axis, camera_offset)
    if n % n_coarse:
        n = n_coarse * max(1, round(n / n_coarse))

    def pose(start_deg, k, j):
        return orbit_poses(*args, rolls[j], n=1, start_deg=start_deg + 360.0 * k / n_coarse,
                           span_deg=0.0, near_radius=near_radius)[0]

    finished = []  # (value, start_move, start_deg, [(q, j), ...])
    for start_deg in start_angles:
        layer = []
        for j in range(n_rolls):
            p, R = pose(start_deg, 0, j)
            seeds = [here, READY_JOINTS] + [rng.uniform(lo, hi) for _ in range(seeds_per_roll)]
            for seed in seeds:
                q, ok = ik(p, R, seed)
                if not ok or not _config_ok(q) or not _start_move_ok(here, q):
                    continue
                if _start_move_low_point(here, q, camera_offset) < min_start_z:
                    continue
                layer.append(_Node(q, j, _margin(q), float(np.max(np.abs(q - here))), None))
        layer = _prune(layer, beam)
        for k in range(1, n_coarse + 1):
            nxt = []
            for nd in layer:
                for dj in range(-roll_reach, roll_reach + 1):
                    j = (nd.j + dj) % n_rolls
                    q, ok = ik(*pose(start_deg, k, j), nd.q, reg=0.02)
                    if ok and np.max(np.abs(q - nd.q)) <= 0.3 and _config_ok(q):
                        nxt.append(_Node(q, j, min(nd.value, _margin(q)), nd.start_move, nd))
            if len(_prune(nxt, beam)) < beam // 2:
                # Branches are dying (heading into joint limits): solve this point afresh and
                # attach each new configuration to the best branch it can reach smoothly.
                for j in range(n_rolls):
                    for _ in range(2):
                        q, ok = ik(*pose(start_deg, k, j), rng.uniform(lo, hi))
                        if not ok or not _config_ok(q):
                            continue
                        parents = [nd for nd in layer
                                   if min((nd.j - j) % n_rolls, (j - nd.j) % n_rolls) <= roll_reach
                                   and np.max(np.abs(q - nd.q)) <= 0.3]
                        if parents:
                            nd = max(parents, key=lambda p: p.value)
                            nxt.append(_Node(q, j, min(nd.value, _margin(q)), nd.start_move, nd))
            layer = _prune(nxt, beam)
            if not layer:
                break
        for nd in layer:
            path = []
            while nd is not None:
                path.append((nd.q, nd.j))
                nd = nd.parent
            path.reverse()
            finished.append((min(_margin(q) for q, _ in path),
                             float(np.max(np.abs(path[0][0] - here))), start_deg, path))
        if any(f[0] >= GOOD_MARGIN for f in finished):
            break  # good enough; skip the remaining start angles to save planning time
    if not finished:
        raise ValueError('no safe lap found for this orbit: every option runs into a joint limit, '
                         'folds the elbow too far (the robot would stop with a self-collision '
                         "error) or swings the camera too low; try a smaller radius, a larger "
                         "height, or move to 'ready' first")

    good = [f for f in finished if f[0] >= GOOD_MARGIN]
    ranked = sorted(good, key=lambda f: f[1]) if good else sorted(finished, key=lambda f: -f[0])

    # Refine the best coarse laps on a fine grid (roll interpolated) for a smooth trajectory.
    steps = n // n_coarse
    for _, _, start_deg, path in ranked[:5]:
        # Unwrapped roll at each coarse point (so interpolation never jumps by 2 pi).
        coarse_roll = [rolls[path[0][1]]]
        for c in range(1, n_coarse + 1):
            d = (rolls[path[c][1]] - rolls[path[c - 1][1]] + math.pi) % (2 * math.pi) - math.pi
            coarse_roll.append(coarse_roll[-1] + d)
        roll_seq, seed_seq = [], []
        for c in range(n_coarse):
            qa, qb = path[c][0], path[c + 1][0]
            for s in range(steps):
                f = s / steps
                roll_seq.append(coarse_roll[c] + f * (coarse_roll[c + 1] - coarse_roll[c]))
                seed_seq.append(qa + f * (qb - qa))
        roll_seq.append(coarse_roll[-1])
        seed_seq.append(path[-1][0])
        n_fine = len(roll_seq) - 1
        poses = orbit_poses(*args, np.array(roll_seq), n=n_fine, start_deg=start_deg,
                            near_radius=near_radius)
        qs = [ik(*poses[0], path[0][0])[0]]
        ok = True
        for i in range(1, n_fine + 1):
            q, good_ik = ik(*poses[i], 0.5 * (qs[-1] + seed_seq[i]), reg=0.02)
            if not good_ik or np.max(np.abs(q - qs[-1])) > 0.15 or not _config_ok(q):
                ok = False
                break
            qs.append(q)
        if not ok:
            continue
        qs = np.array(qs)
        if (_start_move_ok(here, qs[0])
                and _start_move_low_point(here, qs[0], camera_offset) >= min_start_z):
            return {'q': qs, 'poses': poses, 'roll': np.array(roll_seq), 'margin': _margin_all(qs),
                    'start_move': float(np.max(np.abs(qs[0] - here)))}
    raise ValueError('orbit refinement failed; try a smaller radius or a larger height')
