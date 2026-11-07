"""Plan a camera orbit for a Franka FR3 entirely in joint space, before the robot moves.

A camera held in the gripper circles an object while always looking at it. MoveIt's Cartesian
planner follows such a loop greedily from wherever the arm starts and often runs into a joint
limit (or flips the wrist) part-way round. Instead, this module:
  1. builds the circle of camera poses for several camera roll angles,
  2. solves inverse kinematics continuously around the loop from several start configurations,
  3. keeps the loop that stays furthest from the joint limits,
  4. turns it into a smooth, timed joint trajectory (gentle ramp up / constant speed / ramp down).

Needs numpy and scipy. No ROS imports, so it can be tested offline.
"""
import math

import numpy as np

# FR3 kinematics (Franka's modified DH: a, d, alpha per joint), flange and hand TCP.
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2),
       (0.0825, 0.0, math.pi / 2), (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2),
       (0.088, 0.0, math.pi / 2)]
_FLANGE_D = 0.107      # link7 -> link8 (flange)
_TCP_D = 0.1034        # flange -> fr3_hand_tcp (between the fingertips)
_TCP_YAW = -math.pi / 4

# FR3 joint position limits (rad), current Franka spec.
FR3_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
FR3_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
LIMIT_PAD = 0.05       # never plan closer than this to a limit (rad)
MIN_MARGIN = 0.10      # reject loops that come closer than this to a limit (rad)
GOOD_MARGIN = 0.15     # among loops at least this safe, prefer the one closest to the current arm pose
START_MOVE_MIN_Z = 0.25  # gripper and lens stay at least this high (m) while moving to the start
# The FR3 firmware stops with 'self_collision_avoidance_violation' when the elbow folds so far that
# the wrist comes near the upper arm (observed: stopped at 0.295 m; ready pose is 0.40 m).
MIN_WRIST_TO_UPPER_ARM = 0.35
MAX_JOINT_VEL = 0.5    # rad/s, well under the FR3 limits; slower lap if needed
MAX_JOINT_ACC = 1.0    # rad/s^2

READY_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

AXES = {'+x': (1, 0, 0), '-x': (-1, 0, 0), '+y': (0, 1, 0),
        '-y': (0, -1, 0), '+z': (0, 0, 1), '-z': (0, 0, -1)}


# ---------------- Small rotation helpers ----------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_between(a, b):
    """Smallest rotation matrix that turns direction a onto direction b."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:  # opposite directions: half turn about any perpendicular axis
        axis = _unit(np.cross(a, (1, 0, 0) if abs(a[0]) < 0.9 else (0, 1, 0)))
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rot_z(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_to_quat(R):
    """Rotation matrix -> (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


# ---------------- Kinematics ----------------

def _frames(q):
    """World transforms of joint frames 1..7, then the TCP."""
    T = np.eye(4)
    out = []
    for (a, d, alpha), th in zip(_DH, q):
        ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(th), math.sin(th)
        T = T @ np.array([[ct, -st, 0, a],
                          [st * ca, ct * ca, -sa, -d * sa],
                          [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
        out.append(T)
    tcp = np.eye(4)
    tcp[:3, :3] = rot_z(_TCP_YAW)
    tcp[2, 3] = _FLANGE_D + _TCP_D
    out.append(T @ tcp)
    return out


def fk(q):
    """TCP position and rotation for joint angles q."""
    T = _frames(q)[-1]
    return T[:3, 3].copy(), T[:3, :3].copy()


def _jacobian(frames):
    p = frames[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = frames[i][:3, 2], frames[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def ik(p_target, R_target, seed, reg=0.0, pos_tol=1e-3, ang_tol_deg=0.5):
    """Bounded least-squares IK near `seed`. Returns (q, ok)."""
    from scipy.optimize import least_squares
    lo, hi = FR3_LOWER + LIMIT_PAD, FR3_UPPER - LIMIT_PAD
    seed = np.clip(np.asarray(seed, dtype=float), lo + 1e-6, hi - 1e-6)
    w_rot = 0.3  # metres per radian: weight of orientation vs position error

    def residual(q):
        fr = _frames(q)
        p, R = fr[-1][:3, 3], fr[-1][:3, :3]
        E = R_target @ R.T
        rot_err = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        return np.concatenate([p_target - p, w_rot * rot_err, reg * (q - seed)])

    def jac(q):
        J = _jacobian(_frames(q))
        return np.vstack([-J[:3], -w_rot * J[3:], reg * np.eye(7)])

    res = least_squares(residual, seed, jac=jac, bounds=(lo, hi), xtol=1e-12, ftol=1e-12,
                        gtol=1e-12, max_nfev=200)
    p, R = fk(res.x)
    ang = math.degrees(math.acos(np.clip((np.trace(R_target @ R.T) - 1) / 2, -1, 1)))
    return res.x, bool(np.linalg.norm(p - p_target) < pos_tol and ang < ang_tol_deg)


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


def _clear_of_table(qs, min_z=0.08):
    """Elbow and wrist frames stay above the table for every configuration."""
    return all(min(T[2, 3] for T in _frames(q)[2:]) > min_z for q in qs)


def wrist_to_upper_arm(q):
    """Distance (m) from the wrist/flange to the shoulder-elbow segment (elbow fold check)."""
    fr = _frames(q)
    shoulder, elbow = fr[0][:3, 3], fr[2][:3, 3]
    flange = fr[6][:3, 3] + fr[6][:3, 2] * _FLANGE_D
    ab = elbow - shoulder
    best = np.inf
    for p in (fr[4][:3, 3], fr[6][:3, 3], flange):
        t = np.clip((p - shoulder) @ ab / (ab @ ab), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (shoulder + t * ab))))
    return best


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
    return float(np.min(np.minimum(np.asarray(q) - FR3_LOWER, FR3_UPPER - np.asarray(q))))


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


def _margin_all(qs):
    return float(np.min(np.minimum(qs - FR3_LOWER, FR3_UPPER - qs)))


# ---------------- Timing ----------------

def time_trajectory(qs, duration, dt=0.05):
    """Time the joint path: smooth ramp up, constant speed, smooth ramp down.

    Stretches `duration` if needed to respect MAX_JOINT_VEL / MAX_JOINT_ACC.
    Returns (duration, times, positions, velocities, accelerations).
    """
    from scipy.interpolate import CubicSpline
    u = np.linspace(0.0, 1.0, len(qs))
    spline = CubicSpline(u, qs, axis=0)
    d1, d2 = spline.derivative(1), spline.derivative(2)
    while True:
        ramp = min(3.0, duration / 4)
        vmax = 1.0 / (duration - ramp)  # path speed in u per second (cosine ramps)
        t = np.arange(0.0, duration + 1e-9, dt)
        if t[-1] < duration:
            t = np.append(t, duration)

        def speed(tt):
            if tt < ramp:
                return vmax * (1 - math.cos(math.pi * tt / ramp)) / 2
            if tt > duration - ramp:
                return vmax * (1 - math.cos(math.pi * (duration - tt) / ramp)) / 2
            return vmax

        def accel(tt):
            if tt < ramp:
                return vmax * math.pi / (2 * ramp) * math.sin(math.pi * tt / ramp)
            if tt > duration - ramp:
                return -vmax * math.pi / (2 * ramp) * math.sin(math.pi * (duration - tt) / ramp)
            return 0.0

        # Integrate path position u(t) finely, then sample.
        fine = np.linspace(0.0, duration, int(duration * 200) + 1)
        sp = np.array([speed(tt) for tt in fine])
        uf = np.concatenate([[0.0], np.cumsum((sp[1:] + sp[:-1]) / 2 * np.diff(fine))])
        uf /= uf[-1]
        us = np.interp(t, fine, uf)
        v = np.array([speed(tt) for tt in t])
        a = np.array([accel(tt) for tt in t])
        pos = spline(us)
        vel = d1(us) * v[:, None]
        acc = d2(us) * (v ** 2)[:, None] + d1(us) * a[:, None]
        vel[0] = vel[-1] = 0.0
        acc[-1] = 0.0
        scale = max(np.max(np.abs(vel)) / MAX_JOINT_VEL,
                    math.sqrt(np.max(np.abs(acc)) / MAX_JOINT_ACC))
        if scale <= 1.0:
            return duration, t, pos, vel, acc
        duration *= scale * 1.05
