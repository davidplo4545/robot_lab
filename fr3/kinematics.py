"""Franka FR3 kinematics: forward kinematics, Jacobian, inverse kinematics, joint limits.

Pure numpy/scipy (no ROS), so it works offline for planning and testing. Verified against the
MuJoCo menagerie FR3 model. All angles in radians, lengths in metres, frames relative to the
robot base (fr3_link0); the TCP is fr3_hand_tcp (between the fingertips).
"""
import math

import numpy as np

# Franka's modified DH parameters (a, d, alpha) per joint, then the flange and hand TCP.
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2),
       (0.0825, 0.0, math.pi / 2), (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2),
       (0.088, 0.0, math.pi / 2)]
FLANGE_D = 0.107       # link7 -> link8 (flange)
TCP_D = 0.1034         # flange -> fr3_hand_tcp
TCP_YAW = -math.pi / 4

# FR3 joint position limits (rad), current Franka spec.
LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
LIMIT_PAD = 0.05       # IK never goes closer than this to a limit (rad)

# Franka's standard "ready" pose: arm bent, gripper pointing down, far from singularities.
READY_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# The FR3 firmware stops with 'self_collision_avoidance_violation' when the elbow folds so far
# that the wrist comes near the upper arm (observed: stopped at 0.295 m; ready pose is 0.40 m).
MIN_WRIST_TO_UPPER_ARM = 0.35


# ---------------- Rotation helpers ----------------

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_between(a, b):
    """Smallest rotation matrix that turns direction a onto direction b."""
    a, b = unit(a), unit(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:  # opposite directions: half turn about any perpendicular axis
        axis = unit(np.cross(a, (1, 0, 0) if abs(a[0]) < 0.9 else (0, 1, 0)))
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


def quat_to_mat(q):
    """(x, y, z, w) -> rotation matrix."""
    x, y, z, w = unit(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------- Kinematics ----------------

def frames(q):
    """World transforms (4x4) of joint frames 1..7, then the TCP."""
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
    tcp[:3, :3] = rot_z(TCP_YAW)
    tcp[2, 3] = FLANGE_D + TCP_D
    out.append(T @ tcp)
    return out


def fk(q):
    """TCP position (3,) and rotation (3x3) for joint angles q."""
    T = frames(q)[-1]
    return T[:3, 3].copy(), T[:3, :3].copy()


def jacobian(frs):
    """6x7 geometric Jacobian of the TCP (linear rows first), from frames(q)."""
    p = frs[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = frs[i][:3, 2], frs[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def ik(p_target, R_target, seed, reg=0.0, pos_tol=1e-3, ang_tol_deg=0.5):
    """Bounded least-squares IK for the TCP, starting near `seed`. Returns (q, ok).

    reg > 0 pulls the solution toward the seed (useful for tracking a path smoothly).
    """
    from scipy.optimize import least_squares
    lo, hi = LOWER + LIMIT_PAD, UPPER - LIMIT_PAD
    seed = np.clip(np.asarray(seed, dtype=float), lo + 1e-6, hi - 1e-6)
    w_rot = 0.3  # metres per radian: weight of orientation vs position error

    def residual(q):
        fr = frames(q)
        p, R = fr[-1][:3, 3], fr[-1][:3, :3]
        E = R_target @ R.T
        rot_err = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
        return np.concatenate([p_target - p, w_rot * rot_err, reg * (q - seed)])

    def jac(q):
        J = jacobian(frames(q))
        return np.vstack([-J[:3], -w_rot * J[3:], reg * np.eye(7)])

    res = least_squares(residual, seed, jac=jac, bounds=(lo, hi), xtol=1e-12, ftol=1e-12,
                        gtol=1e-12, max_nfev=200)
    p, R = fk(res.x)
    ang = math.degrees(math.acos(np.clip((np.trace(R_target @ R.T) - 1) / 2, -1, 1)))
    return res.x, bool(np.linalg.norm(p - p_target) < pos_tol and ang < ang_tol_deg)


# ---------------- Safety checks ----------------

def joint_margin(q):
    """Smallest distance (rad) of any joint from its limit; for one config or an (N, 7) array."""
    q = np.asarray(q)
    return float(np.min(np.minimum(q - LOWER, UPPER - q)))


def wrist_to_upper_arm(q):
    """Distance (m) from the wrist/flange to the shoulder-elbow segment (elbow fold check).

    Keep it >= MIN_WRIST_TO_UPPER_ARM or the FR3 firmware may stop with a self-collision error.
    """
    fr = frames(q)
    shoulder, elbow = fr[0][:3, 3], fr[2][:3, 3]
    flange = fr[6][:3, 3] + fr[6][:3, 2] * FLANGE_D
    ab = elbow - shoulder
    best = np.inf
    for p in (fr[4][:3, 3], fr[6][:3, 3], flange):
        t = np.clip((p - shoulder) @ ab / (ab @ ab), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (shoulder + t * ab))))
    return best


def clear_of_table(qs, min_z=0.08):
    """Elbow and wrist frames stay above height min_z for every configuration in qs."""
    return all(min(T[2, 3] for T in frames(q)[2:]) > min_z for q in qs)
