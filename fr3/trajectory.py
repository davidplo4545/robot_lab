"""Turn a joint-space path into a smooth, timed trajectory (no ROS).

Feed the result to Arm.execute_joint_trajectory().
"""
import math

import numpy as np

MAX_JOINT_VEL = 0.5    # rad/s, well under the FR3 limits
MAX_JOINT_ACC = 1.0    # rad/s^2


def time_trajectory(qs, duration, dt=0.05, max_vel=MAX_JOINT_VEL, max_acc=MAX_JOINT_ACC):
    """Time the joint path qs (N x 7): smooth ramp up, constant speed, smooth ramp down.

    Stretches `duration` if needed to respect max_vel / max_acc.
    Returns (duration, times, positions, velocities, accelerations) as numpy arrays.
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
        scale = max(np.max(np.abs(vel)) / max_vel, math.sqrt(np.max(np.abs(acc)) / max_acc))
        if scale <= 1.0:
            return duration, t, pos, vel, acc
        duration *= scale * 1.05


def blend_start(times, pos, vel, acc, q_now, fade=None):
    """Fade a small offset between q_now and pos[0] out smoothly over the first `fade` s.

    MoveIt requires a trajectory to start exactly at the current joints; this avoids a jump.
    Modifies pos/vel/acc in place and returns them.
    """
    delta = np.asarray(q_now, dtype=float) - pos[0]
    fade = min(3.0, times[-1] / 4) if fade is None else fade
    for i, t in enumerate(times):
        if t < fade:
            w = 0.5 * (1 + math.cos(math.pi * t / fade))
            dw = -0.5 * math.pi / fade * math.sin(math.pi * t / fade)
            ddw = -0.5 * (math.pi / fade) ** 2 * math.cos(math.pi * t / fade)
            pos[i] += delta * w
            if vel is not None:
                vel[i] += delta * dw
            if acc is not None:
                acc[i] += delta * ddw
    return pos, vel, acc
