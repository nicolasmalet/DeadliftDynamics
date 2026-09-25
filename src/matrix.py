import numpy as np

from bone import Bone
from config import t


def a_ij(i: int, j: int, bones: tuple[Bone, ...], c: np.ndarray, s: np.ndarray) -> float:
    """Computes the coefficient A[i, j] for the system matrix A."""
    n = len(bones)
    qi, qj = i // n, j // n
    ri, rj = i % n, j % n

    if qi == 0:
        if qj == 0:
            if ri == rj:
                return bones[ri].m * bones[ri].r * c[ri] / 2 / t**2
            if ri > rj:
                return bones[ri].m * bones[rj].r * c[rj] / t**2
        if qj == 1:
            if rj == ri:
                return -1
            if rj == ri + 1:
                return 1

    if qi == 1:
        if qj == 0:
            if ri == rj:
                return bones[ri].m * bones[ri].r * s[ri] / 2 / t**2
            if ri > rj:
                return bones[ri].m * bones[rj].r * s[rj] / t**2
        if qj == 2:
            if rj == ri:
                return -1
            if rj == ri + 1:
                return 1

    if qi == 2:
        if qj == 0 and ri == rj:
            return bones[ri].J / t**2
        if qj == 1:
            if rj == ri:
                return bones[ri].r * c[ri] / 2
            if rj == ri + 1:
                return bones[ri].r * c[ri] / 2
        if qj == 2:
            if rj == ri:
                return bones[ri].r * s[ri] / 2
            if rj == ri + 1:
                return bones[ri].r * s[ri] / 2
    return 0


def b_i(
    i: int,
    bones: tuple[Bone, ...],
    theta: np.ndarray,
    theta_previous: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
) -> float:
    """Computes the coefficient B[i] for the system vector B."""
    n = len(bones)
    q, index = i // n, i % n
    bone = bones[index]
    th = theta[index]
    _th = theta_previous[index]

    if q == 2:
        return bone.J / t**2 * (2 * th - _th) + torques[index]

    m = bone.m
    r = bone.r
    th_dot = (th - _th) / t

    if q == 0:
        return (
            forces[index][0]
            + m * r / 2 / t**2 * (c[index] * (2 * th - _th) + s[index] * (t * th_dot) ** 2)
            + m
            / t**2
            * sum(
                bones[p].r * (c[p] * (2 * theta[p] - theta_previous[p]) + s[p] * (theta[p] - theta_previous[p]) ** 2)
                for p in range(index)
            )
        )

    return (
        forces[index][1]
        + m * r / 2 / t**2 * (s[index] * (2 * th - _th) - c[index] * (t * th_dot) ** 2)
        + m
        / t**2
        * sum(
            bones[p].r * (s[p] * (2 * theta[p] - theta_previous[p]) - c[p] * (theta[p] - theta_previous[p]) ** 2)
            for p in range(index)
        )
    )
