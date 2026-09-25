from copy import copy
from typing import Callable, List, Union

import numpy as np

import state
from config import t
from update import update_model, reverse_l_gravity_center_changes


def make_decision() -> np.ndarray:
    """
    Determines the next set of muscle efforts by minimizing -Q.
    """
    state.efforts = gradient_descent(state.efforts, minus_Q, 0.001, 10 ** -3 / t, 0.7)
    state.l_efforts.append(copy(state.efforts))
    state.l_Q.append(Q(True))

    return state.efforts


def gradient_descent(
    v: np.ndarray,
    f: Callable[[], float],
    epsilon: float,
    k: float,
    gamma: float,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Minimizes f by projected gradient descent with finite differences.
    """
    m = len(state.muscles)
    x = v.copy()
    bone_states = [bone.get_state() for bone in state.bones]
    delta = 1e-3

    for n in range(max_iterations):
        nabla = np.zeros(m)
        for i in range(m):
            lower = max(0.0, x[i] - delta)
            upper = min(1.0, x[i] + delta)

            update_model(x + (lower - x[i]) * e_i(i, m), shallow_update=True)
            f_lower = f()
            reverse_changes(bone_states)

            update_model(x + (upper - x[i]) * e_i(i, m), shallow_update=True)
            f_upper = f()
            reverse_changes(bone_states)

            nabla[i] = (f_upper - f_lower) / (upper - lower)

        updated = np.clip(x - k * gamma ** n * nabla, 0, 1)
        if np.linalg.norm(updated - x, ord=np.inf) < epsilon:
            return updated
        x = updated

    raise RuntimeError("control update did not become smaller than epsilon")


def minus_Q() -> float:
    """Returns the loss minimized by the controller."""
    return -Q()


def Q(explicit: bool = False) -> Union[float, List[float]]:
    """
    The cost/quality function to be optimized. Evaluates the current state of the simulation.
    """
    a = 50 / t
    b = - 1 * 10 ** 7
    c = - 1 * 10 ** 3
    d = - 1 * 10 ** 4
    e = - 2 * 10 ** - 6

    y1 = a * (state.bones[2].end[1] - 0.8)
    y2 = b * (state.l_gravity_center[-1][0] - 0.14) ** 2
    y3 = c * ((state.l_gravity_center[-1][0] - state.l_gravity_center[-2][0]) / t) ** 2
    y4 = d * g(state.bones[2].end[0], state.bones[3].end[0], state.bones[0].end[0])
    y5 = e * ((y4 + state.l_Q[-1][4]) / t) ** 2
    y = y1 + y2 + y3 + y4 + y5

    if explicit:
        return [y, y1, -y2, -y3, -y4, -y5]

    return y


def g(x: float, y: float, z: float) -> float:
    """
    Helper geometry function used in the cost function.
    """
    return ((x - y) ** 2 + (x - z) ** 2 + (y - z) ** 2) ** 0.5


def e_i(i: int, j: int) -> np.ndarray:
    """
    Returns a basis vector with 1 at index i and length j.
    """
    e = np.zeros(j)
    e[i] = 1
    return e


def reverse_changes(saved_bone_states: List[List[np.ndarray]]) -> None:
    """
    Reverts the simulation state to the provided saved state.
    """
    for bone in state.bones:
        bone.l_theta.pop()
        bone.theta = bone.l_theta[-1]
        bone.set_state(saved_bone_states[bone.index])
    reverse_l_gravity_center_changes()
