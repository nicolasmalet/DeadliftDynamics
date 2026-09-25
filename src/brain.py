from typing import Callable

import numpy as np

from config import t
from state import State
from update import update_model


def make_decision(state: State) -> np.ndarray:
    """Determine the next muscle efforts by minimizing -Q."""
    state.efforts = gradient_descent(state, state.efforts, minus_Q, 0.001, 10**-3 / t, 0.7)
    state.l_efforts.append(state.efforts.copy())
    state.l_Q.append(Q(state, explicit=True))
    return state.efforts


def gradient_descent(
    state: State,
    v: np.ndarray,
    f: Callable[[State], float],
    epsilon: float,
    k: float,
    gamma: float,
    max_iterations: int = 100,
) -> np.ndarray:
    """Minimize f by projected gradient descent with finite differences."""
    m = len(state.muscles)
    x = v.copy()
    delta = 1e-3

    for n in range(max_iterations):
        nabla = np.zeros(m)
        for i in range(m):
            lower = max(0.0, x[i] - delta)
            upper = min(1.0, x[i] + delta)

            lower_state = state.copy()
            update_model(lower_state, x + (lower - x[i]) * e_i(i, m), shallow_update=True)
            f_lower = f(lower_state)

            upper_state = state.copy()
            update_model(upper_state, x + (upper - x[i]) * e_i(i, m), shallow_update=True)
            f_upper = f(upper_state)

            nabla[i] = (f_upper - f_lower) / (upper - lower)

        updated = np.clip(x - k * gamma**n * nabla, 0, 1)
        if np.linalg.norm(updated - x, ord=np.inf) < epsilon:
            return updated
        x = updated

    raise RuntimeError("control update did not become smaller than epsilon")


def minus_Q(state: State) -> float:
    """Return the loss minimized by the controller."""
    return -Q(state)


def Q(state: State, explicit: bool = False) -> float | list[float]:
    """Evaluate the control score on the supplied state."""
    a = 50 / t
    b = -(10**7)
    c = -(10**3)
    d = -(10**4)
    e = -(2 * 10**-6)

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
    return ((x - y) ** 2 + (x - z) ** 2 + (y - z) ** 2) ** 0.5


def e_i(i: int, j: int) -> np.ndarray:
    e = np.zeros(j)
    e[i] = 1
    return e
