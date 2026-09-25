from collections.abc import Callable

import numpy as np

from config import t
from state import Model, State
from update import PreparedStep, prepare_step, update_model


def make_decision(model: Model, state: State) -> np.ndarray:
    """Determine the next muscle efforts by minimizing -Q."""
    return gradient_descent(model, state, state.efforts, minus_Q, prepare_step(model, state), 0.001, 10**-3 / t, 0.7)


def gradient_descent(
    model: Model,
    state: State,
    v: np.ndarray,
    f: Callable[[Model, State], float],
    prepared: PreparedStep,
    epsilon: float,
    k: float,
    gamma: float,
    max_iterations: int = 100,
) -> np.ndarray:
    """Minimize f by projected gradient descent with finite differences."""
    m = len(model.muscles)
    x = v.copy()
    delta = 1e-3
    directions = np.eye(m)

    for n in range(max_iterations):
        nabla = np.zeros(m)
        for i in range(m):
            lower = max(0.0, x[i] - delta)
            upper = min(1.0, x[i] + delta)

            lower_state = update_model(model, state, x + (lower - x[i]) * directions[i], prepared)
            f_lower = f(model, lower_state)

            upper_state = update_model(model, state, x + (upper - x[i]) * directions[i], prepared)
            f_upper = f(model, upper_state)

            nabla[i] = (f_upper - f_lower) / (upper - lower)

        updated = np.clip(x - k * gamma**n * nabla, 0, 1)
        if np.linalg.norm(updated - x, ord=np.inf) < epsilon:
            return updated
        x = updated

    raise RuntimeError("control update did not become smaller than epsilon")


def minus_Q(model: Model, state: State) -> float:
    """Return the loss minimized by the controller."""
    return -Q(model, state)


def Q(model: Model, state: State) -> float:
    """Evaluate the scalar control score on the supplied state."""
    return Q_terms(model, state)[0]


def Q_terms(model: Model, state: State) -> list[float]:
    """Evaluate the control score and its five components."""
    a = 50 / t
    b = -(10**7)
    c = -(10**3)
    d = -(1 * 10**5)
    e = -(2 * 10**-6)

    shoulder_y = -sum(bone.r * np.cos(state.theta[i]) for i, bone in enumerate(model.bones[:3]))
    y1 = a * (shoulder_y - 0.8)
    y2 = b * (state.gravity_center[0] - 0.14) ** 2
    y3 = c * ((state.gravity_center[0] - state.gravity_center_previous[0]) / t) ** 2
    y4 = d * state.alignment
    y5 = e * (d * (state.alignment - state.alignment_previous) / t) ** 2
    y = y1 + y2 + y3 + y4 + y5

    return [y, y1, -y2, -y3, -y4, -y5]


def g(x: float, y: float, z: float) -> float:
    return (x - y) ** 2 + (x - z) ** 2 + (y - z) ** 2
