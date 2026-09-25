from collections.abc import Callable

import numpy as np

from bone import Frame
from config import t
from state import Model, State, gravity_center
from update import PreparedStep, prepare_step, update_model


def make_decision(model: Model, state: State, efforts: np.ndarray) -> np.ndarray:
    """Determine the next muscle efforts by minimizing -Q."""
    return gradient_descent(model, state, efforts, minus_Q, prepare_step(model, state), 0.001, 10**-3 / t, 0.7)


def gradient_descent(
    model: Model,
    state: State,
    v: np.ndarray,
    f: Callable[[Model, State, Frame], float],
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
            f_lower = f(model, lower_state, prepared.current)

            upper_state = update_model(model, state, x + (upper - x[i]) * directions[i], prepared)
            f_upper = f(model, upper_state, prepared.current)

            nabla[i] = (f_upper - f_lower) / (upper - lower)

        updated = np.clip(x - k * gamma**n * nabla, 0, 1)
        if np.linalg.norm(updated - x, ord=np.inf) < epsilon:
            return updated
        x = updated

    raise RuntimeError("control update did not become smaller than epsilon")


def minus_Q(model: Model, state: State, previous: Frame) -> float:
    """Return the loss minimized by the controller."""
    return -Q(model, state, previous)


def Q(model: Model, state: State, previous: Frame) -> float:
    """Evaluate the scalar control score on the supplied state."""
    return Q_terms(model, state, previous)[0]


def Q_terms(model: Model, state: State, previous: Frame) -> list[float]:
    """Evaluate the control score and its five components."""
    current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
    center = gravity_center(model, current.centers)
    center_previous = gravity_center(model, previous.centers)
    aligned = g(current.ends[0, 0], current.ends[3, 0], current.ends[2, 0])
    aligned_previous = g(previous.ends[0, 0], previous.ends[3, 0], previous.ends[2, 0])
    shoulder_y = -sum(bone.r * np.cos(state.theta[i]) for i, bone in enumerate(model.bones[:3]))
    features = np.array(
        [
            shoulder_y - 0.8,
            (center[0] - 0.14) ** 2,
            ((center[0] - center_previous[0]) / t) ** 2,
            aligned,
            (10**5 * (aligned - aligned_previous) / t) ** 2,
        ]
    )
    weights = np.array([50 / t, -(10**7), -(10**3), -(10**5), -(2 * 10**-6)])
    contributions = weights * features
    return [float(weights @ features), float(contributions[0]), *(-contributions[1:]).tolist()]


def g(x: float, y: float, z: float) -> float:
    return (x - y) ** 2 + (x - z) ** 2 + (y - z) ** 2
