from dataclasses import dataclass

import numpy as np

from bone import Frame, muscle_actions
from config import g
from matrix import a_ij, b_i
from state import Model, State


@dataclass(frozen=True)
class PreparedStep:
    current: Frame
    matrix: np.ndarray
    muscle_forces: np.ndarray
    muscle_torques: np.ndarray


def prepare_step(model: Model, state: State) -> PreparedStep:
    current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
    muscle_forces, muscle_torques = muscle_actions(model.bones, model.muscles, current)
    c, s = np.cos(state.theta), np.sin(state.theta)
    n = len(model.bones)
    matrix = np.array([[a_ij(i, j, model.bones, c, s) for j in range(3 * n)] for i in range(3 * n)])
    return PreparedStep(current, matrix, muscle_forces, muscle_torques)


def update_model(model: Model, state: State, efforts: np.ndarray, prepared: PreparedStep | None = None) -> State:
    """Solve one time step and return its dynamic state."""
    prepared = prepared or prepare_step(model, state)
    forces = np.array([efforts @ prepared.muscle_forces[i] + np.array([0.0, -bone.m * g]) for i, bone in enumerate(model.bones)])
    torques = prepared.muscle_torques @ efforts
    c, s = np.cos(state.theta), np.sin(state.theta)
    n = len(model.bones)
    vector = np.array([b_i(i, model.bones, state.theta, state.theta_previous, c, s, forces, torques) for i in range(3 * n)])
    theta = np.linalg.solve(prepared.matrix, vector)[:n]
    return State(state.theta, theta)
