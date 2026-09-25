import numpy as np

from bone import Frame, forces_and_torques
from config import g
from state import Model


def total_kinetic_energy(model: Model, current: Frame) -> float:
    return sum(
        0.5 * bone.J * current.theta_dot[i] ** 2 + 0.5 * bone.m * float(np.linalg.norm(current.center_velocity[i]) ** 2)
        for i, bone in enumerate(model.bones)
    )


def total_potential_energy(model: Model, current: Frame) -> float:
    return sum(bone.m * g * (current.centers[i, 1] - model.initial_centers[i, 1]) for i, bone in enumerate(model.bones))


def muscle_power(model: Model, current: Frame, efforts: np.ndarray) -> np.ndarray:
    _, _, forces, torques = forces_and_torques(model.bones, model.muscles, current, efforts)
    return np.array(
        [
            effort
            * sum(
                np.dot(forces[bone, muscle.index], current.center_velocity[bone])
                + torques[bone, muscle.index] * current.theta_dot[bone]
                for bone in (muscle.bone0, muscle.bone1)
                if bone != -1
            )
            for muscle, effort in zip(model.muscles, efforts, strict=True)
        ]
    )
