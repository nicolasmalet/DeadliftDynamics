from dataclasses import dataclass

import numpy as np

from config import g, t


@dataclass(frozen=True)
class Bone:
    name: str
    r: float
    m: float
    color: tuple[int, int, int]

    @property
    def J(self) -> float:
        return self.m * self.r**2 / 12


@dataclass(frozen=True)
class Muscle:
    name: str
    index: int
    bone0: int
    bone1: int
    relative_0: tuple[float, float]
    relative_1: tuple[float, float]
    max_force: float


@dataclass(frozen=True)
class Frame:
    e_r: np.ndarray
    e_theta: np.ndarray
    origins: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    theta_dot: np.ndarray
    center_velocity: np.ndarray


def frame(bones: tuple[Bone, ...], theta: np.ndarray, theta_previous: np.ndarray) -> Frame:
    lengths = np.array([bone.r for bone in bones])
    e_r = np.column_stack((np.sin(theta), -np.cos(theta)))
    e_theta = np.column_stack((np.cos(theta), np.sin(theta)))
    origins = np.zeros((len(bones), 2))
    origins[1:] = np.cumsum(lengths[:-1, None] * e_r[:-1], axis=0)
    ends = origins + lengths[:, None] * e_r
    centers = (origins + ends) / 2
    theta_dot = (theta - theta_previous) / t
    link_velocity = lengths[:, None] * theta_dot[:, None] * e_theta
    origin_velocity = np.zeros_like(link_velocity)
    origin_velocity[1:] = np.cumsum(link_velocity[:-1], axis=0)
    center_velocity = origin_velocity + link_velocity / 2
    return Frame(e_r, e_theta, origins, ends, centers, theta_dot, center_velocity)


def attachment(muscle: Muscle, bone: int, current: Frame) -> np.ndarray:
    relative = muscle.relative_0 if bone == muscle.bone0 else muscle.relative_1
    if bone == -1:
        return np.array(relative)
    return current.origins[bone] + np.column_stack((current.e_r[bone], current.e_theta[bone])) @ relative


def forces_and_torques(
    bones: tuple[Bone, ...], muscles: tuple[Muscle, ...], current: Frame, efforts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    muscle_forces = np.zeros((len(bones), len(muscles), 2))
    muscle_torques = np.zeros((len(bones), len(muscles)))
    for muscle in muscles:
        p0 = attachment(muscle, muscle.bone0, current)
        p1 = attachment(muscle, muscle.bone1, current)
        direction = p1 - p0
        direction /= np.linalg.norm(direction)
        for bone, sign, point in ((muscle.bone0, 1, p0), (muscle.bone1, -1, p1)):
            if bone == -1:
                continue
            force = sign * muscle.max_force * direction
            muscle_forces[bone, muscle.index] = force
            arm = point - current.centers[bone]
            muscle_torques[bone, muscle.index] = arm[0] * force[1] - arm[1] * force[0]

    forces = np.array([efforts @ muscle_forces[i] + np.array([0.0, -bone.m * g]) for i, bone in enumerate(bones)])
    torques = muscle_torques @ efforts
    return forces, torques, muscle_forces, muscle_torques
