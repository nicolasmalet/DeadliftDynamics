from dataclasses import dataclass

import numpy as np

from config import t


@dataclass(frozen=True)
class Bone:
    name: str
    r: float
    m: float
    color: tuple[int, int, int]

    @property
    def J(self) -> float:
        return self.m * self.r**2 / 12

    def get_e_r_and_e_theta(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        s, c = np.sin(theta), np.cos(theta)
        return np.array([s, -c]), np.array([c, s])

    def get_P(self, theta: float) -> np.ndarray:
        return np.column_stack(self.get_e_r_and_e_theta(theta))

    def get_end(self, origin: np.ndarray, theta: float) -> np.ndarray:
        e_r, _ = self.get_e_r_and_e_theta(theta)
        return origin + self.r * e_r

    def get_G(self, origin: np.ndarray, theta: float) -> np.ndarray:
        return (origin + self.get_end(origin, theta)) / 2

    @staticmethod
    def get_theta_dot(theta: float, theta_previous: float) -> float:
        return (theta - theta_previous) / t


@dataclass(frozen=True)
class Muscle:
    name: str
    index: int
    bone0: int
    bone1: int
    relative_0: tuple[float, float]
    relative_1: tuple[float, float]
    max_force: float

    def tendon_position(self, bone: int, current: "Frame") -> np.ndarray:
        relative = self.relative_0 if bone == self.bone0 else self.relative_1
        if bone == -1:
            return np.array(relative)
        return current.origins[bone] + np.column_stack((current.e_r[bone], current.e_theta[bone])) @ relative

    def get_force(self, current: "Frame") -> np.ndarray:
        direction = self.tendon_position(self.bone1, current) - self.tendon_position(self.bone0, current)
        direction /= np.linalg.norm(direction)
        return self.max_force * direction

    @staticmethod
    def get_torque(point: np.ndarray, center: np.ndarray, force: np.ndarray) -> float:
        arm = point - center
        return arm[0] * force[1] - arm[1] * force[0]


@dataclass(frozen=True)
class Frame:
    e_r: np.ndarray
    e_theta: np.ndarray
    origins: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    theta_dot: np.ndarray
    center_velocity: np.ndarray

    @classmethod
    def from_angles(cls, bones: tuple[Bone, ...], theta: np.ndarray, theta_previous: np.ndarray) -> "Frame":
        lengths = np.array([bone.r for bone in bones])
        e_r, e_theta = cls.get_directions(bones, theta)
        origins = cls.get_origins(lengths, e_r)
        centers = cls.get_centers(bones, origins, theta)
        ends = cls.get_ends(origins, centers)
        theta_dot = cls.get_theta_dot(bones, theta, theta_previous)
        velocity = cls.get_center_velocity(lengths, e_theta, theta_dot)
        return cls(e_r, e_theta, origins, ends, centers, theta_dot, velocity)

    @staticmethod
    def get_directions(bones: tuple[Bone, ...], theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotations = np.array([bone.get_P(theta[i]) for i, bone in enumerate(bones)])
        return rotations[:, :, 0], rotations[:, :, 1]

    @staticmethod
    def get_origins(lengths: np.ndarray, e_r: np.ndarray) -> np.ndarray:
        origins = np.zeros((len(lengths), 2))
        origins[1:] = np.cumsum(lengths[:-1, None] * e_r[:-1], axis=0)
        return origins

    @staticmethod
    def get_centers(bones: tuple[Bone, ...], origins: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.array([bone.get_G(origins[i], theta[i]) for i, bone in enumerate(bones)])

    @staticmethod
    def get_ends(origins: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return 2 * centers - origins

    @staticmethod
    def get_theta_dot(bones: tuple[Bone, ...], theta: np.ndarray, theta_previous: np.ndarray) -> np.ndarray:
        return np.array([bone.get_theta_dot(theta[i], theta_previous[i]) for i, bone in enumerate(bones)])

    @staticmethod
    def get_center_velocity(lengths: np.ndarray, e_theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        link_velocity = lengths[:, None] * theta_dot[:, None] * e_theta
        origin_velocity = np.zeros_like(link_velocity)
        origin_velocity[1:] = np.cumsum(link_velocity[:-1], axis=0)
        return origin_velocity + link_velocity / 2


def muscle_actions(
    bones: tuple[Bone, ...], muscles: tuple[Muscle, ...], current: Frame
) -> tuple[np.ndarray, np.ndarray]:
    muscle_forces = np.zeros((len(bones), len(muscles), 2))
    muscle_torques = np.zeros((len(bones), len(muscles)))
    for muscle in muscles:
        force = muscle.get_force(current)
        for bone, applied_force in ((muscle.bone0, force), (muscle.bone1, -force)):
            if bone == -1:
                continue
            point = muscle.tendon_position(bone, current)
            muscle_forces[bone, muscle.index] = applied_force
            muscle_torques[bone, muscle.index] = muscle.get_torque(point, current.centers[bone], applied_force)
    return muscle_forces, muscle_torques
