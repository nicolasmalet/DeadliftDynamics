from __future__ import annotations

from typing import Any

import numpy as np

from config import g, t


class Bone:
    """
    Represents a bone segment in the biomechanical model.
    """

    def __init__(self, name: str, previous_bone: Bone | None, muscles: list[Muscle], length: float,
                 theta0: float, mass: float, color: tuple[int, int, int]) -> None:
        """
        Initializes a Bone instance.
        """
        self.name: str = name
        self.previous_bone: Bone | None = previous_bone
        self.index: int = self.get_index()
        self.r: float = length
        self.m: float = mass
        self.color: tuple[int, int, int] = color
        self.J: float = 1 / 12 * mass * length ** 2
        self.muscles: list[Muscle] = muscles
        self.Ep0: float = 0

        self.theta: float = theta0
        self.l_theta: list[float] = [theta0, theta0, theta0]

        self.e_r: np.ndarray
        self.e_theta: np.ndarray
        self.e_r, self.e_theta = self.get_e_r_and_e_theta()

        self.origin: np.ndarray = self.get_origin()
        self.end: np.ndarray = self.get_end()

        self.theta_dot: float = self.get_theta_dot()

        self.G: np.ndarray = self.get_G()
        self.G_dot: np.ndarray = np.array([0.0, 0.0])

        self.P: np.ndarray = self.get_P()

        self.F_max_muscles: dict[str, np.ndarray] = {}
        self.C_max_muscles: dict[str, float] = {}

        self.first_state: list[Any] | None = None

    def update(self, bones: list[Bone], shallow_update: bool = False) -> None:
        """
        Updates the physical properties of the bone based on the current angle theta.
        """
        self.e_r, self.e_theta = self.get_e_r_and_e_theta()

        self.origin = self.get_origin()
        self.end = self.get_end()

        self.P = self.get_P()

        self.G = self.get_G()

        if not shallow_update:

            self.theta_dot = self.get_theta_dot()

            self.G_dot = self.get_G_dot(bones)

            for muscle in self.muscles:
                self.F_max_muscles[muscle.name] = self.get_F_max_muscle(muscle)
                self.C_max_muscles[muscle.name] = self.get_C_max_muscle(muscle)

    def get_state(self) -> list[Any]:
        """
        Returns the current state of the bone as a list.
        """
        return [self.e_r, self.e_theta, self.origin, self.end, self.theta_dot, self.G, self.G_dot, self.P,
                self.F_max_muscles, self.C_max_muscles]

    def set_state(self, state: list[Any]) -> None:
        """
        Sets the state of the bone from a provided list.
        """
        self.F_max_muscles, self.C_max_muscles = {}, {}
        (self.e_r, self.e_theta, self.origin, self.end, self.theta_dot, self.G, self.G_dot, self.P,
         self.F_max_muscles, self.C_max_muscles) = state

    def get_index(self) -> int:
        """
        Recursively calculates the index of the bone in the chain.
        """
        if self.previous_bone is None:
            return -1
        return self.previous_bone.index + 1

    def get_origin(self) -> np.ndarray:
        """
        Calculates the position of the bone's origin (joint with previous bone).
        """
        if self.previous_bone is None:
            return np.array([0.0, 0.0])
        return self.previous_bone.end

    def get_end(self) -> np.ndarray:
        """
        Calculates the position of the bone's end.
        """
        return self.origin + self.r * self.e_r

    def get_e_r_and_e_theta(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the unit vectors e_r and e_theta based on the current angle.
        """
        s, c = np.sin(self.theta), np.cos(self.theta)
        return np.array([s, -c]), np.array([c, s])

    def get_G(self) -> np.ndarray:
        """
        Calculates the position of the center of gravity of the bone.
        """
        return 0.5 * (self.origin + self.end)

    def get_P(self) -> np.ndarray:
        """
        Returns the change of basis matrix from local to absolute coordinates.
        """
        return np.array([self.e_r, self.e_theta]).T

    def get_theta_dot(self) -> float:
        """
        Calculates the angular velocity using finite differences.
        """
        return (self.l_theta[-1] - self.l_theta[-2]) / t

    def get_G_dot(self, bones: list[Bone]) -> np.ndarray:
        """
        Calculates the velocity of the center of gravity.
        """
        return 0.5 * self.r * self.theta_dot * self.e_theta + \
            np.sum(np.array([bones[i].r * bones[i].theta_dot * bones[i].e_theta for i in range(self.index)]), axis=0)

    def get_F_max_muscle(self, muscle: Muscle) -> np.ndarray:
        """
        Calculates the maximum force vector that a muscle can exert on this bone.
        """
        u = muscle.tendon_position(muscle.other_bone(self)) - muscle.tendon_position(self)
        if (norm := np.linalg.norm(u)) != 0:
            u = u / norm
        return muscle.max_force * u

    def F_tot(self, efforts: np.ndarray) -> np.ndarray:
        """
        Calculates the total force exerted by muscles and gravity on this bone.
        """
        F_tot_muscle = np.sum(
            np.array([efforts[muscle.index] * self.F_max_muscles[muscle.name] for muscle in self.muscles]), axis=0)
        F_gravity = np.array([0, - self.m * g])
        return F_tot_muscle + F_gravity

    def get_C_max_muscle(self, muscle: Muscle) -> float:
        """
        Calculates the maximum torque a muscle can exert on this bone.
        """
        OM = muscle.tendon_position(self) - self.G
        F = self.F_max_muscles[muscle.name]
        return OM[0] * F[1] - OM[1] * F[0]

    def C_tot(self, efforts: np.ndarray) -> float:
        """
        Calculates the total torque exerted by all muscles on this bone.
        """
        C_tot_muscle = sum(efforts[muscle.index] * self.C_max_muscles[muscle.name] for muscle in self.muscles)
        return C_tot_muscle


class Muscle:
    """
    Represents a muscle connecting two bones.
    """

    def __init__(self, name: str, index: int, relative_start: list[float], relative_end: list[float], max_force: float,
                 color: tuple[int, int, int]) -> None:
        """
        Initializes a Muscle instance.
        """
        self.name: str = name
        self.index: int = index
        self.bone0: Bone | None = None
        self.bone1: Bone | None = None
        self.relative_0: list[float] = relative_start
        self.relative_1: list[float] = relative_end
        self.max_force: float = max_force
        self.color: tuple[int, int, int] = color

    def origin(self) -> np.ndarray:
        """
        Calculates the absolute position of the muscle's origin point.
        """
        assert self.bone0 is not None
        return self.bone0.origin + np.dot(self.bone0.P, self.relative_0)

    def end(self) -> np.ndarray:
        """
        Calculates the absolute position of the muscle's insertion point.
        """
        assert self.bone1 is not None
        return self.bone1.origin + np.dot(self.bone1.P, self.relative_1)

    def other_bone(self, bone: Bone) -> Bone:
        """
        Returns the bone connected to the muscle that is not the given bone.
        """
        other = self.bone1 if bone == self.bone0 else self.bone0
        assert other is not None
        return other

    def tendon_position(self, bone: Bone) -> np.ndarray:
        """
        Calculates the absolute position of the tendon attachment on the given bone.
        """
        return bone.origin + np.dot(bone.P, self.relative_tendon_position(bone))

    def relative_tendon_position(self, bone: Bone) -> list[float]:
        """
        Returns the relative position of the tendon attachment on the given bone.
        """
        if bone == self.bone0:
            return self.relative_0
        else:
            return self.relative_1
