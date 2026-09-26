from dataclasses import dataclass

import numpy as np

from bone import Bone, Frame, Muscle
from config import bar_mass, bone_color


@dataclass(frozen=True)
class Model:
    bones: tuple[Bone, ...]
    muscles: tuple[Muscle, ...]
    initial_centers: np.ndarray


@dataclass(frozen=True)
class State:
    theta_previous: np.ndarray
    theta: np.ndarray


def gravity_center(model: Model, centers: np.ndarray) -> np.ndarray:
    masses = np.array([bone.m for bone in model.bones])
    return masses @ centers / masses.sum()


def create_state() -> tuple[Model, State, np.ndarray]:
    bones = (
        Bone("tibia", 0.49, 6, bone_color),
        Bone("femur", 0.40, 14, bone_color),
        Bone("back", 0.49, 38, bone_color),
        Bone("arm", 0.65, 8 + bar_mass, bone_color),
    )
    muscles = (
        Muscle("calves", 0, -1, 0, (-0.05, 0), (0.4, 0), 10000),
        Muscle("quadriceps", 1, 0, 1, (0.55, 0), (0.35, 0), 10000),
        Muscle("hamstrings", 2, 1, 2, (0.05, 0), (-0.05, 0), 10000),
        Muscle("low back", 3, 1, 2, (0.45, 0), (0.44, 0), 10000),
        Muscle("lats", 4, 2, 3, (0.05, 0), (0.05, 0), 2000),
    )
    theta = np.array([2.758207465905639, -1.8630010384296538, 2.2435477244889137, -0.005690488236282301])
    current = Frame.from_angles(bones, theta, theta)
    model = Model(bones, muscles, current.centers.copy())
    state = State(theta.copy(), theta)
    efforts = np.array([0.36, 0.68, 1.0, 1.0, 0.0])
    return model, state, efforts
