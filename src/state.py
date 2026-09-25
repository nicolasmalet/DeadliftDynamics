from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np

from bone import Bone, Muscle
from config import bar_mass, bone_color, g, muscle_color
from energy import get_gravity_center, total_potential_energy


@dataclass
class State:
    ground: Bone
    bones: list[Bone]
    muscles: list[Muscle]
    efforts: np.ndarray
    l_efforts: list[np.ndarray]
    l_Q: list[list[float]]
    Ec: list[float]
    Ep: list[float]
    l_p_muscle: list[list[float]]
    l_gravity_center: list[np.ndarray]

    def copy_for_trial(self) -> "State":
        """Copy only the values changed by a gradient trial."""
        state = copy(self)
        state.bones = [copy(bone) for bone in self.bones]
        previous_bone = self.ground
        for bone in state.bones:
            bone.previous_bone = previous_bone
            bone.l_theta = bone.l_theta[-3:].copy()
            previous_bone = bone
        state.efforts = self.efforts.copy()
        state.l_gravity_center = [value.copy() for value in self.l_gravity_center[-2:]]
        return state

    def reset_bones(self) -> None:
        for bone in self.bones:
            bone.l_theta = bone.l_theta[0:3]
            bone.theta = bone.l_theta[-1]
            assert bone.first_state is not None
            bone.set_state(bone.first_state)


def create_state() -> State:
    calves = Muscle("calves", 0, [-0.05, 0], [0.4, 0], 10000, muscle_color)
    quadriceps = Muscle("quadriceps", 1, [0.55, 0], [0.35, 0], 10000, muscle_color)
    hamstrings = Muscle("hamstrings", 2, [0.05, 0], [-0.05, 0], 10000, muscle_color)
    low_back = Muscle("low back", 3, [0.45, 0], [0.44, 0], 10000, muscle_color)
    lats = Muscle("lats", 4, [0.05, 0], [0.05, 0], 2000, muscle_color)

    ground = Bone("ground", None, [calves], 0, np.pi / 2, 0, bone_color)
    tibia = Bone("tibia", ground, [calves, quadriceps], 0.49, 2.758207465905639, 6, bone_color)
    femur = Bone("femur", tibia, [quadriceps, hamstrings, low_back], 0.40, -1.8630010384296538, 14, bone_color)
    back = Bone("back", femur, [hamstrings, low_back, lats], 0.49, 2.2435477244889137, 38, bone_color)
    arm = Bone("arm", back, [lats], 0.65, -0.005690488236282301, 8 + bar_mass, bone_color)

    bones = [tibia, femur, back, arm]
    muscles = [calves, quadriceps, hamstrings, low_back, lats]

    for bone in [ground, *bones]:
        for muscle in muscles:
            if muscle in bone.muscles:
                if muscle.bone0 is None:
                    muscle.bone0 = bone
                else:
                    muscle.bone1 = bone

    for bone in bones:
        bone.update(bones)
        bone.first_state = deepcopy(bone.get_state())
        bone.Ep0 = bone.m * g * bone.G[1]

    efforts = np.array([0.36, 0.68, 1, 1, 0])
    gravity_center = get_gravity_center(bones)
    return State(
        ground=ground,
        bones=bones,
        muscles=muscles,
        efforts=efforts,
        l_efforts=[efforts.copy()],
        l_Q=[[-1532.1913424428103, 3749.6096493863097, 24.366758337800864, 0.0, 51.02162584457079, 5206.412607646748]],
        Ec=[0.0],
        Ep=[total_potential_energy(bones)],
        l_p_muscle=[[0.0] for _ in muscles],
        l_gravity_center=[gravity_center.copy(), gravity_center.copy()],
    )
