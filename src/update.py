import numpy as np

from energy import get_gravity_center, p_muscle, total_kinetic_energy, total_potential_energy
from matrix import a_ij, b_i
from state import State


def update_bones(state: State, X: np.ndarray, shallow_update: bool = False) -> None:
    """Update the angular position of every bone from the solution vector."""
    for bone in state.bones:
        bone.theta = X[bone.index]
        bone.l_theta.append(bone.theta)
        bone.update(state.bones, shallow_update)


def update_model(state: State, efforts: np.ndarray, shallow_update: bool = False) -> None:
    """Assemble and solve one step of the dynamics."""
    bones = state.bones
    n = len(bones)
    c = [np.cos(bone.theta) for bone in bones]
    s = [np.sin(bone.theta) for bone in bones]
    l_forces = [bone.F_tot(efforts) for bone in bones]
    l_torques = [bone.C_tot(efforts) for bone in bones]

    A = np.array([[a_ij(i, j, bones, c, s) for j in range(3 * n)] for i in range(3 * n)])
    B = np.array([b_i(i, bones, c, s, l_forces, l_torques) for i in range(3 * n)])
    X = np.linalg.solve(A, B)

    update_bones(state, X, shallow_update)
    state.l_gravity_center.append(get_gravity_center(bones))

    if not shallow_update:
        for i, muscle in enumerate(state.muscles):
            state.l_p_muscle[i].append(p_muscle(muscle, efforts[i]))
        state.Ec.append(total_kinetic_energy(bones))
        state.Ep.append(total_potential_energy(bones))


def reset_energy(state: State) -> None:
    state.Ec[:] = [0.0]
    state.Ep[:] = [total_potential_energy(state.bones)]
    state.l_p_muscle[:] = [[0.0] for _ in state.muscles]
