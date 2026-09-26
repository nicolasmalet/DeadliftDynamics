from dataclasses import dataclass, field

import numpy as np

from bone import Frame
from brain import Q_terms
from energy import muscle_power, total_kinetic_energy, total_potential_energy
from state import Model, State


@dataclass
class History:
    states: list[State]
    efforts: list[np.ndarray]
    q_terms: list[list[float]] = field(default_factory=list)
    kinetic_energy: list[float] = field(default_factory=list)
    potential_energy: list[float] = field(default_factory=list)
    muscle_power: list[np.ndarray] = field(default_factory=list)

    def update_q(self, model: Model, state: State, previous: Frame) -> None:
        self.q_terms.append(Q_terms(model, state, previous))

    def update_kinetic_energy(self, model: Model, current: Frame) -> None:
        self.kinetic_energy.append(total_kinetic_energy(model, current))

    def update_potential_energy(self, model: Model, current: Frame) -> None:
        self.potential_energy.append(total_potential_energy(model, current))

    def update_muscle_power(self, model: Model, current: Frame, efforts: np.ndarray) -> None:
        self.muscle_power.append(muscle_power(model, current, efforts))

    def update(self, model: Model, state: State, efforts: np.ndarray, previous: Frame, current: Frame) -> None:
        self.states.append(state)
        self.efforts.append(efforts)
        self.update_q(model, state, previous)
        self.update_kinetic_energy(model, current)
        self.update_potential_energy(model, current)
        self.update_muscle_power(model, current, efforts)
