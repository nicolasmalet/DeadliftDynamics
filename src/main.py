import pygame as pg

from bone import Frame
from brain import Q_terms, make_decision
from config import plot_e, plot_eff, plot_m, plot_p, plot_Q_function, review, show_model, t
from energy import muscle_power, total_kinetic_energy, total_potential_energy
from plot import plot_efforts, plot_energies, plot_movement, plot_phase_portrait, plot_Q
from pygame_interface import create_display, update_display
from state import create_state
from update import update_model


def main() -> None:
    model, state, history = create_state()
    screen, font = create_display()
    i = 0

    while True:
        state = update_model(model, state, make_decision(model, state))
        current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
        history.states.append(state)
        history.q_terms.append(Q_terms(model, state))
        history.kinetic_energy.append(total_kinetic_energy(model, current))
        history.potential_energy.append(total_potential_energy(model, current))
        history.muscle_power.append(muscle_power(model, current, state.efforts))
        if show_model and not update_display(model, state, i * t, screen, font):
            break
        i += 1

    if review:
        for i, saved_state in enumerate(history.states):
            update_display(model, saved_state, i * t, screen, font)

    pg.quit()

    if plot_m:
        plot_movement(model, history)
    if plot_e:
        plot_energies(model, history)
    if plot_p:
        plot_phase_portrait(model, history)
    if plot_eff:
        plot_efforts(model, history)
    if plot_Q_function:
        plot_Q(model, history)


if __name__ == "__main__":
    main()
