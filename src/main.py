import pygame as pg

from brain import make_decision
from config import plot_e, plot_eff, plot_m, plot_p, plot_Q_function, review, show_model, t
from plot import plot_efforts, plot_energies, plot_movement, plot_phase_portrait, plot_Q
from pygame_interface import create_display, update_display
from state import create_state
from update import reset_energy, update_model


def main() -> None:
    state = create_state()
    screen, font = create_display()
    i = 0

    while True:
        state.efforts = make_decision(state)
        update_model(state, state.efforts)
        if show_model and not update_display(state, i * t, screen, font):
            break
        i += 1

    if review:
        state.reset_bones()
        reset_energy(state)
        for i in range(1, len(state.l_efforts)):
            state.efforts = state.l_efforts[i]
            update_model(state, state.efforts)
            update_display(state, i * t, screen, font)

    pg.quit()

    if plot_m:
        plot_movement(state)
    if plot_e:
        plot_energies(state)
    if plot_p:
        plot_phase_portrait(state)
    if plot_eff:
        plot_efforts(state)
    if plot_Q_function:
        plot_Q(state)


if __name__ == "__main__":
    main()
