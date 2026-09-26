import pygame as pg

from bone import Frame
from brain import make_decision
from config import plot_e, plot_eff, plot_m, plot_p, plot_Q_function, review, t
from history import History
from plot import plot_efforts, plot_energies, plot_movement, plot_phase_portrait, plot_Q
from pygame_interface import create_display, update_display
from state import create_state
from update import update_model


def main() -> None:
    model, state, efforts = create_state()
    history = History(states=[state], efforts=[efforts])
    screen, font = create_display()
    current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
    i = 0

    while True:
        previous = current
        efforts = make_decision(model, state, history.efforts[-1])
        state = update_model(model, state, efforts)
        current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
        history.update(model, state, efforts, previous, current)
        if not update_display(model, state, efforts, i * t, screen, font):
            break
        i += 1

    if review:
        for i, (saved_state, efforts) in enumerate(zip(history.states, history.efforts, strict=True)):
            update_display(model, saved_state, efforts, i * t, screen, font)

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
