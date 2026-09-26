import sys
from pathlib import Path

import numpy as np
import pygame as pg

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bone import Frame
from brain import make_decision
from history import History
from pygame_interface import create_display, update_display
from state import create_state
from update import update_model

model, state, efforts = create_state()
history = History(states=[state], efforts=[efforts])
current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
for _ in range(3):
    previous = current
    efforts = make_decision(model, state, efforts)
    state = update_model(model, state, efforts)
    current = Frame.from_angles(model.bones, state.theta, state.theta_previous)
    history.update(model, state, efforts, previous, current)

assert np.isfinite(state.theta).all()
assert np.all((efforts >= 0) & (efforts <= 1))
assert len(history.states) == len(history.efforts) == 4
screen, font = create_display()
assert update_display(model, state, efforts, 0, screen, font)
pg.quit()
