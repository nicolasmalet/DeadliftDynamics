import sys
from pathlib import Path

import numpy as np
import pygame as pg

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from brain import make_decision
from pygame_interface import create_display, update_display
from state import create_state
from update import update_model

model, state, history = create_state()
efforts = history.efforts[-1]
for _ in range(3):
    efforts = make_decision(model, state, efforts)
    state = update_model(model, state, efforts)

assert np.isfinite(state.theta).all()
assert np.all((efforts >= 0) & (efforts <= 1))
screen, font = create_display()
assert update_display(model, state, efforts, 0, screen, font)
pg.quit()
