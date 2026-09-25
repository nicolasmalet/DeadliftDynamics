
import numpy as np
import pygame as pg
from pygame.gfxdraw import aacircle, filled_circle

from bone import Bone, Muscle
from config import (
    background_color,
    bone_color,
    focus,
    ground_color,
    pixel_per_meter,
    r_bar,
    r_head,
    ratio_screen_reality,
    screen_size_x,
    screen_size_y,
    show_bar,
    show_gravity_center,
    show_ground,
    show_time,
)
from state import State


def create_display() -> tuple[pg.Surface, pg.font.Font]:
    pg.font.init()
    screen = pg.display.set_mode((screen_size_x, screen_size_y))
    screen.fill(background_color)
    return screen, pg.font.SysFont('Calibri', 50)


def pos_to_screen(pos: np.ndarray) -> tuple[int, int]:
    """
    Converts simulation coordinates to screen coordinates.
    """
    return (int((pos[0] - focus[0]) * pixel_per_meter * ratio_screen_reality) + screen_size_x // 2,
            int(-(pos[1] - focus[1]) * pixel_per_meter * ratio_screen_reality) + screen_size_y // 2)


def draw_line(screen: pg.Surface, p1: np.ndarray, p2: np.ndarray, color: tuple[int, int, int]) -> None:
    """
    Draws an anti-aliased line on the screen.
    """
    pg.draw.aaline(screen, color, pos_to_screen(p1), pos_to_screen(p2))


def draw_bone(screen: pg.Surface, bone: Bone) -> None:
    """
    Draws a bone on the screen.
    """
    draw_line(screen, bone.origin, bone.end, bone.color)


def draw_muscle(screen: pg.Surface, muscle: Muscle, effort: float) -> None:
    """
    Draws a muscle with color intensity based on effort.
    """
    draw_line(screen, muscle.origin(), muscle.end(), color_gradient(effort))


def draw_head(screen: pg.Surface, bones: list[Bone]) -> None:
    """
    Draws the head of the character.
    """
    x, y = pos_to_screen(bones[2].end + r_head * bones[2].e_r)
    r = int(r_head * pixel_per_meter * ratio_screen_reality)
    aacircle(screen, x, y, r, bone_color)


def draw_ground(screen: pg.Surface) -> None:
    """
    Draws the ground line.
    """
    y = pos_to_screen(np.array([0, 0]))[1]
    pg.draw.aaline(screen, ground_color, [0, y], [screen_size_x, y])


def draw_point(screen: pg.Surface, pos: np.ndarray) -> None:
    """
    Draws a small circle at a specific position.
    """
    x, y = pos_to_screen(pos)
    filled_circle(screen, x, y, 5, (100, 100, 255))


def draw_time(screen: pg.Surface, font: pg.font.Font, time: float) -> None:
    """
    Displays the simulation time on the screen.
    """
    text_surface = font.render(str(round(time, 2)) + 's', True, (220, 220, 220))
    screen.blit(text_surface, (40, screen_size_y - 54))


def draw_bar(screen: pg.Surface, bones: list[Bone]) -> None:
    """
    Draws the barbell.
    """
    x, y = pos_to_screen(bones[-1].end)
    r = int(r_bar * pixel_per_meter * ratio_screen_reality)
    filled_circle(screen, x, y, r, background_color)
    aacircle(screen, x, y, r, bone_color)


def color_gradient(x: float) -> tuple[int, int, int]:
    """
    Returns a color tuple shading from white to red based on intensity x (0 to 1).
    """
    return int(139 + x * (255 - 139)), int(255 * x) if x >= 0 else 0, int(-255 * x) if x <= 0 else 0


def update_display(state: State, time: float, screen: pg.Surface, font: pg.font.Font) -> bool:
    """
    Updates the PyGame display with the current state of the simulation.
    Returns False if the quit event is triggered, True otherwise.
    """
    bones, muscles, efforts = state.bones, state.muscles, state.efforts
    screen.fill(background_color)
    if show_ground:
        draw_ground(screen)

    for muscle in muscles:
        draw_muscle(screen, muscle, efforts[muscle.index])

    if show_time:
        draw_time(screen, font, time)

    for bone in bones:
        draw_bone(screen, bone)
    draw_head(screen, bones)

    if show_bar:
        draw_bar(screen, bones)

    if show_gravity_center:
        draw_point(screen, state.l_gravity_center[-1])

    pg.display.update()

    return all(event.type != pg.QUIT for event in pg.event.get())
