import matplotlib.pyplot as plt
import numpy as np

from config import background_color, t
from state import History, Model
from utils import differentiate, integrate

plt.style.use("dark_background")


def finish(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(np.array(background_color) / 255)
    for axis in fig.get_axes():
        axis.legend()
        axis.set_facecolor(np.array(background_color) / 255)
    plt.show()


def angles(history: History) -> np.ndarray:
    return np.array([state.theta for state in history.states])


def plot_movement(model: Model, history: History) -> None:
    theta = angles(history)
    time = np.arange(len(theta)) * t
    fig, axes = plt.subplots(2, 2)
    for i in range(len(model.bones)):
        axes[0, 0].plot(time, theta[:, i], label=f"theta{i}")
        axes[0, 1].plot(time[1:], differentiate(theta[:, i].tolist()), label=f"theta{i}_dot")
        axes[1, 0].plot(time[2:], differentiate(differentiate(theta[:, i].tolist())), label=f"theta{i}_dotdot")
    kinetic = np.array(history.kinetic_energy)
    potential = np.array(history.potential_energy)
    energy_time = time[1:]
    axes[1, 1].plot(energy_time, kinetic, label="Ec", color="magenta")
    axes[1, 1].plot(energy_time, potential, label="Ep", color="cyan")
    axes[1, 1].plot(energy_time, kinetic + potential, label="Em", color="white")
    for axis, title, unit in zip(
        axes.flat,
        ("Angle", "Angular velocity", "Angular acceleration", "Energy"),
        ("angle (rad)", "angular velocity (rad/s)", "angular acceleration (rad/s2)", "Energy (J)"),
        strict=True,
    ):
        axis.set(title=title, xlabel="time (s)", ylabel=unit)
    finish(fig)


def plot_energies(model: Model, history: History) -> None:
    kinetic = np.array(history.kinetic_energy)
    potential = np.array(history.potential_energy)
    power = np.array(history.muscle_power).T
    time = np.arange(1, len(history.states)) * t
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(time, kinetic, label="Ec", color="magenta")
    axes[0, 0].plot(time, potential, label="Ep", color="cyan")
    axes[0, 0].plot(time, kinetic + potential, label="Em", color="white")
    for muscle, values in zip(model.muscles, power, strict=True):
        axes[0, 1].plot(time, integrate(values.tolist())[1:], label=muscle.name)
    axes[1, 0].plot(time, kinetic + potential, label="Em", color="white")
    axes[1, 0].plot(time, integrate(power.sum(axis=0).tolist())[1:], label="E tot muscles", color="magenta")
    axes[1, 1].plot(time[1:], differentiate((kinetic + potential).tolist()), label="P system", color="white")
    axes[1, 1].plot(time, power.sum(axis=0), label="P tot muscles", color="magenta")
    finish(fig)


def plot_phase_portrait(model: Model, history: History) -> None:
    theta = angles(history)
    for i in range(len(model.bones)):
        plt.plot(theta[:-1, i], np.diff(theta[:, i]) / t)
    plt.show()


def plot_efforts(model: Model, history: History) -> None:
    efforts = np.array(history.efforts[1:])
    time = np.arange(1, len(history.states)) * t
    fig, axes = plt.subplots(2, 3)
    locations = ((1, 1), (0, 1), (1, 0), (0, 0), (0, 2))
    colors = ("#81b1d2", "#feffb3", "#fa8174", "#8dd3c7", "#bfbbd9")
    for muscle, location, color in zip(model.muscles, locations, colors, strict=True):
        axis = axes[location]
        axis.plot(time, efforts[:, muscle.index], label=muscle.name, color=color)
        axis.set(xlabel="time (s)", ylabel="normalized force")
    names = ("Q", "shoulder height", "gravity center pos", "gravity center speed", "g", "dg/dt")
    for i, name in enumerate(names):
        axes[1, 2].plot(time, np.array(history.q_terms)[:, i], label=name)
    axes[1, 2].set_yscale("log")
    finish(fig)


def plot_Q(model: Model, history: History) -> None:
    del model
    fig, axis = plt.subplots()
    time = np.arange(1, len(history.states)) * t
    names = ("Q", "shoulder height", "gravity center pos", "gravity center speed", "g", "dg/dt")
    for i, name in enumerate(names):
        axis.plot(time, np.array(history.q_terms)[:, i], label=name)
    axis.set_yscale("log")
    finish(fig)
