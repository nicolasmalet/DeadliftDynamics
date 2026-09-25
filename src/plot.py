import matplotlib.pyplot as plt
import numpy as np

from config import background_color, t
from state import State
from utils import differentiate, integrate

plt.style.use('dark_background')


def plot_movement(state: State) -> None:
    """
    Plots the angular position, velocity, acceleration, and system energy over time.
    """
    N = len(state.bones[0].l_theta) - 2

    fig, axs = plt.subplots(2, 2)
    t_axis = [i * t for i in range(N)]

    for i in range(len(state.bones)):
        axs[0, 0].plot(t_axis, state.bones[i].l_theta[2:], label='theta' + str(i), linewidth=1)
        axs[0, 1].plot(t_axis, differentiate(state.bones[i].l_theta[1:]), label='theta' + str(i) + '_dot', linewidth=1)
        axs[1, 0].plot(t_axis[:-1],
                       differentiate(differentiate(state.bones[i].l_theta[1:])), label='theta' + str(i) + '_dotdot',
                       linewidth=1)

    axs[1, 1].plot(t_axis, state.Ec, label='Ec', color='magenta', linewidth=1)
    axs[1, 1].plot(t_axis, state.Ep, label='Ep', color='cyan', linewidth=1)
    axs[1, 1].plot(t_axis, np.sum(np.array([state.Ec, state.Ep]), axis=0), label='Em', color='white', linewidth=1)

    axs[0, 0].set_title('Angle')
    axs[0, 1].set_title('Angular velocity')
    axs[1, 0].set_title('Angular acceleration')
    axs[1, 1].set_title('Energy')

    axs[0, 0].set_xlabel('time (s)')
    axs[0, 1].set_xlabel('time (s)')
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 1].set_xlabel('time (s)')

    axs[0, 0].set_ylabel('angle (rad)')
    axs[0, 1].set_ylabel('angular velocity (rad/s)')
    axs[1, 0].set_ylabel('angular acceleration (rad/s2)')
    axs[1, 1].set_ylabel('Energy (J)')

    fig.patch.set_facecolor(background_color / 255)

    for ax in fig.get_axes():
        ax.legend()
        ax.set_facecolor(background_color / 255)

    plt.show()


def plot_energies(state: State) -> None:
    """
    Plots detailed energy breakdowns including muscle work and power.
    """
    N = len(state.bones[0].l_theta) - 2

    fig, axs = plt.subplots(2, 2)
    t_axis = [i * t for i in range(N)]

    axs[0, 0].plot(t_axis, state.Ec, label='Ec', color='magenta', linewidth=1)
    axs[0, 0].plot(t_axis, state.Ep, label='Ep', color='cyan', linewidth=1)
    axs[0, 0].plot(t_axis, np.sum(np.array([state.Ec, state.Ep]), axis=0), label='Em', color='white', linewidth=1)

    for i in range(len(state.l_p_muscle)):
        axs[0, 1].plot(t_axis, integrate(state.l_p_muscle[i])[1: N + 1], label=state.muscles[i].name, linewidth=1)

    axs[1, 0].plot(t_axis, np.sum(np.array([state.Ec, state.Ep]), axis=0), label='Em', color='white', linewidth=1)
    axs[1, 0].plot(t_axis, integrate(np.sum(np.array(state.l_p_muscle), axis=0))[1: N + 1],
                   label='E tot muscles', color='magenta', linewidth=1)
    axs[1, 1].plot(t_axis[1:-1], differentiate(np.sum(np.array([state.Ec[1:], state.Ep[1:]]), axis=0)),
                   label='P system',
                   color='white', linewidth=1)
    axs[1, 1].plot(t_axis, np.sum(np.array(state.l_p_muscle), axis=0), label='P tot muscles', color='magenta',
                   linewidth=1)

    axs[0, 0].set_title('system energy')
    axs[0, 1].set_title('muscle energy')
    axs[1, 0].set_title('mechanical energy and total muscle energy')
    axs[1, 1].set_title('system power and total muscle power')

    axs[0, 0].set_xlabel('time (s)')
    axs[0, 1].set_xlabel('time (s)')
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 1].set_xlabel('time (s)')

    axs[0, 0].set_ylabel('Energy (J)')
    axs[0, 1].set_ylabel('Energy (J)')
    axs[1, 0].set_ylabel('Energy (J)')
    axs[1, 1].set_ylabel('Power (W)')

    fig.patch.set_facecolor(background_color / 255)

    for ax in fig.get_axes():
        ax.legend()
        ax.set_facecolor(background_color / 255)
    plt.show()


def plot_phase_portrait(state: State) -> None:
    """
    Plots the phase portrait (angle vs angular velocity) for each bone.
    """
    N = len(state.bones[0].l_theta) - 2

    for bone in state.bones:
        plt.plot(bone.l_theta[1:N], differentiate(bone.l_theta[2:N + 2]), linewidth=1)
    plt.show()


def plot_efforts(state: State) -> None:
    """
    Plots the muscle efforts and the components of the Q cost function.
    """
    fig, axs = plt.subplots(2, 3)

    N = len(state.bones[0].l_theta) - 2

    t_axis = [i * t for i in range(N)]

    calves, quadriceps, hamstrings, low_back, lats = state.muscles
    axs[0, 0].plot(t_axis[1:], [state.l_efforts[i][low_back.index] for i in range(1, len(state.l_efforts))],
                   label=low_back.name,
                   color='#8dd3c7', linewidth=1)
    axs[0, 1].plot(t_axis[1:], [state.l_efforts[i][quadriceps.index] for i in range(1, len(state.l_efforts))],
                   label=quadriceps.name, color='#feffb3', linewidth=1, )
    axs[0, 2].plot(t_axis[1:], [state.l_efforts[i][lats.index] for i in range(1, len(state.l_efforts))],
                   label=lats.name, color='#bfbbd9', linewidth=1)
    axs[1, 0].plot(t_axis[1:], [state.l_efforts[i][hamstrings.index] for i in range(1, len(state.l_efforts))],
                   label=hamstrings.name, color='#fa8174', linewidth=1)
    axs[1, 1].plot(t_axis[1:], [state.l_efforts[i][calves.index] for i in range(1, len(state.l_efforts))],
                   label=calves.name,
                   color='#81b1d2', linewidth=1)

    name = {0: 'Q', 1: 'shoulder height', 2: 'gravity center pos', 3: 'gravity center speed',
            4: 'g', 5: 'dg/dt'}

    for j in range(len(state.l_Q[0])):
        axs[1, 2].plot(t_axis[1:], [state.l_Q[i][j] for i in range(1, len(state.l_Q))], label=name[j], linewidth=1)

    axs[0, 0].set_xlabel('time (s)')
    axs[0, 1].set_xlabel('time (s)')
    axs[0, 2].set_xlabel('time (s)')
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 1].set_xlabel('time (s)')
    axs[1, 2].set_xlabel('time (s)')

    axs[0, 0].set_ylabel('normalized force')
    axs[0, 1].set_ylabel('normalized force')
    axs[0, 2].set_ylabel('normalized force')
    axs[1, 0].set_ylabel('normalized force')
    axs[1, 1].set_ylabel('normalized force')

    fig.patch.set_facecolor(background_color / 255)

    for ax in fig.get_axes():
        ax.legend()
        ax.set_facecolor(background_color / 255)

    axs[1, 2].set_yscale('log')

    plt.show()


def plot_Q(state: State) -> None:
    """
    Plots the Q cost function and its components over time using a logarithmic scale.
    """
    fig, ax = plt.subplots(1, 1)

    N = len(state.bones[0].l_theta) - 2

    t_axis = [i * t for i in range(N)]

    name = {0: 'Q', 1: 'shoulder height', 2: 'gravity center pos', 3: 'gravity center speed',
            4: 'g', 5: 'dg/dt'}

    fig.patch.set_facecolor(background_color / 255)

    for j in range(len(state.l_Q[0])):
        ax.plot(t_axis[1:], [state.l_Q[i][j] for i in range(1, len(state.l_Q))], label=name[j], linewidth=1)
        ax.set_facecolor(background_color / 255)
    ax.set_yscale('log')

    ax.legend()
    plt.show()
