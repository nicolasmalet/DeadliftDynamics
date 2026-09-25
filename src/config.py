import numpy as np

g = 9.81  # gravity constant of Earth at the sea level
bar_mass = 175  # mass of the deadlift barr

t = 1e-3  # time for euler's method, the lower the time, the better the approximation

# You can change the following variables depending on what you want to see

review = True

plot_m = False  # plot movement
plot_e = True  # plot energies
plot_p = False  # plot phase portrait
plot_eff = True  # plot efforts
plot_Q_function = False  # plot the Q function
show_model = True  # show the system
show_ground = True  # draw the ground
show_bar = True  # draw the deadlift bar
show_time = True  # show the timer
show_gravity_center = False  # show the center of gravity of the entire system

background_color = np.array([27, 34, 26])
ground_color = (50, 50, 50)
bone_color = (255, 255, 240)
muscle_color = (255, 0, 0)
r_head = 0.11
r_bar = 0.225

screen_size_x = 950
screen_size_y = 950
pixel_per_meter = 3176  # the ratio of my screen
ratio_screen_reality = 0.17

focus = [0, 0.76]  # camera focus
