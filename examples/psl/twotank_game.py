'''
Graphical pygame interface for controlling the psl.Nonautonomous.TwoTank environment wrappen in PslGym
Good for testing the environment and difficulty of control problem
Press space to enter menu and change control actions
Press escape to exit game
'''


import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import matplotlib
matplotlib.use("Agg")
import pygame
from pygame.locals import KEYDOWN, KEYUP, K_ESCAPE, K_SPACE
import pygame_menu
from pygame_menu.baseimage import BaseImage
from pygame_menu.themes import Theme
import sys

from neuromancer.psl import pslgym
from neuromancer.psl.pslgym import PslGym

shape = (1000,800)

screen = pygame.display.set_mode(shape)
black = (0, 0, 0)

pygame.init()
red_transparent = (128,0,0,128)
mytheme = Theme(background_color=red_transparent)
menu = pygame_menu.Menu(width=shape[0], height=shape[1], title='Menu',mouse_enabled=True, mouse_motion_selection=True, mouse_visible=True, mouse_visible_update=True, theme=mytheme)

menu_up = False
done = False
def exit_game():
    global done, menu_up
    done = True

menu.add.button('Exit Game', exit_game)
# make slider for each action
names = ['pump','nozzle']
for i in range(2):
    menu.add.range_slider(names[i], .5, (0, 1), .01,
                        rangeslider_id='range_slider'+str(i),
                        value_format=lambda x: str(np.round(x,2)))

def plot(env):
    (x, rs), ref = env.get_trace(), env.ref_t
    t = np.arange(0,env.nsim)
    fig, axs = plt.subplots(env.nx+1, 1)
    for i, ax in enumerate(axs[:-1]):
        axs[i].plot(x[:,i],color='blue')
        axs[i].plot(t, ref[:,i], color='red',linestyle='dashed')
    axs[-1].plot(rs, color='green')
    title = f'Pump {np.round(a[0],2)}, Nozzle {np.round(a[1],2)}, Current Reward {np.round(rs[-1],2)}'
    plt.suptitle(title)
    # plt.show()
    return fig, axs

env = PslGym(env_str='TwoTank', reward_class=pslgym.MSE)
env.reset()
a = np.array([.5,.5])

while not done:

    for i in range(2):
        a[i] = menu.get_widget('range_slider'+str(i)).get_value()
    if not menu_up:
        s, reward, terminated, truncated, info = env.step(a)
    fig, axs = plot(env)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0,0))
    plt.close(fig)
    events = pygame.event.get()
    for event in events:
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                if menu_up:
                    menu.disable()
                    menu_up = False
                    screen.fill(black)
                else:
                    done = True
            elif event.key == K_SPACE:
                if menu_up:
                    menu_up = False
                    menu.disable()
                    screen.fill(black)
                else:
                    menu_up = True
                    menu.enable()
                    
    if menu_up:
        menu.update(events)
        menu.draw(screen)
    pygame.display.flip()
    
pygame.quit()
