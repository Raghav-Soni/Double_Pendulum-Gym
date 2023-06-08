import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class dp_plot:

    def __init__(self, l1, l2, dt):
        self.l1 = l1
        self.l2 = l2
        self.dt = dt

        self.init = [0.0, 0.0, 0.0, 0.0]

        self.r = 0.05

        self.fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
        self.ax = self.fig.add_subplot(111)


    def animate_step(self, x):
        x1 = self.l1 * np.sin(x[0])
        y1 = -self.l1 * np.cos(x[0])

        x2 = x1 + self.l2 * np.sin(x[1])
        y2 = y1 - self.l2 * np.cos(x[1])

        self.ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')

        c0 = Circle((0, 0), self.r/2, fc='k', zorder=10)
        c1 = Circle((x1, y1), self.r, fc='b', ec='b', zorder=10)
        c2 = Circle((x2, y2), self.r, fc='r', ec='r', zorder=10)
        self.ax.add_patch(c0)
        self.ax.add_patch(c1)
        self.ax.add_patch(c2)

        # Centre the image on the fixed anchor point, and ensure the axes are equal
        self.ax.set_xlim(-self.l1-self.l2-self.r, self.l1+self.l2+self.r)
        self.ax.set_ylim(-self.l1-self.l2-self.r, self.l1+self.l2+self.r)
        self.ax.set_aspect('equal', adjustable='box')
        plt.pause(0.0001)
        plt.axis('off')
        plt.cla()
