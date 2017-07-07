

from os import sep

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def plot_consumer_result(result, grid=None, title='', path=None):
    for l in result:
        plt.plot(grid, l)
    plt.title(title)
    plt.xlabel("time, $t$")
    plt.ylabel("position of process")
    if path:
        plt.savefig(path + sep + title + '.pdf')
        plt.close()


def plot_timewave_result(result, title='', path=None):
    # Plot a basic wireframe.
    x, y, z = result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap=cm.coolwarm)
    plt.title(title)
    if path:
        plt.savefig(path + sep + title + '.pdf')
        plt.close()

    # '.' + sep + 'pdf'

    # closing a figure does not work for mac
    #
    # def close_event():
    #     plt.close('all')
    # class close_event(object):
    #     def __init__(self):
    #         self.first = True
    #     def __call__(self):
    #         if self.first:
    #             self.first = False
    #         else:
    #             plt.close(0)
    # fig = plt.figure()
    # timer = fig.canvas.new_timer(interval=3000)
    # timer.add_callback(close_event)
    # timer.start()
