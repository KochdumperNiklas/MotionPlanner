from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import cv2

def createVideo(file, scenario, planning_problem, param, x):

    # create video folder if it does not exist yet
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isdir(os.path.join(path, 'results')):
        os.mkdir(os.path.join(path, 'results'))

    if not os.path.isdir(os.path.join(path, 'results', 'videos')):
        os.mkdir(os.path.join(path, 'results', 'videos'))

    # initialize video writer
    fps = 10
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(os.path.join(path, 'results', 'videos', file[0:-4] + '.mp4'), fourcc, fps, (2500, 1500))

    # create figure
    plt.figure(figsize=(25, 15))

    # loop over all time steps
    for i in range(0, x.shape[1]):

        plt.cla()
        rnd = MPRenderer()
        canvas = FigureCanvasAgg(rnd.f)

        # draw traffic scenario
        rnd.draw_params.time_begin = i
        scenario.draw(rnd)
        planning_problem.draw(rnd)

        # draw ego vehicle
        settings = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='#d95558')
        r = Rectangle(length=param['length'], width=param['width'], center=np.array([x[0, i], x[1, i]]), orientation=x[3, i])
        r.draw(rnd, settings)

        # formatting
        rnd.render()
        plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
        plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.title(file[0:-4], fontsize=40, y=1.03)

        # extract image from figure
        canvas.draw()
        buf = canvas.buffer_rgba()
        frame = np.asarray(buf)

        # add frame to video
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()