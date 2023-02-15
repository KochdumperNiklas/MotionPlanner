from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import numpy as np
import os

from highLevelPlanner import highLevelPlanner
from lowLevelPlanner import lowLevelPlanner

# parameter for the car
param = {}
param['length'] = 4.3                   # length of the car
param['width'] = 1.7                    # width of the car
param['wheelbase'] = 2.3                # length of the wheelbase
param['a_max'] = 9                      # maximum acceleration
param['s_max'] = np.deg2rad(24.0)       # maximum steering angle

# select the CommonRoad scenario that should be solved
file = "ZAM_Zip-1_19_T-1.xml"

# load the CommonRoad scenario
scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open()

# high-level planner: decides on which lanelets to be at which points in time
plan, space, vel = highLevelPlanner(scenario, planning_problem, param)

# low-level planner: plans a concrete trajectory for the high-level plan
x, u = lowLevelPlanner(scenario, planning_problem, param, plan, space, vel)

# visualization
plt.figure(figsize=(25, 10))

for i in range(0, x.shape[1]):
    plt.cla()
    rnd = MPRenderer()

    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    planning_problem.draw(rnd)

    settings = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='r')
    r = Rectangle(length=param['length'], width=param['width'], center=np.array([x[0, i], x[1, i]]), orientation=x[3, i])
    r.draw(rnd, settings)

    rnd.render()
    plt.pause(0.1)
