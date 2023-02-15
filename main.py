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

L = 4.3         # length of the car
W = 1.7         # width of the car

# select the CommonRoad scenario that should be solved
file = "ZAM_Zip-1_19_T-1.xml"

# load the CommonRoad scenario
scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open()

# high-level planner: decides on which lanelets to be at which points in time
plan, space, space_xy = highLevelPlanner(scenario, planning_problem)

# low-level planner: plans a concrete trajectory for the high-level plan
x, u = lowLevelPlanner(planning_problem, plan, space, space_xy)

# visualization
plt.figure(figsize=(25, 10))

for i in range(0, x.shape[1]):
    plt.cla()
    rnd = MPRenderer()

    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    planning_problem.draw(rnd)

    param = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='r')
    r = Rectangle(length=L, width=W, center=np.array([x[0, i], x[1, i]]), orientation=x[3, i])
    r.draw(rnd, param)

    rnd.render()
    plt.pause(0.1)
