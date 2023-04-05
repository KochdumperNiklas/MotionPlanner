from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from vehicle.vehicleParameter import vehicleParameter
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization

import warnings
warnings.filterwarnings("ignore")

# select the CommonRoad scenario that should be solved
file = "ZAM_Zip-1_19_T-1.xml"
file = "ZAM_Tutorial-1_1_T-1.xml"
#file = "FRA_Sete-1_1_T-1.xml"
#file = "BEL_Putte-4_2_T-1.xml"
#file = "ESP_Monzon-2_1_T-1.xml"

# load parameter for the car
param = vehicleParameter()

# load the CommonRoad scenario
scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open()

# high-level planner: decides on which lanelets to be at which points in time
plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)

# low-level planner: plans a concrete trajectory for the high-level plan
x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj)

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
    plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
    plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
    plt.pause(0.1)
