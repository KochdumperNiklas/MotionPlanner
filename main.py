from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import numpy as np
import os

from highLevelPlannerNew import highLevelPlannerNew
from highLevelPlanner import highLevelPlanner
from lowLevelPlanner import lowLevelPlanner
from lowLevelPlannerNew import lowLevelPlannerNew
from lowLevelPlannerOptimization import lowLevelPlannerOptimization

# parameter for the car
param = {}
param['length'] = 4.3                   # length of the car
param['width'] = 1.7                    # width of the car
param['wheelbase'] = 2.3                # length of the wheelbase
param['a_max'] = 9                      # maximum acceleration
param['s_max'] = np.deg2rad(24.0)       # maximum steering angle

# select the CommonRoad scenario that should be solved
file = "ZAM_Zip-1_19_T-1.xml"
file = "ZAM_Tutorial-1_1_T-1.xml"
file = "USA_US101-6_2_T-1.xml"
#file = "USA_US101-7_1_T-1.xml"
#file = "ZAM_HW-1_1_S-1.xml"
#file = "FRA_Sete-1_1_T-1.xml"
file = "BEL_Putte-4_2_T-1.xml"

# load the CommonRoad scenario
scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open()

# high-level planner: decides on which lanelets to be at which points in time
plan, space, vel, space_all, ref_traj = highLevelPlannerNew(scenario, planning_problem, param)

# low-level planner: plans a concrete trajectory for the high-level plan
x, u = lowLevelPlannerOptimization(scenario, planning_problem, param, plan, space, vel, space_all, ref_traj)

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
