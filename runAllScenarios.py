from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import os
import numpy as np
from os import listdir
from os.path import isfile, join

from highLevelPlannerNew import highLevelPlannerNew
from highLevelPlanner import highLevelPlanner
from lowLevelPlanner import lowLevelPlanner

# parameter for the car
param = {}
param['length'] = 4.3                   # length of the car
param['width'] = 1.7                    # width of the car
param['wheelbase'] = 2.3                # length of the wheelbase
param['a_max'] = 9                      # maximum acceleration
param['s_max'] = np.deg2rad(24.0)       # maximum steering angle

# get all available CommonRoad scenarios
path = 'scenarios'
files = [f for f in listdir(path) if isfile(join(path, f))]

# loop over all scenarios
for f in files:

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    try:
        plan, space, vel = highLevelPlannerNew(scenario, planning_problem, param)
        x, u = lowLevelPlanner(scenario, planning_problem, param, plan, space, vel)
        print(f + ': success')
    except:
        print(f + ': failed')
