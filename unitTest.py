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
from lowLevelPlannerNew import lowLevelPlannerNew

# parameter for the car
param = {}
param['length'] = 4.3                   # length of the car
param['width'] = 1.7                    # width of the car
param['wheelbase'] = 2.3                # length of the wheelbase
param['a_max'] = 9                      # maximum acceleration
param['s_max'] = np.deg2rad(24.0)       # maximum steering angle

# scenarios that are working
path = 'scenarios'
files = ["ZAM_Zip-1_19_T-1.xml", "ZAM_Tutorial-1_1_T-1.xml", "FRA_Sete-1_1_T-1.xml", "BEL_Putte-4_2_T-1.xml",
         "ESP_Monzon-2_1_T-1.xml", "BEL_Zwevegem-2_3_T-1.xml"]

# loop over all scenarios
for f in files:

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    plan, vel, space, ref_traj = highLevelPlannerNew(scenario, planning_problem, param)
    x, u = lowLevelPlannerNew(scenario, planning_problem, param, plan, vel, space, ref_traj)
    print(f + ': success')
