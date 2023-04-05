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

from vehicle.vehicleParameter import vehicleParameter
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton

import warnings
warnings.filterwarnings("ignore")

# parameter for the car
param = vehicleParameter()

# get all available CommonRoad scenarios
path = 'scenarios'
files = [f for f in listdir(path) if isfile(join(path, f))]

# loop over all scenarios
for f in files:

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    try:
        plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)
        x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj)
        if x is None:
            print(f + ': failed')
        #print(f + ': success')
    except:
        print(f + ': failed')
