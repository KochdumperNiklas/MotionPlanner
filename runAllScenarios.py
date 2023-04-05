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
import time

from vehicle.vehicleParameter import vehicleParameter
from auxiliary.createVideo import createVideo
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton

import warnings
warnings.filterwarnings("ignore")

VIDEO = False           # create videos for all scenarios that can be solved

# parameter for the car
param = vehicleParameter()

# get all available CommonRoad scenarios
path = 'scenarios'
files = [f for f in listdir(path) if isfile(join(path, f))]
data = []

# loop over all scenarios
for f in files:

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    try:
        start_time = time.time()
        plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)
        x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj)
        comp_time = time.time() - start_time
        if x is None:
            print(f + ': failed')
        else:
            final_time = x.shape[1] * param['time_step']
            comp_time = comp_time / final_time
            data.append([f, comp_time])
            if VIDEO:
                createVideo(f, scenario, planning_problem, param, x)
    except:
        print(f + ': failed')

# save computation time in .csv file
np.savetxt('computation_time.csv', np.asarray(data), delimiter=",", fmt="%s")

# display results
print(' ')
print('success rate: ' + str(len(data)/len(files) * 100) + "%")