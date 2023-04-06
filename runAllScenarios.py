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
import multiprocessing
import pickle

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from auxiliary.createVideo import createVideo
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton

import warnings
warnings.filterwarnings("ignore")

VIDEO = False           # create videos for all scenarios that can be solved
TIMEOUT = 100           # maximum computation time

def solve_scenario(file, return_dict):
    """solve a single motion planning problem"""

    comp_time = 'error'

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open()

    # load maneuver automaton
    filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
    MA = pickle.load(filehandler)

    # run the motion planner
    try:
        start_time = time.time()
        plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)
        x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj, MA)
        comp_time = time.time() - start_time
        if x is None:
            print(f + ': failed')
            comp_time = 'failed'
        else:
            final_time = x.shape[1] * param['time_step']
            comp_time = comp_time / final_time
            if VIDEO:
                createVideo(f, scenario, planning_problem, param, x)
    except:
        print(f + ': failed')

    return_dict['comp_time'] = comp_time

if __name__ == "__main__":
    """main entry point"""

    # parameter for the car
    param = vehicleParameter()

    # get all available CommonRoad scenarios
    path = 'scenarios'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    cnt = 0

    # loop over all scenarios
    for f in files:

        # solve the scenario
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=solve_scenario, args=(f, return_dict))
        p.start()

        # kill process if the computation time exceeds the maximum
        p.join(TIMEOUT)

        # get results from the process
        if p.is_alive():
            p.terminate()
            p.join()
            comp_time = 'timeout'
        else:
            comp_time = return_dict['comp_time']

        data.append([f, comp_time])

        if not isinstance(comp_time, str):
            cnt = cnt + 1

    # save computation time in .csv file
    np.savetxt('computation_time.csv', np.asarray(data), delimiter=",", fmt="%s")

    # display results
    print(' ')
    print('success rate: ' + str(cnt/len(files) * 100) + "%")