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
from scipy.io import savemat

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from maneuverAutomaton.loadAROCautomaton import loadAROCautomaton
from auxiliary.createVideo import createVideo
from auxiliary.collisionChecker import collisionChecker
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization
from src.maneuverAutomatonPlannerStandalone import maneuverAutomatonPlannerStandalone

import warnings
warnings.filterwarnings("ignore")

PLANNER = 'HighLevel'   # planner ('HighLevel', 'Automaton', 'AutomatonStandalone' or 'Optimization')
VIDEO = False           # create videos for all scenarios that can be solved
TIMEOUT = 100           # maximum computation time

def solve_scenario(file, return_dict, MA):
    """solve a single motion planning problem"""

    comp_time = 'error'
    collision = 'no collision'
    x = None

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open(lanelet_assignment=True)

    # get final time and tags for the scenario
    steps = 0
    planning_problem_ = list(planning_problem.planning_problem_dict.values())[0]

    for i in range(len(planning_problem_.goal.state_list)):
        goal_state = planning_problem_.goal.state_list[i]
        steps = np.maximum(goal_state.time_step.end - planning_problem_.initial_state.time_step, steps)

    return_dict['final_time'] = steps * scenario.dt
    return_dict['tags'] = ';'.join([t.name for t in scenario.tags])

    # parameter for the car
    param = vehicleParameter()

    # run the motion planner
    try:
        if PLANNER == 'Automaton':
            start_time = time.time()
            plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param, desired_velocity='init')
            x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj, MA)
            comp_time = time.time() - start_time
        elif PLANNER == 'Optimization':
            start_time = time.time()
            plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param, compute_free_space=False,
                                                          desired_velocity='init')
            x, u, _ = lowLevelPlannerOptimization(scenario, planning_problem, param, plan, vel, space, ref_traj)
            comp_time = time.time() - start_time
        elif PLANNER == 'HighLevel':
            start_time = time.time()
            plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param, compute_free_space=False,
                                                          desired_velocity='init')
            comp_time = time.time() - start_time
            x = ref_traj
            u = np.zeros((2, ref_traj.shape[1]))
        elif PLANNER == 'AutomatonStandalone':
            start_time = time.time()
            x, u = maneuverAutomatonPlannerStandalone(scenario, planning_problem, param, MA)
            comp_time = time.time() - start_time

        if x is None:
            print(f + ': failed')
            comp_time = 'failed'
        else:
            comp_time = comp_time / return_dict['final_time']
            savemat(join('solutions', PLANNER, file[:-4]) + '.mat', {'x': x, 'u': u})
            print(f + ': ' + str(comp_time))
            if not collisionChecker(scenario, x, param):
                collision = 'collision'
            elif VIDEO:
                createVideo(f, scenario, planning_problem, param, x)
    except Exception as e:
        comp_time = str(e).replace("\n", "").replace(",", "")
        print(f + ': ' + str(e))

    return_dict['comp_time'] = comp_time
    return_dict['collision'] = collision

if __name__ == "__main__":
    """main entry point"""

    # create folders for storing the solutions
    if not os.path.isdir('solutions'):
        os.mkdir('solutions')

    if not os.path.isdir(join('solutions', PLANNER)):
        os.mkdir(join('solutions', PLANNER))

    # load maneuver automaton
    if PLANNER == 'Automaton' or PLANNER == 'AutomatonStandalone':
        MA = loadAROCautomaton()
    else:
        MA = None

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
        p = multiprocessing.Process(target=solve_scenario, args=(f, return_dict, MA))
        p.start()

        # kill process if the computation time exceeds the maximum
        p.join(TIMEOUT)

        # get results from the process
        if p.is_alive():
            p.terminate()
            p.join()
            comp_time = 'timeout'
            collision = ''
            tags = ''
            final_time = ''
        else:
            comp_time = return_dict['comp_time']
            collision = return_dict['collision']

        tags = return_dict['tags']
        final_time = return_dict['final_time']

        data.append([f, comp_time, collision, tags, final_time])

        if not isinstance(comp_time, str):
            cnt = cnt + 1

    # save computation time in .csv file
    np.savetxt('computation_time_' + PLANNER + '.csv', np.asarray(data), delimiter=",", fmt="%s")

    # display results
    print(' ')
    print('success rate: ' + str(cnt/len(files) * 100) + "%")