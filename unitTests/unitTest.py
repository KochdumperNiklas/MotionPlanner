from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from maneuverAutomaton.loadAROCautomaton import loadAROCautomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from unitTests.testFreeSpace import test_free_space
from auxiliary.collisionChecker import collisionChecker


# load parameter for the car
param = vehicleParameter()

# load maneuver automaton
MA = loadAROCautomaton()

# scenarios that are working
path = 'scenarios'
files = ["ZAM_Zip-1_19_T-1.xml", "ZAM_Tutorial-1_1_T-1.xml", "FRA_Sete-1_1_T-1.xml", "BEL_Zwevegem-2_3_T-1.xml",
         "PRI_Barceloneta-4_3_T-1.xml", "BEL_Nivelles-19_1_T-1.xml", "BEL_Zaventem-1_2_T-1.xml",
         "BEL_Zaventem-4_1_T-1.xml"]

# loop over all scenarios
for f in files:

    success = True

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    plan, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)
    x, u, controller = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, deepcopy(space), ref_traj, MA)

    # check if the driving corridor computed by the decision module intersects an obstacle
    try:
        test_free_space(scenario, space)
    except:
        success = False
        print(f + ': failed due to intersection of the driving corridor with an obstacle')

    # check if the planned trajectory is intersection-free
    if x is not None and not collisionChecker(scenario, x, param):
        success = False
        print(f + ': failed due to intersection of with an obstacle')

    if success:
        print(f + ': success')
