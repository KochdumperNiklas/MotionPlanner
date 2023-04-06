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
import pickle

import sys
sys.path.append('./')

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton

import warnings
warnings.filterwarnings("ignore")

# load parameter for the car
param = vehicleParameter()

# load maneuver automaton
filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
MA = pickle.load(filehandler)

# scenarios that are working
path = 'scenarios'
files = ["ZAM_Zip-1_19_T-1.xml", "ZAM_Tutorial-1_1_T-1.xml", "FRA_Sete-1_1_T-1.xml", "BEL_Putte-4_2_T-1.xml",
         "ESP_Monzon-2_1_T-1.xml", "BEL_Zwevegem-2_3_T-1.xml", "PRI_Barceloneta-4_3_T-1.xml",
         "DEU_Flensburg-14_1_T-1.xml", "BEL_Nivelles-19_1_T-1.xml", "BEL_Zaventem-1_2_T-1.xml",
         "BEL_Zaventem-4_1_T-1.xml"]

# loop over all scenarios
for f in files:

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', f)).open()

    # run the motion planner
    plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)
    x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj, MA)
    print(f + ': success')
