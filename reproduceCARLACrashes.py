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
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.planning_problem import PlanningProblemSet

import warnings
warnings.filterwarnings("ignore")

# load parameter for the car
param = vehicleParameter()
param['a_max'] = 5
param['wheelbase'] = 2.7

# load maneuver automaton
filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
MA = pickle.load(filehandler)

# load the CommonRoad scenario
filehandler = open('/media/niklas/USB DISK/fails/CARLAdataStandstillTrafficLight.obj', 'rb')
data = pickle.load(filehandler)
scenario = data['scenario']
planning_problem = data['problem']

# high-level planner: decides on which lanelets to be at which points in time
plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)

# low-level planner: plans a concrete trajectory for the high-level plan
#x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj, MA)

# debug
tmp = list(planning_problem.planning_problem_dict.values())[0].initial_state
x = np.expand_dims(np.array([tmp.position[0], tmp.position[1], tmp.velocity, tmp.orientation]), axis=1)
x = x * np.ones((1, 30))

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
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.pause(0.1)