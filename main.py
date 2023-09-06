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
from maneuverAutomaton.loadAROCautomaton import loadAROCautomaton
from maneuverAutomaton.loadAROCcontroller import loadAROCcontroller
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization
from src.maneuverAutomatonPlannerStandalone import maneuverAutomatonPlannerStandalone
from auxiliary.prediction import prediction
from auxiliary.overlappingLanelets import overlappingLanelets
from auxiliary.polygonLaneletNetwork import polygonLaneletNetwork
from auxiliary.collisionChecker import collisionChecker
from auxiliary.simulation import simulate_disturbed_controller
from unitTests.testFreeSpace import test_free_space

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

# select the CommonRoad scenario that should be solved
file = "ZAM_Zip-1_19_T-1.xml"
#file = "ZAM_Tutorial-1_1_T-1.xml"
#file = "FRA_Sete-1_1_T-1.xml"
file = "BEL_Putte-4_2_T-1.xml"
#file = "ESP_Monzon-2_1_T-1.xml"
#file = "DEU_Backnang-9_1_T-1.xml"
#file = "ESP_SantBoideLlobregat-6_2_T-1.xml"
#file = "ZAM_Tjunction-1_307_T-1.xml"
#file = "USA_Lanker-1_5_T-1.xml"
#file = "ARG_Carcarana-10_4_T-1.xml"
#file = "DEU_Cologne-63_5_I-1.cr.xml"
#file = "DEU_Aachen-3_2_I-1.cr.xml"
#file = "DEU_Aachen-23_7_I-1.cr.xml"
#file = "DEU_Cologne-40_11_I-1.cr.xml"
#file = "DEU_Aachen-29_14_I-1.cr.xml"

#file = "ESP_SantBoideLlobregat-6_2_T-1.xml"
#file = "RUS_Bicycle-9_1_T-1.xml"
#file = "ZAM_Urban-7_1_S-1.xml"
#file = "DEU_Guetersloh-17_2_T-1.xml"
#file = "BEL_Putte-11_2_T-1.xml"
#file = "DEU_Guetersloh-16_2_T-1.xml"
#file = "ESP_SantBoideLlobregat-6_2_T-1.xml"

#file = "ZAM_HW-1_1_S-1.xml"
#file = "USA_US101-7_1_T-1.xml"
#file = "ESP_Inca-7_1_T-1.xml"
#file = "RUS_Bicycle-12_1_T-1.xml"
file = "RUS_Bicycle-9_1_T-1.xml"
file = "BEL_Zwevegem-7_2_T-1.xml"
file = "ZAM_Tjunction-1_32_T-1.xml"
file = "ITA_Foggia-6_1_T-1.xml"
#file = "BEL_Wervik-2_1_T-1.xml"
file = "ZAM_Tjunction-1_493_T-1.xml"
file = "DEU_Flensburg-53_1_T-1.xml"
file = "DEU_Muc-2_1_T-1.xml"
file = "DEU_Guetersloh-16_2_T-1.xml"
file = "BEL_Putte-8_2_T-1.xml"
file = "ITA_Foggia-19_1_T-1.xml"

#file = "USA_US101-13_1_T-1.xml"
#file = "RUS_Bicycle-1_1_T-1.xml"
#file = "USA_Peach-2_1_T-1.xml"

#file = "FRA_MoissyCramayel-3_3_T-1.xml"
#file = "ESP_Berga-6_1_T-1.xml"
#file = "ITA_Siderno-1_4_T-1.xml"
#file = "DEU_Ibbenbueren-8_3_T-1.xml"

#file = "USA_US101-24_1_T-1.xml"
#file = "DEU_Flensburg-42_1_T-1.xml"
#file = "C-DEU_B471-1_1_T-1.xml"
file = "RUS_Bicycle-3_1_T-1.xml"
file = "RUS_Bicycle-9_1_T-1.xml"
file = "DEU_Cologne-40_11_I-1.cr.xml"
file = "DEU_Flensburg-73_1_T-1.xml"


# load parameter for the car
param = vehicleParameter()

# load maneuver automaton
#MA = loadAROCautomaton()

#filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
#MA = pickle.load(filehandler)

# load the CommonRoad scenario
#scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', file)).open(lanelet_assignment=True)

#x, u = maneuverAutomatonPlannerStandalone(scenario, planning_problem, param, MA)

filehandler = open('/media/niklas/FLASHDEVICE/CARLAdata.obj', 'rb')
data = pickle.load(filehandler)
scenario = data['scenario']
planning_problem = data['problem']

"""tmp = list(planning_problem.planning_problem_dict.values())[0]
goal_states = [CustomState(time_step=tmp.goal.state_list[0].time_step)]
goal_region = GoalRegion(goal_states)
planning_problem = PlanningProblemSet([PlanningProblem(tmp.planning_problem_id, tmp.initial_state, goal_region)])"""

"""overlaps = overlappingLanelets(scenario)
tmp = list(planning_problem.planning_problem_dict.values())[0].initial_state
x0 = np.array([tmp.position[0], tmp.position[1], tmp.velocity, tmp.orientation])
scenario = prediction(scenario, 30, x0, overlapping_lanelets=overlaps)"""


# high-level planner: decides on which lanelets to be at which points in time
plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param, desired_velocity='init', compute_free_space=False)

#x = ref_traj
#test = test_free_space(scenario, space)


# low-level planner: plans a concrete trajectory for the high-level plan
x, u, controller = lowLevelPlannerOptimization(scenario, planning_problem, param, plan, vel, space, ref_traj, feedback_control=True)
#x = x[[0, 1, 3, 4], :]
#x, u = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, plan, vel, space, ref_traj, MA)

"""diff_x = np.expand_dims(np.array([0.1, 0.1, 0.2, 0.05]), axis=1)
dist = np.array([0.5, 0.05])

for i in range(10):
    x0 = x[:, [0]] + np.random.uniform(low=-diff_x, high=diff_x)
    x_, u_ = simulate_disturbed_controller(controller, x0, param['steps']*param['time_step'], dist, param)
    plt.plot(x_[0, :], x_[1, :], 'r')

plt.plot(x[0, :], x[1, :], 'b')"""

filehandler = open('trajectory.obj', 'wb')
pickle.dump({'x': x, 'u': u}, filehandler)

test = collisionChecker(scenario, x, param)
print(test)

# visualization
plt.figure(figsize=(25, 10))

for i in range(0, x.shape[1]):
    plt.cla()
    rnd = MPRenderer()

    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    planning_problem.draw(rnd)

    settings = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='r')
    r = Rectangle(length=param['length'], width=param['width'], center=np.array([x[0, i], x[1, i]]), orientation=np.mod(x[3, i], 2*np.pi))
    r.draw(rnd, settings)

    rnd.render()
    plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
    plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.pause(0.1)
