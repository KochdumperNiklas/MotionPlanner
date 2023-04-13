from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from copy import deepcopy

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization
from auxiliary.prediction import prediction

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route

import warnings
warnings.filterwarnings("ignore")

HORIZON = 30
REPLAN = 5


# select the CommonRoad scenario that should be solved
file = "Town01.xml"

# load parameter for the car
param = vehicleParameter()

# load maneuver automaton
filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
MA = pickle.load(filehandler)

# load the CommonRoad scenario
scenario, planning_problem = CommonRoadFileReader(os.path.join('auxiliary', file)).open()

# define planning problem
#goal_state = CustomState(time_step=Interval(1, 2), position=scenario.lanelet_network.lanelets[1].polygon)
#goal_region = GoalRegion([goal_state], lanelets_of_goal_position={0: [9]})
#goal_region = GoalRegion([goal_state], lanelets_of_goal_position=None)
#initial_state = InitialState(position=np.array([396.5, -30]), velocity=10, orientation=np.pi/2, yaw_rate=0, slip_angle=0, time_step=0)
#planning_problem = PlanningProblemSet([PlanningProblem(1, initial_state, goal_region)])

# plan route
planning_problem = list(planning_problem.planning_problem_dict.values())[0]
route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
candidate_holder = route_planner.plan_routes()

list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
route = candidate_holder.retrieve_first_route()
#visualize_route(route, draw_route_lanelets=True, draw_reference_path=False, size_x=6)

# initialization
state = planning_problem.initial_state
x0 = np.concatenate((state.position, np.array([state.velocity, state.orientation])))
#x0 = np.concatenate((state.position, np.array([5, state.orientation])))
lanelet = route.list_ids_lanelets[0]
cnt_init = 0
plt.figure(figsize=(25, 10))

# loop until car arrived at destination
while lanelet != route.list_ids_lanelets[-1]:

    # predict future positions of the surrounding traffic participants
    vehicles = [{'width': 2, 'length': 5, 'x': 380, 'y': -2, 'velocity': 10, 'orientation': 0}]
    vehicles.append({'width': 2, 'length': 5, 'x': 320, 'y': -2, 'velocity': 10, 'orientation': 0})
    #vehicles.append({'width': 2, 'length': 5, 'x': 396.5, 'y': -10, 'velocity': 0, 'orientation': np.pi / 2})

    scenario_ = prediction(vehicles, deepcopy(scenario), HORIZON)

    # create motion planning problem
    goal_states = []
    lanelets_of_goal_position = {}

    dist_max = x0[2] * scenario.dt * HORIZON + 0.5 * param['a_max'] * (scenario.dt * HORIZON)**2
    dist = 0

    for i in range(cnt_init, len(route.list_ids_lanelets)):
        goal_id = route.list_ids_lanelets[i]
        goal_lane = scenario.lanelet_network.find_lanelet_by_id(goal_id)
        goal_states.append(CustomState(time_step=Interval(HORIZON, HORIZON), position=goal_lane.polygon))
        lanelets_of_goal_position[len(goal_states)-1] = [goal_id]
        if i > cnt_init:
            dist = dist + goal_lane.distance[-1]
        if dist > dist_max:
            break

    goal_region = GoalRegion(goal_states, lanelets_of_goal_position=lanelets_of_goal_position)
    initial_state = InitialState(position=x0[0:2], velocity=x0[2], orientation=x0[3], yaw_rate=0, slip_angle=0, time_step=0)
    planning_problem = PlanningProblemSet([PlanningProblem(1, initial_state, goal_region)])

    # solve motion planning problem
    plan, vel, space, ref_traj = highLevelPlanner(scenario_, planning_problem, param)
    x, u = lowLevelPlannerManeuverAutomaton(scenario_, planning_problem, param, plan, vel, space, ref_traj, MA)

    # visualization
    for i in range(REPLAN):
        plt.cla()
        rnd = MPRenderer()

        rnd.draw_params.time_begin = i
        rnd.draw_params.time_end = HORIZON
        rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.draw_arrow = False
        rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.radius = 0
        scenario.draw(rnd)
        planning_problem.draw(rnd)

        # plot prediction for the other vehicles
        for d in scenario_.dynamic_obstacles:
            for j in range(len(d.prediction.trajectory.state_list), 0, -1):
                s = d.prediction.trajectory.state_list[j-1]
                if s.time_step >= i:
                    if s.time_step == i:
                        settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='#1d7eea')
                    else:
                        settings = ShapeParams(opacity=0.2, edgecolor="k", linewidth=0.0, facecolor='#1d7eea')
                    r = Rectangle(length=d.obstacle_shape.length, width=d.obstacle_shape.width, center=s.position,
                                  orientation=s.orientation)
                    r.draw(rnd, settings)

        # plot planned trajectory
        for j in range(x.shape[1]-1, -1, -1):
            if j >= i:
                if j == i:
                    settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='r')
                else:
                    settings = ShapeParams(opacity=0.2, edgecolor="k", linewidth=0.0, facecolor='r')
                r = Rectangle(length=param['length'], width=param['width'], center=np.array([x[0, j], x[1, j]]),
                              orientation=x[3, j])
                r.draw(rnd, settings)

        rnd.render()
        plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
        plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.pause(0.1)

    # update state and lanelet
    x0 = x[:, REPLAN]
    lane = scenario.lanelet_network.find_lanelet_by_id(lanelet)

    if not lane.polygon.shapely_object.contains(Point(x0[0], x0[1])):

        for i in range(cnt_init, len(route.list_ids_lanelets)):

            l = scenario.lanelet_network.find_lanelet_by_id(route.list_ids_lanelets[i])

            if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
                cnt_init = i
                lanelet = l.lanelet_id



# high-level planner: decides on which lanelets to be at which points in time
#plan, vel, space, ref_traj = highLevelPlanner(scenario, planning_problem, param)