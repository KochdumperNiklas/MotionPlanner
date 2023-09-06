import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.linalg
from scipy.integrate import solve_ivp

from commonroad.geometry.shape import ShapeGroup
from shapely.geometry import Point
from shapely.affinity import affine_transform

import sys
sys.path.append('./maneuverAutomaton/')
sys.path.append('./auxiliary/')
from ManeuverAutomaton import ManeuverAutomaton
from loadAROCcontroller import loadAROCcontroller
from polygonLaneletNetwork import polygonLaneletNetwork


def maneuverAutomatonPlannerStandalone(scenario, planning_problem, param, MA):
    """solve motion planning problem with a maneuver automaton"""

    # get initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    init_state = planning_problem.initial_state
    x0 = np.array([init_state.position[0], init_state.position[1], init_state.velocity, init_state.orientation])

    # get goal sets
    goal_set = get_goal_set(scenario, planning_problem)

    # get free space for each time step
    space = free_space(scenario, goal_set, x0)

    # check if the maneuver automaton contains occupancy sets for time intervals or time points
    param['timepoint'] = True
    param['steps'] = len(space)
    param['time_step'] = scenario.dt

    if 'starttime' in MA.primitives[1].occ[0]:
        param['timepoint'] = False
        space_tmp = []
        for i in range(len(space) - 1):
            space_tmp.append(space[i].intersection(space[i + 1]))
        space = space_tmp

    # transform initial state to the rear axis of the car
    x = np.expand_dims(x0, axis=1)

    x[0, 0] = x[0, 0] - np.cos(x[3, 0]) * param['b']
    x[1, 0] = x[1, 0] - np.sin(x[3, 0]) * param['b']

    # initialize queue for the search problem
    ind = MA.velocity2primitives(x0[2])
    node = Node([], x, 0)
    queue = []
    phi = node.x[3, -1]
    T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))

    for i in ind:
        queue.append(expand_node(node, MA.primitives[i], i, T, goal_set))

    # loop until goal set is reached or queue is empty
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = queue.pop(0)
        primitive = MA.primitives[node.primitives[-1]]

        # check if motion primitive is collision free
        if collision_check(node, primitive, space, goal_set, param):

            # check if goal set has been reached
            res, ind = goal_check(node, primitive, goal_set, param)

            if res:
                if param['timepoint']:
                    u = extract_control_inputs(node, MA.primitives)
                    t = param['time_step'] * np.arange(0, x.shape[1] + 1)
                    K = [np.zeros([u.shape[0], x.shape[0]]) for i in range(u.shape[1])]
                    controller = FeedbackController(x, u, t, K)
                    return transform_trajectory(node.x[:, :ind + 1], param), u[:, :ind], controller
                else:
                    x, u, controller = simulate_controller(MA, node, x, ind + 1, param)
                    return transform_trajectory(x, param), u, controller

            # precompute transformation matrix for speed-up
            phi = node.x[3, -1]
            T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))

            # construct child nodes
            for i in primitive.successors:
                queue.append(expand_node(node, MA.primitives[i], i, T, goal_set))

    return None, None, None

def get_goal_set(scenario, planning_problem):
    """extract a list of goal sets from the planning problem set"""

    goal = []

    for i in range(len(planning_problem.goal.state_list)):

        goal_state = planning_problem.goal.state_list[i]
        param_ = {}
        shapes = 1

        if hasattr(goal_state, 'position') and isinstance(goal_state.position, ShapeGroup):
            shapes = len(goal_state.position.shapes)
        elif not planning_problem.goal.lanelets_of_goal_position is None:
            shapes = len(list(planning_problem.goal.lanelets_of_goal_position.values())[i])

        for j in range(shapes):

            param_['time_start'] = goal_state.time_step.start - planning_problem.initial_state.time_step
            param_['time_end'] = goal_state.time_step.end - planning_problem.initial_state.time_step

            if hasattr(goal_state, 'position'):

                if isinstance(goal_state.position, ShapeGroup):
                    set = goal_state.position.shapes[j].shapely_object
                else:
                    set = goal_state.position.shapely_object

                param_['space'] = set

                if planning_problem.goal.lanelets_of_goal_position is None:
                    for l in scenario.lanelet_network.lanelets:
                        if l.polygon.shapely_object.intersects(set):
                            param_['lane'] = l.lanelet_id
                            goal.append(deepcopy(param_))
                else:
                    param_['lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[i][j]
                    goal.append(deepcopy(param_))

            else:

                if planning_problem.goal.lanelets_of_goal_position is None:
                    param_['lane'] = None
                    param_['space'] = None
                else:
                    param_['lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[i][j]
                    l = lanelets[param_['lane']]
                    param_['space'] = l.polygon.shapely_object

                goal.append(deepcopy(param_))

    return goal

def free_space(scenario, goal_set, x0):
    """construct the free space for each time step"""

    space = []

    # construct polygon that represents the road network
    road = polygonLaneletNetwork(scenario, safe_only=True, x0=x0)

    # determine number of time steps
    steps = 0
    for goal in goal_set:
        steps = np.maximum(steps, goal['time_end'])

    # loop over all time steps
    for i in range(steps+1):

        space_tmp = deepcopy(road)

        # loop over all obstacles
        for obs in scenario.obstacles:

            pgon = obs.occupancy_at_time(i)

            if pgon is not None:
                pgon = get_shapely_object(pgon.shape)

                if pgon.intersects(road):
                    space_tmp = space_tmp.difference(pgon)

        space.append(space_tmp)

    return space

def get_shapely_object(set):
    """construct the shapely polygon object for the given CommonRoad set"""

    if hasattr(set, 'shapes'):
        pgon = set.shapes[0].shapely_object
        for i in range(1, len(set.shapes)):
            pgon = pgon.union(set.shapes[i].shapely_object)
    else:
        pgon = set.shapely_object

    return pgon

def collision_check(node, primitive, space, goal_set, param):
    """check if a motion primitive is collision free"""

    # check if the maximum time is exceeded
    ind = node.x.shape[1] - primitive.x.shape[1]

    time_exceeded = True

    for goal in goal_set:
        if ind <= goal['time_end']:
            time_exceeded = False
            break

    if time_exceeded:
        return False

    # get state at the before the last primitive
    x = node.x[:, ind]

    # loop over all time steps
    for o in primitive.occ:
        if param['timepoint']:
            time = int(o['time']/param['time_step'])
        else:
            time = int(o['starttime']/param['time_step'])
        if ind + time <= param['steps']:
            pgon = affine_transform(o['space'], [np.cos(x[3]), -np.sin(x[3]), np.sin(x[3]), np.cos(x[3]), x[0], x[1]])
            if ind + time < len(space) and not space[ind + time].contains(pgon):
                return False

    return True

def goal_check(node, primitive, goal_set, param):
    """check if the goal set has been reached"""

    # get index of the state at before the last primitive
    ind = node.x.shape[1] - primitive.x.shape[1]

    # loop over all time steps
    for i in range(ind, node.x.shape[1]):
        for goal in goal_set:
            if goal['time_start'] <= i <= goal['time_end']:
                p = Point(node.x[0, i] + np.cos(node.x[3, i]) * param['b'],
                          node.x[1, i] + np.sin(node.x[3, i]) * param['b'])
                if goal['space'] is None or goal['space'].contains(p):
                    return True, i

    return False, None

def simulate_controller(MA, node, x0, ind, param):

    # load controller object
    controller = loadAROCcontroller(MA, node.primitives, x0[:, 0])

    # system dynamics
    ode = lambda t, x, u: [x[2] * np.cos(x[3]),
                           x[2] * np.sin(x[3]),
                           u[0],
                           x[2] * np.tan(u[1]) / param['wheelbase']]

    # simulate the controlled system
    x = np.zeros((x0.shape[0], ind))
    u = np.zeros((2, ind-1))

    x[:, 0] = x0[:, 0]

    for i in range(ind-1):
        u[:, i] = controller.get_control_input(i*param['time_step'], x[:, [i]])
        sol = solve_ivp(ode, [0, param['time_step']], x[:, i], args=(u[:, i], ), dense_output=True)
        x[:, i+1] = sol.sol(param['time_step'])

    return x, u, controller

def extract_control_inputs(node, primitives):
    """construct the sequence of control inputs for the given node"""

    u = []

    for i in range(len(node.primitives)):
        primitive = primitives[node.primitives[i]]
        u_new = np.expand_dims(primitive.u, axis=1) @ np.ones((1, primitive.x.shape[1]-1))
        if i == 0:
            u = u_new
        else:
            u = np.concatenate((u, u_new), axis=1)

    return u

def transform_trajectory(x, param):
    """transform trajectory from rear-axis to vehicle center"""

    for i in range(x.shape[1]):
        x[0, i] = x[0, i] + np.cos(x[3, i]) * param['b']
        x[1, i] = x[1, i] + np.sin(x[3, i]) * param['b']

    return x

def expand_node(node, primitive, ind, T, goal_set):
    """add a new primitive to a node"""

    # add current primitive to the list of primitives
    primitives = node.primitives + [ind]

    # combine trajectories
    x_ = T @ primitive.x + np.array([[node.x[0, -1]], [node.x[1, -1]], [0], [node.x[3, -1]]])
    x = np.concatenate((node.x[:, :-1], x_), axis=1)

    # compute costs
    cost = np.inf
    for j in range(node.x.shape[1]+1, x.shape[1]):
        for goal in goal_set:
            if not goal['space'] is None:
                cost = min(cost, goal['space'].exterior.distance(Point(x[0, j], x[1, j])))

    return Node(primitives, x, cost)


class Node:
    """class representing a node for A*-search"""

    def __init__(self, primitives, x, cost):
        """class constructor"""

        self.primitives = primitives
        self.x = x
        self.cost = cost

