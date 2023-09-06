import numpy as np
import pickle
import scipy.linalg
from copy import deepcopy
from shapely.affinity import affine_transform
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys
sys.path.append('./maneuverAutomaton/')
from ManeuverAutomaton import ManeuverAutomaton
from loadAROCcontroller import loadAROCcontroller
from maneuverAutomaton.Controller import FeedbackController


def lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, space_all, ref_traj, MA, search_horizon=2):
    """plan a concrete trajectory for the given high-level plan using a maneuver automaton"""

    # check if the maneuver automaton contains occupancy sets for time intervals or time points
    timepoint = True

    if 'starttime' in MA.primitives[1].occ[0]:
        timepoint = False
        space = []
        for i in range(len(space_all)-1):
            space.append(space_all[i].intersection(space_all[i+1]))
        space_all = space

    # construct initial state
    x = np.expand_dims(np.array([param['x0'][0], param['x0'][1], param['v_init'], param['orientation']]), 1)

    x[0, 0] = x[0, 0] - np.cos(x[3, 0]) * param['b']
    x[1, 0] = x[1, 0] - np.sin(x[3, 0]) * param['b']

    # initialize queue for the search problem
    ind = MA.velocity2primitives(param['v_init'])
    node = Node([], x, 0)
    queue = []
    cnt = 0
    fixed = []
    phi = node.x[3, -1]
    T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))

    for i in ind:
        queue.append(expand_node(node, MA.primitives[i], i, ref_traj, fixed, T))

    # loop until goal set is reached or queue is empty
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = queue.pop(0)
        primitive = MA.primitives[node.primitives[-1]]

        # check if motion primitive is collision free
        if collision_check(node, primitive, space_all, param, timepoint):

            # fix motion primitive if planning horizon is reached
            if len(node.primitives) == cnt + search_horizon:
                for i in range(len(queue)):
                    if cnt < len(queue[i].primitives) and queue[i].primitives[cnt] != node.primitives[cnt]:
                        queue[i].cost = queue[i].cost + 1000
                fixed.append(node.primitives[cnt])
                cnt = cnt + 1

            # check if goal set has been reached
            res, ind = goal_check(node, primitive, param)
            if res:
                if timepoint:
                    u = extract_control_inputs(node, MA.primitives)
                    t = param['time_step'] * np.arange(0, x.shape[1] + 1)
                    K = [np.zeros([u.shape[0], x.shape[0]]) for i in range(u.shape[1])]
                    controller = FeedbackController(x, u, t, K)
                    return transform_trajectory(node.x[:, :ind+1], param), u[:, :ind], controller
                else:
                    x, u, controller = simulate_controller(MA, node, x, ind+1, param)
                    return transform_trajectory(x, param), u, controller

            # precompute transformation matrix for speed-up
            phi = node.x[3, -1]
            T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))

            # construct child nodes
            for i in primitive.successors:
                queue.append(expand_node(node, MA.primitives[i], i, ref_traj, fixed, T))

    return None, None, None


def collision_check(node, primitive, space, param, timepoint):
    """check if a motion primitive is collision free"""

    # check if the maximum time is exceeded
    ind = node.x.shape[1] - primitive.x.shape[1]

    time_exceeded = True

    for goal in param['goal']:
        if ind <= goal['time_end']:
            time_exceeded = False
            break

    if time_exceeded:
        return False

    # get state at the before the last primitive
    x = node.x[:, ind]

    # loop over all time steps
    for o in primitive.occ:
        if timepoint:
            time = int(o['time']/param['time_step'])
        else:
            time = int(o['starttime']/param['time_step'])
        if ind + time <= param['steps']:
            pgon = affine_transform(o['space'], [np.cos(x[3]), -np.sin(x[3]), np.sin(x[3]), np.cos(x[3]), x[0], x[1]])
            if ind + time < len(space) and not space[ind + time].contains(pgon):
                return False

    return True

def goal_check(node, primitive, param):
    """check if the goal set has been reached"""

    # get index of the state at before the last primitive
    ind = node.x.shape[1] - primitive.x.shape[1]

    # loop over all time steps
    for i in range(ind, node.x.shape[1]):
        for goal in param['goal']:
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

def expand_node(node, primitive, ind, ref_traj, fixed, T):
    """add a new primitive to a node"""

    # add current primitive to the list of primitives
    primitives = node.primitives + [ind]

    # combine trajectories
    x_ = T @ primitive.x + np.array([[node.x[0, -1]], [node.x[1, -1]], [0], [node.x[3, -1]]])
    x = np.concatenate((node.x[:, :-1], x_), axis=1)

    # compute costs
    ind = x.shape[1] - primitive.x.shape[1]
    if x.shape[1] <= ref_traj.shape[1]:
        index = range(ind, x.shape[1])
    else:
        index = range(ind, ref_traj.shape[1])

    cost = node.cost + np.sum((ref_traj[0:2, index] - x[0:2, index])**2)

    if len(primitives) <= len(fixed) and primitives[-1] != fixed[len(primitives)-1]:
        cost = cost + 1000

    return Node(primitives, x, cost)


class Node:
    """class representing a node for A*-search"""

    def __init__(self, primitives, x, cost):
        """class constructor"""

        self.primitives = primitives
        self.x = x
        self.cost = cost