from ManeuverAutomaton import ManeuverAutomaton
import numpy as np
import pickle
import scipy
from copy import deepcopy
from shapely.affinity import affine_transform
from shapely.affinity import translate
from shapely.geometry import Point
import matplotlib.pyplot as plt

def lowLevelPlannerNew(scenario, planning_problem, param, plan, space, vel, space_all, ref_traj):
    """plan a concrete trajectory for the given high-level plan using a maneuver automaton"""

    # load the maneuver automaton
    filehandler = open('maneuverAutomaton.obj', 'rb')
    MA = pickle.load(filehandler)

    # construct initial state
    x = np.expand_dims(np.array([param['x0'][0], param['x0'][1], param['v_init'], param['orientation']]), 1)

    # initialize queue for the search problem
    ind = MA.velocity2primitives(param['v_init'])
    node = Node([], x, 0)
    queue = []

    for i in ind:
        queue.append(expand_node(node, MA.primitives[i], ref_traj))

    # loop until goal set is reached or queue is empty
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = queue.pop(0)

        # check if motion primitive is collision free
        if collision_check(node, space_all, vel, param):

            # check if goal set has been reached
            if goal_check(node, param):
                return node.x, extract_control_inputs(node)

            # construct child nodes
            for i in node.primitives[-1].successors:
                queue.append(expand_node(node, MA.primitives[i], ref_traj))

    return None, None


def collision_check(node, space, vel, param):
    """check if a motion primitive is collision free"""

    # check if the maximum time is exceeded
    ind = node.x.shape[1] - node.primitives[-1].x.shape[1]

    if ind > param['goal_time_end']:
        return False

    # check if the final velocity is inside the valid range
    index = min(node.x.shape[1]-1, len(vel)-1)

    v_final = node.x[2, index]

    if v_final <= vel[index][0] or v_final >= vel[index][1]:
        return False

    # get state at the before the last primitive
    x = node.x[:, ind]

    # loop over all time steps
    for o in node.primitives[-1].occ:
        time = int(o['time']/param['time_step'])
        pgon = affine_transform(o['space'], [np.cos(x[3]), -np.sin(x[3]), np.sin(x[3]), np.cos(x[3]), x[0], x[1]])
        if ind + time < len(space) and not space[ind + time].contains(pgon):
            return False

    return True

def goal_check(node, param):
    """check if the goal set has been reached"""

    # get index of the state at before the last primitive
    ind = node.x.shape[1] - node.primitives[-1].x.shape[1]

    # loop over all time steps
    for i in range(ind, node.x.shape[1]):
        if param['goal_time_start'] <= i <= param['goal_time_end']:
            p = Point(node.x[0, i], node.x[1, i])
            if param['goal_set'] is None or param['goal_set'].contains(p):
                return True

    return False

def transform_free_space(space, x, time, length):
    """transform the free space from the global to the local coordinate system"""

    # extract relevant time steps
    space_new = deepcopy(space[time:time+length])

    # loop over all time steps
    for i in range(len(space_new)):
        pgon = translate(space_new[i], -x[0], -x[1])
        space_new[i] = affine_transform(pgon, [np.cos(x[3]), np.sin(x[3]), -np.sin(x[3]), np.cos(x[3]), 0, 0])

    return space_new

def extract_control_inputs(node):
    """construct the sequence of control inputs for the given node"""

    u = []

    for i in range(len(node.primitives)):
        u_new = np.expand_dims(node.primitives[i].u, axis=1) @ np.ones((1, node.primitives[i].x.shape[1]-1))
        if i == 0:
            u = u_new
        else:
            u = np.concatenate((u, u_new), axis=1)


def expand_node(node, primitive, ref_traj):
    """add a new primitive to a node"""

    # add current primitive to the list of primitives
    primitives = node.primitives + [primitive]

    # combine trajectories
    phi = node.x[3, -1]
    T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))
    x_ = T @ primitive.x + np.array([[node.x[0, -1]], [node.x[1, -1]], [0], [phi]])
    x = np.concatenate((node.x[:, :-1], x_), axis=1)

    # compute costs
    if x.shape[1] <= ref_traj.shape[1]:
        ind = x.shape[1] - 1
    else:
        ind = ref_traj.shape[1] - 1

    cost = np.sum((ref_traj[:, ind] - x[0:2, ind])**2) + 1000/len(primitives)

    return Node(primitives, x, cost)


class Node:
    """class representing a node for A*-search"""

    def __init__(self, primitives, x, cost):
        """class constructor"""

        self.primitives = primitives
        self.x = x
        self.cost = cost