from commonroad.common.file_reader import CommonRoadFileReader
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from copy import deepcopy

SCENARIO = 'ZAM_Zip-1_19_T-1.xml'

A_MIN = -9
A_MAX = 9
dt = 0.1

def motionPlanner(file):
    """plan a safe motion for the given CommonRoad scenario"""

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(file).open()

    # high-level planner: decides on which lanelets to be at which points in time
    plan = highLevelPlanner(scenario,planning_problem)

    # low-level planner: plans a concrete trajectory for the high-level plan


def highLevelPlanner(scenario, planning_problem):
    """decide on which lanelets to be at all points in time"""

    # extract goal region and initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    goal = planning_problem.goal
    goal_lanelet = list(goal.lanelets_of_goal_position.values())[0][0]

    # crete dictionary that converts from lanelet ID to index in the list
    lanelets = scenario.lanelet_network.lanelets
    id = [l.lanelet_id for l in lanelets]
    id2index = dict(zip(id, [i for i in range(0, len(id))]))

    # compute costs (= number of lane changes to reach goal) for each lanelet
    cost_lane = cost_lanelets(lanelets, goal_lanelet, id2index)

    # compute free space on each lanelet at different times
    free_space = free_space_lanelet(lanelets, scenario.obstacles)

    # determine the high-level plan using the A*-search algorithm
    plan = a_star_search(free_space, cost_lane, planning_problem, lanelets, id2index)


def cost_lanelets(lanelets, goal_id, id2index):
    """assign a cost to each lanelet that is based on the number of required lane changes to reach goal set"""

    # initialize costs
    cost = [np.inf for i in range(0, len(lanelets))]
    cost[id2index[goal_id]] = 0

    # initialize queue with goal lanelet
    queue = []
    queue.append(goal_id)

    # loop until costs for all lanelets have been assigned
    while len(queue) > 0:

        q = queue.pop(0)
        lanelet = lanelets[id2index[q]]

        for s in lanelet.predecessor:
            if cost[id2index[s]] == np.inf:
                cost[id2index[s]] = cost[id2index[q]]
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if cost[id2index[lanelet.adj_left]] == np.inf:
                cost[id2index[lanelet.adj_left]] = cost[id2index[q]] + 1
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if cost[id2index[lanelet.adj_right]] == np.inf:
                cost[id2index[lanelet.adj_right]] = cost[id2index[q]] + 1
                queue.append(lanelet.adj_right)

    return cost

def free_space_lanelet(lanelets, obstacles):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []

    # loop over all lanelets
    for l in lanelets:

        occupied_space = [[] for i in range(0, len(obstacles[0].prediction.occupancy_set))]

        # loop over all obstacles
        for obs in obstacles:

            # loop over all time steps
            for o in obs.prediction.occupancy_set:

                # check if dynamic obstacles occupancy set intersects the lanelet
                if l.polygon.shapely_object.intersects(o.shape.shapely_object):

                    # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                    dist_min, dist_max = projection_lanelet_centerline(l, o.shape.shapely_object)
                    occupied_space[o.time_step-1].append((dist_min, dist_max))

        # unite occupied spaces that belong to the same time step to obtain free space
        free_space = []

        for o in occupied_space:

            if len(o) > 0:

                o.sort(key=lambda i: i[0])
                lower = [i[0] for i in o]
                upper = [i[1] for i in o]

                space = []

                if lower[0] > l.distance[0]:
                    space.append((l.distance[0], lower[0]))

                cnt = 0

                while cnt < len(o)-1:

                    ind = cnt

                    for i in range(cnt, len(o)):
                        if lower[i] < upper[cnt]:
                            ind = i

                    space.append((upper[ind], lower[ind+1]))
                    cnt = ind + 1

                if max(upper) < l.distance[len(l.distance)-1]:
                    space.append((max(upper), l.distance[len(l.distance)-1]))

            else:
                space = [(l.distance[0], l.distance[len(l.distance)-1])]

            free_space.append(space)

        free_space_all.append(free_space)

    return free_space_all

def projection_lanelet_centerline(lanelet, pgon):
    """project a polygon to the center line of the lanelet to determine the occupied space"""

    # intersect polygon with lanelet
    o_int = lanelet.polygon.shapely_object.intersection(pgon)

    vx, vy = o_int.exterior.coords.xy
    V = np.stack((vx, vy))

    # initialize minimum and maximum range of the projection
    dist_max = -np.inf
    dist_min = np.inf

    # loop over all centerline segments
    for i in range(0, len(lanelet.distance) - 1):

        # compute normalized vector representing the direction of the current segment
        diff = lanelet.center_vertices[i + 1, :] - lanelet.center_vertices[i, :]
        diff = np.expand_dims(diff / np.linalg.norm(diff), axis=0)

        # project the vertices of the polygon onto the centerline
        V_ = diff @ (V - np.transpose(lanelet.center_vertices[[i], :]))

        # update ranges for the projection
        dist_max = max(dist_max, max(V_[0]) + lanelet.distance[i])
        dist_min = min(dist_min, min(V_[0]) + lanelet.distance[i])

    return dist_min, dist_max

def is_intersecting(range1, range2):
    """check if two ranges intersect"""

    if range1[0] <= range2[0]:
        if range2[0] <= range1[1]:
            return True
    else:
        if range1[0] <= range2[1]:
            return True

    return False

def a_star_search(free_space, cost, planning_problem, lanelets, id2index):
    """determine optimal lanelet for each time step using A*-search"""

    # get goal time and set
    goal = planning_problem.goal
    goal_id = list(goal.lanelets_of_goal_position.values())[0][0]
    goal_state = goal.state_list[0]

    goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[id2index[goal_id]],
                                                                     goal_state.position.shapes[0].shapely_object)
    goal_space = (goal_space_start, goal_space_end)

    # get lanelet and space for the initial point
    x0 = planning_problem.initial_state.position

    for l in lanelets:
        if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = l.lanelet_id

    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[id2index[x0_id]], Polygon([(x0[0]-0.1,
                                  x0[1]-0.1), (x0[0]-0.1, x0[1]+0.1), (x0[0]+0.1, x0[1]+0.1), (x0[0]+0.1, x0[1]-0.1)]))

    x0_space = 0.5 * (x0_space_start + x0_space_end)

    # initialize the frontier
    frontier = []
    frontier.append(Node([x0_id], [(x0_space, x0_space)], cost[id2index[x0_id]]))

    # loop until a solution has been found
    while len(frontier) > 0:

        # sort the frontier
        frontier.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = frontier.pop(0)

        # check if goal set has been reached
        if len(node.lanelets) >= goal_state.time_step.start \
                and is_intersecting(goal_space, node.space[len(node.space)-1]):
            return node.lanelets, node.space

        # create child nodes
        if len(node.lanelets) <= goal_state.time_step.end:
            childred = create_child_nodes(node, free_space, cost, lanelets, id2index)
            frontier.append(children)


def create_child_nodes(node, free_space, cost, lanelets, id2index):
    """create all possible child nodes for the current node"""

    # initialization
    children = []
    ind = len(node.lanelets)
    lanelet = lanelets[id2index[node.lanelets[ind-1]]]

    # bloat currently reachable space by maximum acceleration and deceleration
    space = node.space[ind-1]
    space = (space[0] + dt * A_MIN, space[1] + dt * A_MAX)

    # create children resulting from staying on the same lanelet
    space_lanelet = free_space[id2index[node.lanelets[ind-1]]][ind]

    for sp in space_lanelet:
        if is_intersecting(sp, space):
            l = deepcopy(node.lanelets)
            l.append(node.lanelets[ind-1])
            s = deepcopy(node.space)
            s.append((max(sp[0], space[0]), min(sp[1], space[1])))
            c = node.cost
            children.append(Node(l, s, c))

    # create children for lane change to the left
    if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
        space_left = free_space[id2index[lanelet.adj_left]][ind]
        if is_intersecting(sp, space_left):
            l = deepcopy(node.lanelets)
            l.append(lanelet.adj_left)
            s = deepcopy(node.space)
            s.append((max(sp[0], space_left[0]), min(sp[1], space_left[1])))
            c = node.cost
            children.append(Node(l, s, c))

    # create children for lane change to the right



class Node:

    def __init__(self, lanelets, space, cost):

        self.lanelets = lanelets
        self.space = space
        self.cost = cost

if __name__ == "__main__":

    motionPlanner(SCENARIO)