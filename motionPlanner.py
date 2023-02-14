from commonroad.common.file_reader import CommonRoadFileReader
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy

SCENARIO = 'ZAM_Zip-1_19_T-1.xml'

V_MAX = 100
A_MAX = 9
dt = 0.1
L = 4.3

def motionPlanner(file):
    """plan a safe motion for the given CommonRoad scenario"""

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(file).open()

    # high-level planner: decides on which lanelets to be at which points in time
    plan = highLevelPlanner(scenario, planning_problem)

    # low-level planner: plans a concrete trajectory for the high-level plan


def highLevelPlanner(scenario, planning_problem):
    """decide on which lanelets to be at all points in time"""

    # extract goal region and initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    goal = planning_problem.goal
    goal_lanelet = list(goal.lanelets_of_goal_position.values())[0][0]
    final_time = goal.state_list[0].time_step.end

    # crete dictionary that converts from lanelet ID to index in the list
    lanelets = scenario.lanelet_network.lanelets
    id = [l.lanelet_id for l in lanelets]
    id2index = dict(zip(id, [i for i in range(0, len(id))]))

    # compute costs (= number of lane changes to reach goal) for each lanelet
    cost_lane = cost_lanelets(lanelets, goal_lanelet, id2index)

    # compute free space on each lanelet at different times
    free_space = free_space_lanelet(lanelets, scenario.obstacles, final_time + 1)

    # determine the high-level plan using the A*-search algorithm
    plan = a_star_search(free_space, cost_lane, planning_problem, lanelets, id2index)

    return plan


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

def free_space_lanelet(lanelets, obstacles, length):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []

    # loop over all lanelets
    for l in lanelets:

        occupied_space = [[] for i in range(0, length)]

        # loop over all obstacles
        for obs in obstacles:

            # loop over all time steps
            for o in obs.prediction.occupancy_set:

                # check if dynamic obstacles occupancy set intersects the lanelet
                if l.polygon.shapely_object.intersects(o.shape.shapely_object):

                    # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                    dist_min, dist_max = projection_lanelet_centerline(l, o.shape.shapely_object)
                    occupied_space[o.time_step-1].append((dist_min-0.5*L, dist_max+0.5*L))

        # unite occupied spaces that belong to the same time step to obtain free space
        free_space = []

        for o in occupied_space:

            if len(o) > 0:

                o.sort(key=lambda i: i[0])
                lower = [i[0] for i in o]
                upper = [i[1] for i in o]

                space = []

                if lower[0] > l.distance[0]:
                    pgon = interval2polygon([l.distance[0], -V_MAX], [lower[0], V_MAX])
                    space.append(pgon)

                cnt = 0

                while cnt < len(o)-1:

                    ind = cnt

                    for i in range(cnt, len(o)):
                        if lower[i] < upper[cnt]:
                            ind = i

                    pgon = interval2polygon([upper[ind], -V_MAX], [lower[ind+1], V_MAX])
                    space.append(pgon)
                    cnt = ind + 1

                if max(upper) < l.distance[len(l.distance)-1]:
                    pgon = interval2polygon([max(upper), -V_MAX], [l.distance[len(l.distance)-1], V_MAX])
                    space.append(pgon)

            else:
                pgon = interval2polygon([l.distance[0], -V_MAX], [l.distance[len(l.distance)-1], V_MAX])
                space = [pgon]

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

def interval2polygon(lb, ub):
    """convert an interval given by lower and upper bound to a polygon"""

    return Polygon([(lb[0],lb[1]), (lb[0], ub[1]), (ub[0], ub[1]), (ub[0], lb[1])])

def expand_node(node, lanelet_id, space, lane_changes, expect_lane_changes):
    """add value for the current time step to a node for the search algorithm"""

    # add current lanelet to list of lanelets
    l = deepcopy(node.lanelets)
    l.append(lanelet_id)

    # add reachable space to the list
    s = deepcopy(node.space)
    s.append(space)

    # create resulting node
    return Node(l, s, lane_changes, expect_lane_changes)

def a_star_search(free_space, cost, planning_problem, lanelets, id2index):
    """determine optimal lanelet for each time step using A*-search"""

    # get goal time and set
    goal = planning_problem.goal
    goal_id = list(goal.lanelets_of_goal_position.values())[0][0]
    goal_state = goal.state_list[0]

    goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[id2index[goal_id]],
                                                                     goal_state.position.shapes[0].shapely_object)
    v = goal_state.velocity
    goal_space = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])

    # get lanelet and space for the initial point
    x0 = planning_problem.initial_state.position

    for l in lanelets:
        if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = l.lanelet_id

    pgon = interval2polygon([x0[0]-0.01, x0[1]-0.01], [x0[0]+0.01, x0[1]+0.01])
    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[id2index[x0_id]], pgon)

    v = planning_problem.initial_state.velocity
    x0_space = interval2polygon([x0_space_start, v - 0.01], [x0_space_end, v + 0.01])

    # initialize the frontier
    frontier = []
    frontier.append(Node([x0_id], [x0_space], 0, cost[id2index[x0_id]]))

    # loop until a solution has been found
    while len(frontier) > 0:

        # sort the frontier
        frontier.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = frontier.pop(0)

        # check if goal set has been reached
        if len(node.lanelets) >= goal_state.time_step.start and node.lanelets[len(node.lanelets)-1] == goal_id \
                and goal_space.intersects(node.space[len(node.space)-1]):
            return node.lanelets, node.space

        # create child nodes
        if len(node.lanelets) <= goal_state.time_step.end:
            children = create_child_nodes(node, free_space, cost, lanelets, id2index)
            frontier.extend(children)


def create_child_nodes(node, free_space, cost, lanelets, id2index):
    """create all possible child nodes for the current node"""

    # initialization
    children = []
    ind = len(node.lanelets)
    lanelet = lanelets[id2index[node.lanelets[ind-1]]]

    # compute reachable set using maximum acceleration and deceleration
    space = node.space[ind-1]

    space1 = affine_transform(space, [1, dt, 0, 1, 0.5 * dt ** 2 * A_MAX, dt * A_MAX])
    space2 = affine_transform(space, [1, dt, 0, 1, -0.5 * dt ** 2 * A_MAX, -dt * A_MAX])

    space = space1.union(space2)
    space = space.convex_hull

    # create children resulting from staying on the same lanelet
    space_lanelet = free_space[id2index[node.lanelets[ind-1]]][ind]

    for sp in space_lanelet:
        if sp.intersects(space):

            space_new = sp.intersection(space)
            lane_changes = node.lane_changes
            expect_lane_changes = cost[id2index[node.lanelets[ind-1]]]

            node_new = expand_node(node, node.lanelets[ind-1], space_new, lane_changes, expect_lane_changes)
            children.append(node_new)

    # create children for moving to a successor lanelet
    space_ = translate(space, -lanelet.distance[len(lanelet.distance)-1], 0)

    for suc in lanelet.successor:
        for sp in free_space[id2index[suc]][ind]:
            if space_.intersects(sp):

                space_new = sp.intersection(space_)
                lane_changes = node.lane_changes
                expect_lane_changes = cost[id2index[suc]]

                node_new = expand_node(node, suc, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the left
    if not lanelet.adj_left is None and lanelet.adj_left_same_direction:

        for space_left in free_space[id2index[lanelet.adj_left]][ind]:
            if space.intersects(space_left):

                space_new = space.intersection(space_left)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_left]]

                node_new = expand_node(node, lanelet.adj_left, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the right
    if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

        for space_right in free_space[id2index[lanelet.adj_right]][ind]:
            if space.intersects(space_right):

                space_new = space.intersection(space_right)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_right]]

                node_new = expand_node(node, lanelet.adj_right, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    return children

class Node:
    """class representing a single node for A*-search"""

    def __init__(self, lanelets, space, lane_changes, expect_lane_changes):
        """class constructor"""

        # store object properties
        self.lanelets = lanelets
        self.space = space
        self.lane_changes = lane_changes
        self.expect_lane_changes = expect_lane_changes

        # compute costs
        self.cost = self.cost_function()

    def cost_function(self):
        """cost function for A*-search"""

        return self.lane_changes + self.expect_lane_changes - len(self.lanelets)*dt

if __name__ == "__main__":

    motionPlanner(SCENARIO)