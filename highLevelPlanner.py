import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy


def highLevelPlanner(scenario, planning_problem, param):
    """decide on which lanelets to be at all points in time"""

    # store useful properties in parameter dictionary
    param['time_step'] = scenario.dt
    param['v_max'] = 100

    # extract goal region and initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    goal = planning_problem.goal
    goal_lanelet = list(goal.lanelets_of_goal_position.values())[0][0]
    final_time = goal.state_list[0].time_step.end

    # create dictionary that converts from lanelet ID to index in the list
    lanelets = scenario.lanelet_network.lanelets
    id = [l.lanelet_id for l in lanelets]
    id2index = dict(zip(id, [i for i in range(0, len(id))]))

    # compute costs (= number of lane changes to reach goal) as well as distance to goal for each lanelet
    cost_lane, dist_lane = cost_lanelets(lanelets, goal_lanelet, id2index)

    # determine lanelet and correpsoning space for the initial state
    lane_x0 = lanelet_initial_state(planning_problem, lanelets, id2index)

    # compute free space on each lanelet at different times
    free_space = free_space_lanelet(lanelets, scenario.obstacles, final_time + 1, param)

    # compute the desired velocity profile over time
    vel_prof = velocity_profile(planning_problem, scenario.dt, dist_lane, lanelets, id2index, lane_x0, final_time + 1)

    # determine the high-level plan using the A*-search algorithm
    plan, space = a_star_search(free_space, cost_lane, dist_lane, planning_problem,
                                lanelets, id2index, param, lane_x0, vel_prof)

    # shrink space by computing reachable set backward in time starting from final set
    space = reduce_space(space, plan, lanelets, id2index, param)

    # extract the safe velocity intervals at each time point
    vel = []
    for s in space:
        vel.append((s.bounds[1], s.bounds[3]))

    # transform space from lanelet coordinate system to global coordinate system
    space_xy = lanelet2global(space, plan, lanelets, id2index)

    return plan, space_xy, vel


def cost_lanelets(lanelets, goal_id, id2index):
    """assign a cost to each lanelet that is based on the number of required lane changes to reach goal set"""

    # initialize costs and distance to goal lanelet
    cost = [np.inf for i in range(0, len(lanelets))]
    cost[id2index[goal_id]] = 0

    dist = [np.inf for i in range(0, len(lanelets))]
    dist[id2index[goal_id]] = 0

    # initialize queue with goal lanelet
    queue = []
    queue.append(goal_id)

    # loop until costs for all lanelets have been assigned
    while len(queue) > 0:

        q = queue.pop(0)
        lanelet = lanelets[id2index[q]]

        for s in lanelet.predecessor:
            if cost[id2index[s]] == np.inf:
                lanelet_ = lanelets[id2index[s]]
                cost[id2index[s]] = cost[id2index[q]]
                dist[id2index[s]] = dist[id2index[q]] + lanelet_.distance[len(lanelet_.distance)-1]
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if cost[id2index[lanelet.adj_left]] == np.inf:
                cost[id2index[lanelet.adj_left]] = cost[id2index[q]] + 1
                dist[id2index[s]] = dist[id2index[q]]
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if cost[id2index[lanelet.adj_right]] == np.inf:
                cost[id2index[lanelet.adj_right]] = cost[id2index[q]] + 1
                dist[id2index[s]] = dist[id2index[q]]
                queue.append(lanelet.adj_right)

    return cost, dist

def lanelet_initial_state(planning_problem, lanelets, id2index):
    """get lanelet and space for the initial point"""

    x0 = planning_problem.initial_state.position

    for l in lanelets:
        if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = l.lanelet_id

    pgon = interval2polygon([x0[0] - 0.01, x0[1] - 0.01], [x0[0] + 0.01, x0[1] + 0.01])
    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[id2index[x0_id]], pgon)

    return {'id': x0_id, 'space': 0.5*(x0_space_start + x0_space_end)}

def velocity_profile(planning_problem, dt, dist, lanelets, id2index, lane_x0, length):
    """compute the desired velocity profile over time"""

    # initial velocity
    vel_init = planning_problem.initial_state.velocity

    # target velocity
    goal = planning_problem.goal
    vel_goal = goal.state_list[0].velocity

    # get distance from goal-lanelet start to goal set
    goal_state = goal.state_list[0]
    goal_set = goal_state.position.shapes[0]
    goal_id = list(goal.lanelets_of_goal_position.values())[0][0]
    dist_min, dist_max = projection_lanelet_centerline(lanelets[id2index[goal_id]], goal_set.shapely_object)

    # calculate minimum and maximum distance from initial state to goal set
    dist_min = dist[id2index[lane_x0['id']]] - lane_x0['space'] + dist_min
    dist_max = dist[id2index[lane_x0['id']]] - lane_x0['space'] + dist_max

    # calculate minimum and maximum final velocities required to reach the goal set
    vel_min = 2*dist_min/(goal_state.time_step.end*dt) - vel_init
    vel_max = 2*dist_max/(goal_state.time_step.start*dt) - vel_init

    vel_min = max(vel_min, vel_goal.start)
    vel_max = min(vel_max, vel_goal.end)

    if vel_min <= vel_init <= vel_max:
        vel = [vel_init for i in range(length)]
    elif vel_init < vel_min:
        vel = [vel_init + (vel_min - vel_init) * t/length for t in range(length)]
    else:
        vel = [vel_init + (vel_max - vel_init) * t/length for t in range(length)]

    return vel

def free_space_lanelet(lanelets, obstacles, length, param):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []
    v_max = param['v_max']

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
                    occupied_space[o.time_step-1].append((dist_min-0.5*param['length'], dist_max+0.5*param['length']))

        # unite occupied spaces that belong to the same time step to obtain free space
        free_space = []

        for o in occupied_space:

            if len(o) > 0:

                o.sort(key=lambda i: i[0])
                lower = [i[0] for i in o]
                upper = [i[1] for i in o]

                space = []

                if lower[0] > l.distance[0]:
                    pgon = interval2polygon([l.distance[0], -v_max], [lower[0], v_max])
                    space.append(pgon)

                cnt = 0

                while cnt < len(o)-1:

                    ind = cnt

                    for i in range(cnt, len(o)):
                        if lower[i] < upper[cnt]:
                            ind = i

                    pgon = interval2polygon([upper[ind], -v_max], [lower[ind+1], v_max])
                    space.append(pgon)
                    cnt = ind + 1

                if max(upper) < l.distance[len(l.distance)-1]:
                    pgon = interval2polygon([max(upper), -v_max], [l.distance[len(l.distance)-1], v_max])
                    space.append(pgon)

            else:
                pgon = interval2polygon([l.distance[0], -v_max], [l.distance[len(l.distance)-1], v_max])
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

def expand_node(node, lanelet_id, space, dist_goal, lane_changes, expect_lane_changes):
    """add value for the current time step to a node for the search algorithm"""

    # add current lanelet to list of lanelets
    l = deepcopy(node.lanelets)
    l.append(lanelet_id)

    # add reachable space to the list
    s = deepcopy(node.space)
    s.append(space)

    # create resulting node
    return Node(node.dt, l, s, node.vel_prof, dist_goal, lane_changes, expect_lane_changes)

def a_star_search(free_space, cost, dist, planning_problem, lanelets, id2index, param, lane_x0, vel_prof):
    """determine optimal lanelet for each time step using A*-search"""

    # get goal time and set
    goal = planning_problem.goal
    goal_id = list(goal.lanelets_of_goal_position.values())[0][0]
    goal_state = goal.state_list[0]

    goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[id2index[goal_id]],
                                                                     goal_state.position.shapes[0].shapely_object)
    v = goal_state.velocity
    goal_space = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])

    # construct space for the initial point
    v = planning_problem.initial_state.velocity
    x0_space = interval2polygon([lane_x0['space']-0.01, v - 0.01], [lane_x0['space']+0.01, v + 0.01])

    # initialize the frontier
    frontier = []
    frontier.append(Node(param['time_step'], [lane_x0['id']], [x0_space], vel_prof, 0, 0, cost[id2index[lane_x0['id']]]))

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
            children = create_child_nodes(node, free_space, cost, dist, lanelets, id2index, param)
            frontier.extend(children)


def create_child_nodes(node, free_space, cost, dist, lanelets, id2index, param):
    """create all possible child nodes for the current node"""

    # initialization
    children = []
    ind = len(node.lanelets)
    lanelet = lanelets[id2index[node.lanelets[ind-1]]]

    # compute reachable set using maximum acceleration and deceleration
    space = node.space[ind-1]

    dt = param['time_step']
    a_max = param['a_max']

    space1 = affine_transform(space, [1, dt, 0, 1, 0.5 * dt ** 2 * a_max, dt * a_max])
    space2 = affine_transform(space, [1, dt, 0, 1, -0.5 * dt ** 2 * a_max, -dt * a_max])

    space = space1.union(space2)
    space = space.convex_hull

    # create children resulting from staying on the same lanelet
    space_lanelet = free_space[id2index[node.lanelets[ind-1]]][ind]

    for sp in space_lanelet:
        if sp.intersects(space):

            space_new = sp.intersection(space)
            lane_changes = node.lane_changes
            expect_lane_changes = cost[id2index[node.lanelets[ind-1]]]
            dist_goal = dist[id2index[node.lanelets[ind-1]]]

            node_new = expand_node(node, node.lanelets[ind-1], space_new, dist_goal, lane_changes, expect_lane_changes)
            children.append(node_new)

    # create children for moving to a successor lanelet
    space_ = translate(space, -lanelet.distance[len(lanelet.distance)-1], 0)

    for suc in lanelet.successor:
        for sp in free_space[id2index[suc]][ind]:
            if space_.intersects(sp):

                space_new = sp.intersection(space_)
                lane_changes = node.lane_changes
                expect_lane_changes = cost[id2index[suc]]
                dist_goal = dist[id2index[suc]]

                node_new = expand_node(node, suc, space_new, dist_goal, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the left
    if not lanelet.adj_left is None and lanelet.adj_left_same_direction:

        for space_left in free_space[id2index[lanelet.adj_left]][ind]:
            if space.intersects(space_left):

                space_new = space.intersection(space_left)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_left]]
                dist_goal = dist[id2index[lanelet.adj_left]]

                node_new = expand_node(node, lanelet.adj_left, space_new, dist_goal, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the right
    if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

        for space_right in free_space[id2index[lanelet.adj_right]][ind]:
            if space.intersects(space_right):

                space_new = space.intersection(space_right)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_right]]
                dist_goal = dist[id2index[lanelet.adj_right]]

                node_new = expand_node(node, lanelet.adj_right, space_new, dist_goal, lane_changes, expect_lane_changes)
                children.append(node_new)

    return children

def reduce_space(space, plan, lanelets, id2index, param):
    """reduce the drivable space by propagating the goal set backward in time"""

    # loop over all time steps
    for i in range(len(space)-1, 0, -1):

        # propagate reachable set backwards in time
        dt = param['time_step']
        a_max = param['a_max']

        space1 = affine_transform(space[i], [1, -dt, 0, 1, -0.5 * dt ** 2 * a_max, dt * a_max])
        space2 = affine_transform(space[i], [1, -dt, 0, 1, 0.5 * dt ** 2 * a_max, -dt * a_max])
        space_new = space1.union(space2)
        space_new = space_new.convex_hull

        # shift reachable set by lanelet length if the lanelet is changed
        if not plan[i-1] == plan[i] and not lanelets[id2index[plan[i-1]]].adj_left == plan[i] and \
                not lanelets[id2index[plan[i-1]]].adj_right == plan[i]:
            lanelet = lanelets[id2index[plan[i-1]]]
            dist = lanelet.distance[len(lanelet.distance)-1]
            space_new = translate(space_new, dist, 0)

        # intersect with previous rechable set
        space[i - 1] = space_new.intersection(space[i - 1])

    return space

def lanelet2global(space, plan, lanelets, id2index):
    """transform free space from lanelet coordinate system to global coordinate system"""

    space_xy = []

    # loop over all time steps
    for i in range(0, len(space)):

        # initialization
        lanelet = lanelets[id2index[plan[i]]]

        lower = space[i].bounds[0]
        upper = space[i].bounds[2]

        left_vertices = []
        right_vertices = []

        # loop over the single segments of the lanelet
        for j in range(0, len(lanelet.distance)-1):

            intermediate_point = True

            if lower >= lanelet.distance[j] and lower <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d/np.linalg.norm(d) * (lower - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (lower - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

                intermediate_point = False

            if upper >= lanelet.distance[j] and upper <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

                break

            if len(left_vertices) > 0 and intermediate_point:

                p_left = lanelet.left_vertices[j]
                left_vertices.append(Point(p_left[0], p_left[1]))

                p_right = lanelet.right_vertices[j]
                right_vertices.append(Point(p_right[0], p_right[1]))

        # construct the resulting polygon in the global coordinate system
        right_vertices.reverse()
        left_vertices.extend(right_vertices)
        space_xy.append(Polygon(left_vertices))

    return space_xy

class Node:
    """class representing a single node for A*-search"""

    def __init__(self, dt, lanelets, space, vel_prof, dist, lane_changes, expect_lane_changes):
        """class constructor"""

        # store object properties
        self.dt = dt
        self.lanelets = lanelets
        self.space = space
        self.vel_prof = vel_prof
        self.dist = dist - space[len(space)-1].bounds[2]
        self.lane_changes = lane_changes
        self.expect_lane_changes = expect_lane_changes

        # compute costs
        self.cost = self.cost_function()

    def cost_function(self):
        """cost function for A*-search"""

        # current time
        time_curr = len(self.lanelets)*self.dt

        # time until reaching the goal set
        vel = self.space[len(self.space)-1].bounds[3]
        time_goal = self.dist/vel

        # expected total time for reaching the goal set
        time = time_curr + time_goal

        # difference from the desired velocity profile up to now
        vel_curr = 0

        for i in range(len(self.space)):
            if self.vel_prof[i] <= self.space[i].bounds[1]:
                vel_curr = vel_curr + (self.space[i].bounds[1] - self.vel_prof[i])
            elif self.vel_prof[i] >= self.space[i].bounds[3]:
                vel_curr = vel_curr + (self.vel_prof[i] - self.space[i].bounds[3])

        # expected future difference from the velocity profile
        vel_expect = 0
        n = len(self.space)-1

        if self.vel_prof[n] > self.space[n].bounds[3]:
            vel = self.space[n].bounds[3]
        elif self.vel_prof[n] < self.space[n].bounds[1]:
            vel = self.space[n].bounds[1]
        else:
            vel = self.vel_prof[n]

        for i in range(len(self.space), len(self.vel_prof)):
            vel_expect = vel_expect + np.abs(self.vel_prof[i] - vel)

        # expected total average difference from the velocity profile
        vel_diff = (vel_curr + vel_expect)/len(self.vel_prof)

        # expected total number of lane changes until reaching the goal
        lane_changes = self.lane_changes + self.expect_lane_changes

        return time + lane_changes + 0*vel_diff
