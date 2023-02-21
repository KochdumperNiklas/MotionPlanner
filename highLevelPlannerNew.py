import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy
from commonroad.scenario.obstacle import StaticObstacle

# weighting factors for the cost function
W_LANE_CHANGE = 1
W_VELOCITY = 1

# safety distance
DIST_SAFE = 2

# minimum number of consecutive time steps required to perform a lane change
MIN_LANE_CHANGE = 2

def highLevelPlannerNew(scenario, planning_problem, param):
    """decide on which lanelets to be at all points in time"""

    # extract required information from the planning problem
    param, lanelets = initialization(scenario, planning_problem, param)

    # compute free space on each lanelet for each time step
    free_space = free_space_lanelet(lanelets, scenario.obstacles, param)

    # compute the desired velocity profile over time
    vel_prof = velocity_profile(lanelets, param)

    # determine all feasible sequence of lanelets to drive on
    list_seq = feasible_lanelet_sequences(lanelets, free_space, param)

    # select the best sequence of lanelets
    seq = select_best_sequence(list_seq, vel_prof, lanelets)

    # refine the plan: decide on which lanelet to be on for all time steps
    test = 1

    # shrink space by computing reachable set backward in time starting from final set
    space = reduce_space(space, plan, lanelets, id2index, param)

    # extract the safe velocity intervals at each time point
    vel = []
    for s in space:
        vel.append((s.bounds[1], s.bounds[3]))

    # transform space from lanelet coordinate system to global coordinate system
    space_xy = lanelet2global(space, plan, lanelets, id2index)

    return plan, space_xy, vel


def initialization(scenario, planning_problem, param):
    """extract and store the required information from the scenario and the planning problem"""

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    id = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(id, scenario.lanelet_network.lanelets))

    # store useful properties in parameter dictionary
    param['time_step'] = scenario.dt
    param['v_max'] = 100

    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    param['v_init'] = planning_problem.initial_state.velocity
    param['steps'] = planning_problem.goal.state_list[0].time_step.end

    # extract parameter for the goal set
    param['goal_lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[0][0]
    param['goal_time_start'] = planning_problem.goal.state_list[0].time_step.start
    param['goal_time_end'] = planning_problem.goal.state_list[0].time_step.start

    goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[param['goal_lane']],
                                                                     planning_problem.goal.state_list[0].position.shapes[
                                                                         0].shapely_object)
    v = planning_problem.goal.state_list[0].velocity
    param['goal_set'] = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])

    # determine lanelet and corresponding space for the initial state
    x0 = planning_problem.initial_state.position

    for id in lanelets.keys():
        if lanelets[id].polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = id

    pgon = interval2polygon([x0[0] - 0.01, x0[1] - 0.01], [x0[0] + 0.01, x0[1] + 0.01])
    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[x0_id], pgon)

    param['x0_lane'] = x0_id
    param['x0_set'] = 0.5 * (x0_space_start + x0_space_end)

    return param, lanelets


def distance2goal(lanelets, param):
    """compute the distance to the target lanelet for each lanelet"""

    # initialize distance to goal lanelet
    dist = {param['goal_lane']: 0}

    # initialize queue with goal lanelet
    queue = []
    queue.append(param['goal_lane'])

    # loop until distances for all lanelets have been assigned
    while len(queue) > 0:

        q = queue.pop(0)
        lanelet = lanelets[q]

        for s in lanelet.predecessor:
            if not s in dist.keys():
                lanelet_ = lanelets[s]
                dist[s] = dist[q] + lanelet_.distance[len(lanelet_.distance)-1]
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if not lanelet.adj_left in dist.keys():
                dist[lanelet.adj_left] = dist[q]
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if not lanelet.adj_right in dist.keys():
                dist[lanelet.adj_right] = dist[q]
                queue.append(lanelet.adj_right)

    return dist

def velocity_profile(lanelets, param):
    """compute the desired velocity profile over time"""

    # calculate distance to target lanelet for each lanelet
    dist = distance2goal(lanelets, param)

    # calculate minimum and maximum distance from initial state to goal set
    dist_min = dist[param['x0_lane']] - param['x0_set'] + param['goal_set'].bounds[0]
    dist_max = dist[param['x0_lane']] - param['x0_set'] + param['goal_set'].bounds[2]

    # calculate minimum and maximum final velocities required to reach the goal set
    vel_min = 2*dist_min/(param['goal_time_end']*param['time_step']) - param['v_init']
    vel_max = 2*dist_max/(param['goal_time_start']*param['time_step']) - param['v_init']

    vel_min = max(vel_min, param['goal_set'].bounds[1])
    vel_max = min(vel_max, param['goal_set'].bounds[3])

    if vel_min <= param['v_init'] <= vel_max:
        vel = [param['v_init'] for i in range(param['steps'])]
    elif param['v_init'] < vel_min:
        vel = [param['v_init'] + (vel_min - param['v_init']) * t/param['steps'] for t in range(length)]
    else:
        vel = [param['v_init'] + (vel_max - param['v_init']) * t/param['steps'] for t in range(length)]

    return vel

def free_space_lanelet(lanelets, obstacles, param):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []
    v_max = param['v_max']

    # loop over all lanelets
    for id in lanelets.keys():

        l = lanelets[id]
        occupied_space = [[] for i in range(0, param['steps']+1)]

        # loop over all obstacles
        for obs in obstacles:

            # distinguish static and dynamic obstacles
            if isinstance(obs, StaticObstacle):

                # check if static obstacle intersects the lanelet
                if l.polygon.shapely_object.intersects(obs.obstacle_shape.shapely_object):

                    # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                    dist_min, dist_max = projection_lanelet_centerline(l, obs.obstacle_shape.shapely_object)
                    offset = 0.5 * param['length'] + DIST_SAFE

                    # loop over all time steps
                    for i in range(param['steps']+1):
                        occupied_space[i].append((dist_min - offset, dist_max + offset))

            else:

                # loop over all time steps
                for o in obs.prediction.occupancy_set:

                    # check if dynamic obstacles occupancy set intersects the lanelet
                    if l.polygon.shapely_object.intersects(o.shape.shapely_object):

                        # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                        dist_min, dist_max = projection_lanelet_centerline(l, o.shape.shapely_object)
                        offset = 0.5*param['length'] + DIST_SAFE
                        occupied_space[o.time_step-1].append((dist_min-offset, dist_max+offset))

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

    return dict(zip(lanelets.keys(), free_space_all))

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

def interval2polygon(lb, ub):
    """convert an interval given by lower and upper bound to a polygon"""

    return Polygon([(lb[0], lb[1]), (lb[0], ub[1]), (ub[0], ub[1]), (ub[0], lb[1])])


def feasible_lanelet_sequences(lanelets, free_space, param):
    """determine all feasible sequences of lanelets to drive on that reach the goal state"""

    final_nodes = []

    # create initial node
    v = param['v_init']
    space = interval2polygon([param['x0_set'] - 0.01, v - 0.01], [param['x0_set'] + 0.01, v + 0.01])
    x0 = {'step': 0, 'space': space, 'lanelet': param['x0_lane']}

    queue = [Node([x0], [], [], 'none')]

    # loop until queue empty -> all possible lanelet sequences have been explored
    while len(queue) > 0:

        # take node from queue
        node = queue.pop()

        # compute drivable area for the current lanelet
        lanelet = lanelets[node.x0[0]['lanelet']]

        if len(node.lanelets) == 0:
            prev = []
        else:
            prev = node.drive_area[len(node.drive_area)-1]

        drive_area, left, right, suc = compute_drivable_area(lanelet, node.x0, free_space, prev, node.lane_prev, param)

        # check if goal set has been reached
        if lanelet.lanelet_id == param['goal_lane']:
            for d in drive_area:
                if d['space'].intersects(param['goal_set']):
                    final_nodes.append(expand_node(node, [], drive_area, 'none'))
                    break

        # create child nodes
        for entry in left:
            if len(entry) > MIN_LANE_CHANGE:
                queue.append(expand_node(node, entry, drive_area, 'right'))

        for entry in right:
            if len(entry) > MIN_LANE_CHANGE:
                queue.append(expand_node(node, entry, drive_area, 'left'))

        for entry in suc:
            if len(entry) > MIN_LANE_CHANGE:
                queue.append(expand_node(node, entry, drive_area, 'none'))

    return final_nodes

def compute_drivable_area(lanelet, x0, free_space, prev, lane_prev, param):
    """compute the drivable area for a single lanelet"""

    # initialization
    cnt = 0
    cnt_prev = 0

    for i in range(len(prev)):
        if prev[i]['step'] <= x0[cnt]['step']:
            cnt_prev = cnt_prev + 1

    successor_possible = False

    drive_area = [x0[cnt]]
    successors = []
    left = []
    right = []

    # loop over all time steps up to the final time
    for i in range(x0[cnt]['step'], param['steps']-1):

        # compute reachable set using maximum acceleration and deceleration
        space = reach_set_forward(drive_area[len(drive_area)-1]['space'], param)

        # unite with set resulting from doing a lane-change from the predecessor lanelet
        if cnt+1 < len(x0)-1 and x0[cnt+1]['step'] == i+1:
            space = space.union(x0[cnt+1]['space'])
            space = space.convex_hull
            cnt = cnt + 1

        # avoid intersection with lanelet for moving on to a successor lanelet
        if successor_possible:
            space_prev = reach_set_forward(space_prev, param)
        else:
            space_prev = space

        # intersect with free space on the lanelet
        space_lanelet = free_space[lanelet.lanelet_id][i+1]
        tmp = []

        for sp in space_lanelet:
            if sp.intersects(space):
                tmp.append(sp.intersection(space))

        if len(tmp) > 1:
            raise Exception('High-level planning failed!')
        elif len(tmp) == 0:
            break
        else:
            space = tmp[0]

        # check if it is possible to move on to a successor lanelet
        space_ = translate(space, -lanelet.distance[len(lanelet.distance) - 1], 0)
        successor_possible_ = False

        for suc in lanelet.successor:
            for sp in free_space[suc][i+1]:
                if space_.intersects(sp):
                    space_prev_ = translate(space_prev, -lanelet.distance[len(lanelet.distance) - 1], 0)
                    successors = add_transition(successors, sp.intersection(space_prev_), i + 1, suc, param)
                    successor_possible_ = True

        successor_possible = successor_possible_

        # check if it is possible to make a lane change to the left
        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:

            for space_left in free_space[lanelet.adj_left][i+1]:
                if space.intersects(space_left) and not(lane_prev == 'left' and prev[cnt_prev]['step'] == i+1 and
                                                                    space_left.intersects(prev[cnt_prev]['space'])):
                    left = add_transition(left, space.intersection(space_left), i+1, lanelet.adj_left, param)

        # check if it is possible to make a lane change to the right
        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

            for space_right in free_space[lanelet.adj_right][i + 1]:
                if space.intersects(space_right) and not(lane_prev == 'right' and prev[cnt_prev]['step'] == i+1 and
                                                                    space_right.intersects(prev[cnt_prev]['space'])):
                    right = add_transition(right, space.intersection(space_right), i+1, lanelet.adj_right, param)

        # update counter for drivable area on previous lanelet
        if cnt_prev < len(prev) and prev[cnt_prev]['step'] == i+1:
            cnt_prev = cnt_prev + 1

        # store drivable area
        drive_area.append({'space': space, 'step': i+1})

    return drive_area, left, right, successors

def reach_set_forward(space, param):
    """compute the forward reachable set for one time step using maximum acceleration and deceleration"""

    dt = param['time_step']
    a_max = param['a_max']

    space1 = affine_transform(space, [1, dt, 0, 1, 0.5 * dt ** 2 * a_max, dt * a_max])
    space2 = affine_transform(space, [1, dt, 0, 1, -0.5 * dt ** 2 * a_max, -dt * a_max])

    space_new = space1.union(space2)
    space_new = space_new.convex_hull

    return space_new

def reach_set_backward(space, param):
    """compute the backward reachable set for one time step using maximum acceleration and deceleration"""

    dt = param['time_step']
    a_max = param['a_max']

    space1 = affine_transform(space, [1, -dt, 0, 1, -0.5 * dt ** 2 * a_max, dt * a_max])
    space2 = affine_transform(space, [1, -dt, 0, 1, 0.5 * dt ** 2 * a_max, -dt * a_max])

    space_new = space1.union(space2)
    space_new = space_new.convex_hull

    return space_new

def add_transition(transitions, space, time_step, lanelet, param):
    """add a new transition to the list of possible transitions"""

    # propagate reachable set one time step backwards in time
    space_new = reach_set_backward(space, param)

    # loop over all existing transitions
    found = False

    for i in range(len(transitions)):

        last_entry = transitions[i][len(transitions[i]) - 1]

        # check if the current transition can be added to an existing transition
        if last_entry['step'] == time_step-1 and last_entry['lanelet'] == lanelet and \
                space_new.intersects(last_entry['space']):
            transitions[i].append({'space': space, 'step': time_step, 'lanelet': lanelet})
            found = True
            break

    # create a new transition if the current transition could not be added to any exising transition
    if not found:
        transitions.append([{'space': space, 'step': time_step, 'lanelet': lanelet}])

    return transitions



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

def select_best_sequence(list_seq, vel_prof, lanelets):
    """determine the lanelet sequence with the lowest cost"""

    min_cost = np.inf

    for l in list_seq:
        c = l.cost(vel_prof, lanelets)
        if c < min_cost:
            seq = deepcopy(l)
            min_cost = c

    return seq

def expand_node(node, x0, drive_area, lane_prev):
    """add value for the current step to a given node"""

    # add current lanelet to list of lanelets
    l = deepcopy(node.lanelets)
    l.append(node.x0[0]['lanelet'])

    # add driveable area to the list
    s = deepcopy(node.drive_area)
    s.append(drive_area)

    # create resulting node
    return Node(x0, l, s, lane_prev)


class Node:
    """class representing a single node for A*-search"""

    def __init__(self, x0, lanelets, drive_area, lane_prev):
        """class constructor"""

        # store object properties
        self.x0 = x0
        self.lanelets = lanelets
        self.drive_area = drive_area
        self.lane_prev = lane_prev

    def cost(self, vel_prof, lanelets):
        """compute cost function value for the node"""

        # determine cost from deviation to the desired velocity profile
        vel_diff = np.inf * np.ones(len(vel_prof))

        for i in range(len(vel_prof)):

            # loop over all lanelets
            for d in self.drive_area:

                vel_cur = np.inf

                # loop over all time steps
                for set in d:

                    # compute distance from desired velocity profile
                    if set['step'] == i:
                        if vel_prof[i] <= set['space'].bounds[1]:
                            vel_cur = set['space'].bounds[1] - vel_prof[i]
                        elif vel_prof[i] >= set['space'].bounds[3]:
                            vel_cur = vel_prof[i] - set['space'].bounds[3]
                        else:
                            vel_cur = 0
                    elif set['step'] > i:
                        break

                # total distance -> minimum over the distance for all lanelets
                if vel_cur < vel_diff[i]:
                    vel_diff[i] = vel_cur

                if vel_diff[i] == 0:
                    break

        # determine number of lane changes
        lane_changes = 0

        for i in range(1, len(self.lanelets)):
            l = lanelets[self.lanelets[i-1]]
            if not self.lanelets[i] in l.successor:
                lane_changes = lane_changes + 1

        return W_LANE_CHANGE * lane_changes + W_VELOCITY * np.mean(vel_diff)
