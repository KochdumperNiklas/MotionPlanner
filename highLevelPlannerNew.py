import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy
from commonroad.scenario.obstacle import StaticObstacle

# weighting factors for the cost function
W_LANE_CHANGE = 1000
W_DIST = 1

# safety distance to other cars
DIST_SAFE = 2

# minimum number of consecutive time steps required to perform a lane change
MIN_LANE_CHANGE = 2

# desired number of time steps for performing a lane change
DES_LANE_CHANGE = 20

def highLevelPlannerNew(scenario, planning_problem, param):
    """decide on which lanelets to be at all points in time"""

    # extract required information from the planning problem
    param, lanelets = initialization(scenario, planning_problem, param)

    # compute free space on each lanelet for each time step
    free_space = free_space_lanelet(lanelets, scenario.obstacles, param)

    # compute distance to goal lanelet and number of required lane changes for each lanelet
    dist_goal, change_goal = distance2goal(lanelets, param)

    # compute the desired velocity profile over time
    vel_prof = velocity_profile(dist_goal, param)

    # compute the desired reference trajectory
    ref_traj = velocity2trajectory(vel_prof, param)

    # determine best sequence of lanelets to drive on
    seq = best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, param)

    # refine the plan: decide on which lanelet to be on for all time steps
    plan, space = refine_plan(seq, ref_traj, lanelets, param)

    # determine drivable space for lane changes
    space_all, time_lane = space_lane_changes(space, plan, lanelets, free_space, param)

    # shrink space by intersecting with the forward reachable set
    space = reduce_space(space, plan, lanelets, param)

    # extract the safe velocity intervals at each time point
    vel = []
    for s in space:
        vel.append((s.bounds[1], s.bounds[3]))

    # compute a desired reference trajectory
    ref_traj = reference_trajectory(plan, seq, space, vel, time_lane, param, lanelets)

    # transform space from lanelet coordinate system to global coordinate system
    space_xy = lanelet2global(space, plan, lanelets)

    return plan, space_xy, vel, space_all, ref_traj


def initialization(scenario, planning_problem, param):
    """extract and store the required information from the scenario and the planning problem"""

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    id = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(id, scenario.lanelet_network.lanelets))

    # store useful properties in parameter dictionary
    param['time_step'] = scenario.dt
    param['v_max'] = 100
    param['v_min'] = 0

    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    param['v_init'] = planning_problem.initial_state.velocity
    param['steps'] = planning_problem.goal.state_list[0].time_step.end

    # maximum length of the car independent of the orientation
    param['length_max'] = np.sqrt(param['length']**2 + param['width']**2)

    # determine lanelet and corresponding space for the initial state
    x0 = planning_problem.initial_state.position

    for id in lanelets.keys():
        if lanelets[id].polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = id

    pgon = interval2polygon([x0[0] - 0.01, x0[1] - 0.01], [x0[0] + 0.01, x0[1] + 0.01])
    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[x0_id], pgon)

    param['x0_lane'] = x0_id
    param['x0_set'] = 0.5 * (x0_space_start + x0_space_end)
    param['x0'] = x0
    param['orientation'] = planning_problem.initial_state.orientation

    # extract parameter for the goal set
    param['goal_time_start'] = planning_problem.goal.state_list[0].time_step.start
    param['goal_time_end'] = planning_problem.goal.state_list[0].time_step.end

    if hasattr(planning_problem.goal.state_list[0], 'position'):

        set = get_shapely_object(planning_problem.goal.state_list[0].position)
        param['goal_space'] = set

        if planning_problem.goal.lanelets_of_goal_position is None:
            for id in lanelets.keys():
                if lanelets[id].polygon.shapely_object.intersects(set):
                    param['goal_lane'] = id
        else:
            param['goal_lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[0][0]

        goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[param['goal_lane']], set)

    else:

        if planning_problem.goal.lanelets_of_goal_position is None:
            param['goal_lane'] = None
            param['goal_space'] = None
            goal_space_start = -100000
            goal_space_end = 100000
        else:
            param['goal_lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[0][0]
            l = lanelets[param['goal_lane']]
            param['goal_space'] = l.polygon.shapely_object
            goal_space_start, goal_space_end = projection_lanelet_centerline(l, l.polygon.shapely_object)

    if hasattr(planning_problem.goal.state_list[0], 'velocity'):
        v = planning_problem.goal.state_list[0].velocity
        param['goal_set'] = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])
    else:
        param['goal_set'] = interval2polygon([goal_space_start, param['v_min']], [goal_space_end, param['v_max']])

    return param, lanelets

def get_shapely_object(set):
    """construct the shapely polygon object for the given CommonRoad set"""

    if hasattr(set, 'shapes'):
        pgon = set.shapes[0].shapely_object
        for i in range(1, len(set.shapes)):
            pgon = pgon.union(set.shapes[i].shapely_object)
    else:
        pgon = set.shapely_object

    return pgon

def intersects_lanelet(lanelet, pgon):
    """check if a polygon intersects the given lanelet"""

    res = False

    if lanelet.polygon.shapely_object.intersects(pgon):

        set = lanelet.polygon.shapely_object.intersection(pgon)

        if isinstance(set, Polygon):
            res = True

    return res

def distance2goal(lanelets, param):
    """compute the distance to the target lanelet for each lanelet"""

    # catch the case where no goal lane is provided
    if param['goal_lane'] is None:
        default = {}
        for l in lanelets.keys():
            default[l] = 0
        return default, default

    # initialize distance to goal lanelet
    dist = {param['goal_lane']: 0}
    change = {param['goal_lane']: 0}

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
                change[s] = change[q]
                dist[s] = dist[q] + lanelet_.distance[len(lanelet_.distance)-1]
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if not lanelet.adj_left in dist.keys():
                change[lanelet.adj_left] = change[q] + 1
                dist[lanelet.adj_left] = dist[q]
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if not lanelet.adj_right in dist.keys():
                change[lanelet.adj_right] = change[q] + 1
                dist[lanelet.adj_right] = dist[q]
                queue.append(lanelet.adj_right)

    # add costs for lanelet from which it is impossible to reach the goal set
    for l in lanelets.keys():
        if l not in dist.keys():
            dist[l] = np.inf
            change[l] = np.inf

    return dist, change

def velocity_profile(dist, param):
    """compute the desired velocity profile over time"""

    # calculate minimum and maximum velocity required at the end of the time horizon
    vel_min = param['goal_set'].bounds[1]
    vel_max = param['goal_set'].bounds[3]

    if param['goal_lane'] is not None:

        # calculate minimum and maximum distance from initial state to goal set
        dist_min = dist[param['x0_lane']] - param['x0_set'] + param['goal_set'].bounds[0]
        dist_max = dist[param['x0_lane']] - param['x0_set'] + param['goal_set'].bounds[2]

        # calculate minimum and maximum final velocities required to reach the goal set using a linear velocity profile
        vel_min_ = 2*dist_min/(param['goal_time_end']*param['time_step']) - param['v_init']
        vel_max_ = 2*dist_max/(param['goal_time_start']*param['time_step']) - param['v_init']

        # use a quadratic velocity profile if a linear one is not sufficient
        if vel_min_ > vel_max or vel_max_ < vel_min:
            if vel_max_ < vel_min:
                vel_final = vel_min
                x_final = dist_max
                t_final = param['goal_time_start']
            else:
                vel_final = vel_min
                x_final = dist_min
                t_final = param['goal_time_end']

            b = -5.0/6.0 * (x_final - 0.5*t_final*(param['v_init'] + vel_final))/t_final**3
            a = (vel_final - param['v_init'] - b * t_final ** 2)/t_final

            vel = [param['v_init'] + a * t + b * t**2 for t in range(param['steps']+1)]

            return vel
        else:
            vel_min = max(vel_min, vel_min_)
            vel_max = min(vel_max, vel_max_)

    # calculate velocity profile
    if vel_min <= param['v_init'] <= vel_max:
        vel = [param['v_init'] for i in range(param['steps']+1)]
    elif param['v_init'] < vel_min:
        vel = [param['v_init'] + (vel_min - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]
    else:
        vel = [param['v_init'] + (vel_max - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]

    return vel

def free_space_lanelet(lanelets, obstacles, param):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []
    tmp = [deepcopy([[] for i in range(0, param['steps']+1)]) for j in range(len(lanelets))]
    occupied_space = dict(zip(lanelets.keys(), tmp))
    v_max = param['v_max']
    v_min = param['v_min']

    # loop over all lanelets and compute occupied space
    for id in lanelets.keys():

        l = lanelets[id]
        d_min = l.distance[0]
        d_max = l.distance[-1]

        # loop over all obstacles
        for obs in obstacles:

            # distinguish static and dynamic obstacles
            if isinstance(obs, StaticObstacle):

                pgon = get_shapely_object(obs.obstacle_shape)

                # check if static obstacle intersects the lanelet
                if intersects_lanelet(l, pgon):

                    # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                    dist_min, dist_max = projection_lanelet_centerline(l, pgon)
                    offset = 0.5 * param['length_max'] + DIST_SAFE

                    # loop over all time steps
                    for i in range(param['steps']+1):
                        occupied_space[id][i].append((max(d_min, dist_min - offset), min(d_max, dist_max + offset)))

                    # check if occupied space extends to predecessor lanelet
                    if dist_min - offset < 0:
                        for ind in l.predecessor:
                            for i in range(param['steps']+1):
                                d = lanelets[ind].distance[-1]
                                occupied_space[ind][i].append((d + dist_min - offset, d))

                    # check if occupied space extends to successor lanelet
                    if dist_max + offset > l.distance[-1]:
                        for ind in l.successor:
                            for i in range(param['steps']+1):
                                d = dist_max + offset - l.distance[-1]
                                occupied_space[ind][i].append((0, d))

            else:

                # loop over all time steps
                for o in obs.prediction.occupancy_set:

                    pgon = get_shapely_object(o.shape)

                    # check if dynamic obstacles occupancy set intersects the lanelet
                    if intersects_lanelet(l, pgon) and o.time_step < len(occupied_space[id]):

                        # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                        dist_min, dist_max = projection_lanelet_centerline(l, pgon)
                        offset = 0.5*param['length_max'] + DIST_SAFE
                        occupied_space[id][o.time_step].append((max(d_min, dist_min-offset), min(d_max, dist_max+offset)))

                        # check if occupied space extends to predecessor lanelet
                        if dist_min - offset < 0:
                            for ind in l.predecessor:
                                d = lanelets[ind].distance[-1]
                                occupied_space[ind][o.time_step].append((d + dist_min - offset, d))

                        # check if occupied space extends to successor lanelet
                        if dist_max + offset > l.distance[-1]:
                            for ind in l.successor:
                                d = dist_max + offset - l.distance[-1]
                                occupied_space[ind][o.time_step].append((0, d))

    # loop over all lanelets and compute free space
    for id in lanelets.keys():

        l = lanelets[id]

        # unite occupied spaces that belong to the same time step to obtain free space
        free_space = []

        for o in occupied_space[id]:

            if len(o) > 0:

                o.sort(key=lambda i: i[0])
                lower = [i[0] for i in o]
                upper = [i[1] for i in o]

                space = []

                if lower[0] > l.distance[0]:
                    pgon = interval2polygon([l.distance[0], v_min], [lower[0], v_max])
                    space.append(pgon)

                cnt = 0
                finished = False

                while cnt < len(o)-1:

                    ind = cnt
                    up = upper[cnt]

                    for i in range(cnt+1, len(o)):
                        if lower[i] < up:
                            ind = i
                            up = max(up, upper[i])

                    if ind < len(lower)-1:
                        pgon = interval2polygon([up, v_min], [lower[ind+1], v_max])
                    else:
                        pgon = interval2polygon([up, v_min], [l.distance[len(l.distance)-1], v_max])
                        finished = True
                    space.append(pgon)
                    cnt = ind + 1

                if not finished and max(upper) < l.distance[len(l.distance)-1]:
                    pgon = interval2polygon([max(upper), v_min], [l.distance[len(l.distance)-1], v_max])
                    space.append(pgon)

            else:
                pgon = interval2polygon([l.distance[0], v_min], [l.distance[len(l.distance)-1], v_max])
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

        # check if space intersects the current lanelet segment
        seg = Polygon(np.concatenate((lanelet.right_vertices[[i], :], lanelet.right_vertices[[i+1], :],
                                      lanelet.left_vertices[[i+1], :], lanelet.left_vertices[[i], :])))

        if seg.intersects(o_int):

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


def reduce_space(space, plan, lanelets, param):
    """reduce the space of the drivable area by intersecting with the forward reachable set"""

    # loop over all time steps
    for i in range(len(space)-1):

        # compute forward reachable set
        space_ = reach_set_forward(space[i], param)

        # shift set if moving on to a successor lanelet
        if not plan[i+1] == plan[i] and plan[i+1] in lanelets[plan[i]].successor:
            dist = lanelets[plan[i]].distance
            space_ = translate(space_, -dist[len(dist)-1], 0)

        # intersect forward reachable set with drivable area
        space[i+1] = space[i+1].intersection(space_)

    return space

def best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, param):
    """determine the best sequences of lanelets to drive on that reach the goal state"""

    min_cost = None

    # create initial node
    v = param['v_init']
    space = interval2polygon([param['x0_set'] - 0.01, v - 0.01], [param['x0_set'] + 0.01, v + 0.01])
    x0 = {'step': 0, 'space': space, 'lanelet': param['x0_lane']}

    queue = [Node([x0], [], [], 'none', ref_traj, change_goal, lanelets)]

    # loop until queue empty -> all possible lanelet sequences have been explored
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # remove nodes with costs higher than the current minimum cost for reaching the goal set
        if min_cost is not None:
            for i in range(len(queue)):
                if queue[i].cost > min_cost:
                    queue = queue[:i]
                    break
        if len(queue) == 0:
            break

        # take node from queue
        node = queue.pop()

        # compute drivable area for the current lanelet
        lanelet = lanelets[node.x0[0]['lanelet']]

        if len(node.lanelets) == 0:
            prev = []
        else:
            prev = node.drive_area[len(node.drive_area)-1]

        drive_area, left, right, suc, x0 = compute_drivable_area(lanelet, node.x0, free_space, prev, node.lane_prev, param)

        # check if goal set has been reached
        if param['goal_lane'] is None or lanelet.lanelet_id == param['goal_lane']:

            final_sets = []

            for d in drive_area:
                if param['goal_time_start'] <= d['step'] <= param['goal_time_end'] and \
                                                            d['space'].intersects(param['goal_set']):
                    final_sets.append({'space': d['space'].intersection(param['goal_set']), 'step': d['step']})

            if len(final_sets) > 0:
                node_temp = expand_node(node, final_sets, drive_area, 'final', ref_traj, change_goal, lanelets)
                if min_cost is None or node_temp.cost < min_cost:
                    min_cost = node_temp.cost
                    final_node = deepcopy(node_temp)

        # create child nodes
        for entry in x0:
            queue.append(expand_node(node, entry, drive_area, node.lane_prev, ref_traj, change_goal, lanelets))

        for entry in left:
            if len(entry) > MIN_LANE_CHANGE:
                queue.append(expand_node(node, entry, drive_area, 'right', ref_traj, change_goal, lanelets))

        for entry in right:
            if len(entry) > MIN_LANE_CHANGE:
                queue.append(expand_node(node, entry, drive_area, 'left', ref_traj, change_goal, lanelets))

        for entry in suc:
            queue.append(expand_node(node, entry, drive_area, 'none', ref_traj, change_goal, lanelets))

    return final_node

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
    x0_new = []

    # loop over all time steps up to the final time
    for i in range(x0[cnt]['step'], param['steps']):

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

        # check if driveable area intersects multiple free space segments (can happen if another car comes into lane)
        finished = False

        if len(tmp) > 1:
            space = tmp[0]
            for k in range(1, len(tmp)):
                x0_new.append(create_branch(tmp[k], i+1, free_space, x0[cnt:], lanelet.lanelet_id, param))
        elif len(tmp) == 0:
            finished = True
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

        if finished:
            break

        # check if it is possible to make a lane change to the left
        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:

            for space_left in free_space[lanelet.adj_left][i+1]:
                if space.intersects(space_left) and not(lane_prev == 'left' and cnt_prev < len(prev) and
                                  prev[cnt_prev]['step'] == i+1 and space_left.intersects(prev[cnt_prev]['space'])):
                    left = add_transition(left, space.intersection(space_left), i+1, lanelet.adj_left, param)

        # check if it is possible to make a lane change to the right
        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

            for space_right in free_space[lanelet.adj_right][i + 1]:
                if space.intersects(space_right) and not(lane_prev == 'right' and cnt_prev < len(prev) and
                                   prev[cnt_prev]['step'] == i+1 and space_right.intersects(prev[cnt_prev]['space'])):
                    right = add_transition(right, space.intersection(space_right), i+1, lanelet.adj_right, param)

        # update counter for drivable area on previous lanelet
        if cnt_prev < len(prev) and prev[cnt_prev]['step'] == i+1:
            cnt_prev = cnt_prev + 1

        # store drivable area
        drive_area.append({'space': space, 'step': i+1})

    return drive_area, left, right, successors, x0_new

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

    # create a new transition if the current transition could not be added to any existing transition
    if not found:
        transitions.append([{'space': space, 'step': time_step, 'lanelet': lanelet}])

    return transitions


def create_branch(space, time_step, free_space, x0, lanelet, param):
    """create a new seach path if the driveable area intersects multiple free space segments"""

    # create first element
    x0_new = [{'space': space, 'step': time_step, 'lanelet': lanelet}]

    # loop over all possible transitions from the previous lanelet
    for x in x0:

        # compute forward reachable set
        space = reach_set_forward(space, param)

        # intersect with free space
        for fs in free_space[lanelet][time_step]:
            if fs.intersects(space):
                space = fs.intersection(space)

        # check if space intersects the space from the transition
        if x['space'].intersects(space):
            x0_new.append(x)

        time_step = time_step + 1

    return x0_new

def velocity2trajectory(vel_prof, param):
    """compute the reference trajectory for the given velocity profile"""

    ref_traj = []
    x = param['x0_set']

    for i in range(len(vel_prof)):
        ref_traj.append((x, vel_prof[i]))
        x = x + vel_prof[i] * param['time_step']

    return ref_traj

def refine_plan(seq, ref_traj, lanelets, param):
    """refine the plan by deciding on which lanelets to be on at which points in time"""

    # determine shift in position when changing to a successor lanelet
    offset = offsets_lanelet_sequence(seq.lanelets, lanelets)

    # select best final set from all intersections with the goal set
    min_cost = np.inf

    for fin in seq.x0:

        cost = cost_reference_trajectory(ref_traj, fin, offset[-1])

        if cost < min_cost:
            final_set = deepcopy(fin)
            min_cost = cost

    # initialize lists for storing the lanelet and the corresponding space
    plan = [None for i in range(0, final_set['step']+1)]
    plan[len(plan)-1] = seq.lanelets[len(seq.lanelets)-1]

    space = [None for i in range(0, final_set['step']+1)]
    space[len(space)-1] = final_set['space']

    time_step = final_set['step']

    # loop backward over the lanelet sequence
    for i in range(len(seq.lanelets)-1, -1, -1):

        # check if the previous lanelet is a successor or a left/right lanelet
        is_successor = False

        if i > 0 and seq.lanelets[i] in lanelets[seq.lanelets[i-1]].successor:
            is_successor = True

        # initialize auxiliary variables
        if i > 0:
            cnt = len(seq.drive_area[i-1]) - 1
            while seq.drive_area[i-1][cnt]['step'] >= time_step:
                cnt = cnt - 1
            lanelet_prev = lanelets[seq.lanelets[i - 1]]
            dist = lanelet_prev.distance[len(lanelet_prev.distance) - 1]

        transitions = []

        # loop over all time steps on the current lanelet
        for j in range(time_step-1, seq.drive_area[i][0]['step']-1, -1):

            # propagate set one time step backward in time
            space[j] = reach_set_backward(space[j+1], param)
            space_prev = space[j]

            # intersect with the previous set
            step = j - seq.drive_area[i][0]['step']
            space[j] = space[j].intersection(seq.drive_area[i][step]['space'])
            plan[j] = seq.lanelets[i]

            # check if it is possible to change lanelets in this time step
            if i > 0 and seq.drive_area[i-1][cnt]['step'] == j:
                if is_successor:
                    if space_prev.bounds[0] <= -0.1:
                        space_ = translate(space_prev, dist, 0)
                        space_ = space_.intersection(seq.drive_area[i - 1][cnt]['space'])
                        transitions.append({'space': space_, 'step': j})
                else:
                    if space[j].intersects(seq.drive_area[i - 1][cnt]['space']):
                        space_ = space[j].intersection(seq.drive_area[i - 1][cnt]['space'])
                        transitions.append({'space': space_, 'step': j})
                cnt = cnt - 1

        # select the best transition to take to the previous lanelet
        if i > 0:

            min_cost = np.inf

            if len(transitions) == 0:
                time_step = seq.drive_area[i-1][-1]['step']
                space_ = reach_set_backward(space_prev, param)
                space_ = translate(space_, dist, 0)
                space_ = space_.intersection(seq.drive_area[i - 1][-1]['space'])
            else:
                for t in transitions:
                    cost = cost_reference_trajectory(ref_traj, t, offset[i-1])
                    if cost < min_cost:
                        time_step = t['step']
                        space_ = t['space']
                        min_cost = cost

            space[time_step] = space_
            plan[time_step] = seq.lanelets[i-1]

    return plan, space

def space_lane_changes(space, plan, lanelets, free_space, param):
    """compute the drivable space at a lanelet change in the global coordinate frame"""

    # increase free space by the vehicle dimensions
    space = increase_free_space(space, param)
    free_space = increase_free_space(free_space, param)

    # determine indices for all lane changes
    plan = np.asarray(plan)
    ind = np.where(plan[:-1] != plan[1:])[0]

    time = [[] for i in range(len(ind))]

    # get drivable area (without space for lane changes) in the global coordinate frame
    space_glob = lanelet2global(space, plan, lanelets)

    # add space from left and right lanelet for the beginning since the initial position is not necessarily in lanelet
    for i in range(10):

        for f in free_space[plan[i]][i]:
            if f.intersects(space[i]):
                pgon = lanelet2global([f], [plan[i]], lanelets)
                space_glob[i] = union_robust(space_glob[i], pgon[0])

        if not lanelets[plan[i]].adj_right is None and lanelets[plan[i]].adj_right_same_direction:
            for f in free_space[lanelets[plan[i]].adj_right][i]:
                if f.intersects(space[i]):
                    pgon = lanelet2global([f], [lanelets[plan[i]].adj_right], lanelets)
                    space_glob[i] = union_robust(space_glob[i], pgon[0])

        if not lanelets[plan[i]].adj_left is None and lanelets[plan[i]].adj_left_same_direction:
            for f in free_space[lanelets[plan[i]].adj_left][i]:
                if f.intersects(space[i]):
                    pgon = lanelet2global([f], [lanelets[plan[i]].adj_left], lanelets)
                    space_glob[i] = union_robust(space_glob[i], pgon[0])

        for suc in lanelets[plan[i]].successor:
            for f in free_space[suc][i]:
                pgon = translate(f, lanelets[plan[i]].distance[-1], 0)
                if pgon.intersects(space[i]):
                    pgon = lanelet2global([f], [suc], lanelets)
                    space_glob[i] = union_robust(space_glob[i], pgon[0])

        for suc in lanelets[plan[i]].predecessor:
            for f in free_space[suc][i]:
                pgon = translate(f, -lanelets[suc].distance[-1], 0)
                if pgon.intersects(space[i]):
                    pgon = lanelet2global([f], [suc], lanelets)
                    space_glob[i] = union_robust(space_glob[i], pgon[0])

    # loop over all lane changes
    for i in range(len(ind)):

        lanelet_1 = lanelets[plan[ind[i]]]
        lanelet_2 = lanelets[plan[ind[i]+1]]

        # loop backward in time until previous lane change
        if i == 0:
            start = 0
        else:
            start = ind[i-1]

        for j in range(ind[i], start-1, -1):

            if plan[ind[i]+1] in lanelet_1.successor:

                if space[j].bounds[2] >= lanelet_1.distance[-1]:

                    for f in free_space[lanelet_2.lanelet_id][j]:
                        if f.bounds[0] <= 0:
                            pgon = lanelet2global([f], [lanelet_2.lanelet_id], lanelets)
                            space_glob[j] = union_robust(space_glob[j], pgon[0])
                            time[i].append(j)
                else:
                    break

            else:
                intersection = False

                for f in free_space[lanelet_2.lanelet_id][j]:
                    if f.intersects(space[j]):
                        pgon = lanelet2global([f], [lanelet_2.lanelet_id], lanelets)
                        space_glob[j] = union_robust(space_glob[j], pgon[0])
                        intersection = True
                        time[i].append(j)

                if not intersection:
                    break

        # loop forward in time until the next lane change
        if i == len(ind)-1:
            end = len(space)-1
        else:
            end = ind[i+1]

        for j in range(ind[i], end):

            if plan[ind[i]+1] in lanelet_1.successor:

                if space[j].bounds[2] <= 0:

                    for f in free_space[lanelet_1.lanelet_id][j]:
                        if f.bounds[2] >= lanelet_1.distance[-1]:
                            pgon = lanelet2global([f], [lanelet_1.lanelet_id], lanelets)
                            space_glob[j] = union_robust(space_glob[j], pgon[0])
                            time[i].append(j)
                else:
                    break

            else:
                intersection = False

                for f in free_space[lanelet_1.lanelet_id][j]:
                    if f.intersects(space[j]):
                        pgon = lanelet2global([f], [lanelet_1.lanelet_id], lanelets)
                        space_glob[j] = union_robust(space_glob[j], pgon[0])
                        intersection = True
                        time[i].append(j)

                if not intersection:
                    break

        # sort time steps where lane change is possible
        time[i] = np.asarray(time[i])
        time[i] = np.sort(time[i])

    return space_glob, time

def increase_free_space(space, param):
    """increase the free space by the dimension of the car"""

    space = deepcopy(space)

    if type(space) is dict:

        for k in space.keys():
            for i in range(len(space[k])):
                for j in range(len(space[k][i])):
                    pgon1 = translate(space[k][i][j], 0.5*param['length_max'], 0)
                    pgon2 = translate(space[k][i][j], -0.5*param['length_max'], 0)
                    space[k][i][j] = pgon1.union(pgon2).convex_hull

    else:

        for k in range(len(space)):
            pgon1 = translate(space[k], 0.5*param['length_max'], 0)
            pgon2 = translate(space[k], -0.5*param['length_max'], 0)
            space[k] = pgon1.union(pgon2).convex_hull

    return space

def reference_trajectory(plan, seq, space, vel, time_lane, param, lanelets):
    """compute a desired reference trajectory"""

    # compute suitable velocity profile
    v = desired_velocity(vel)

    # compute corresponding position profile
    x = [space[0].bounds[0]]

    for i in range(len(v)-1):
        x.append(x[-1] + v[i] * param['time_step'])

    # update plan (= lanelet-time-assignment)
    plan = np.asarray(plan)
    lanes = seq.lanelets
    dist = 0

    for i in range(len(lanes)-1):
        ind = np.where(plan[:-1] != plan[1:])[0]
        ind = [-1] + ind.tolist() + [len(plan) - 1]
        if lanes[i+1] in lanelets[lanes[i]].successor:
            for j in range(ind[i]+1, ind[i+2]+1):
                if x[j] - dist < lanelets[lanes[i]].distance[-1]:
                    plan[j] = lanes[i]
                else:
                    plan[j] = lanes[i+1]
            dist = dist + lanelets[lanes[i]].distance[-1]

    # determine indices for all lane changes
    ind = np.where(plan[:-1] != plan[1:])[0]

    # loop over all lanelets the car drives on
    dist = 0
    step = 0
    tmp = [[] for i in range(len(x))]
    center_traj = [deepcopy(tmp) for i in range(len(lanes))]

    for j in range(len(lanes)):

        # loop over all time steps
        for i in range(step, len(x)):

            d = x[i] - dist
            lanelet = lanelets[lanes[j]]

            if d > lanelet.distance[-1]:
                if j < len(lanes) - 1 and lanes[j + 1] in lanelet.successor:
                    dist = dist + lanelet.distance[-1]
                    step = i
                    break

            for k in range(1, len(lanelet.distance)):
                if d <= lanelet.distance[k]:
                    p1 = lanelet.center_vertices[k-1, :]
                    p2 = lanelet.center_vertices[k, :]
                    p = p1 + (p2 - p1) * (d - lanelet.distance[k-1])/(lanelet.distance[k] - lanelet.distance[k-1])
                    center_traj[j][i] = np.transpose(p)
                    break

    # store reference trajectory (without considering lane changes)
    ref_traj = np.zeros((2, len(x)))
    cnt = 0

    for i in range(0, len(plan)):
        if len(center_traj[cnt][i]) == 0:
            ref_traj[:, i] = center_traj[cnt+1][i]
        else:
            ref_traj[:, i] = center_traj[cnt][i]
        if i < len(plan)-1 and plan[i] != plan[i+1]:
            cnt = cnt + 1

    # loop over all lane changes
    for i in range(len(ind)):

        # check if it is a lane change or just a change onto a successor lanelet
        if plan[ind[i]+1] not in lanelets[plan[ind[i]]].successor:

            # compute start and end time step for the lane change
            ind_start = max(time_lane[i][0]+1, ind[i] - np.floor(DES_LANE_CHANGE/2)).astype(int)
            ind_end = min(time_lane[i][-1]-1, ind[i] + np.floor(DES_LANE_CHANGE/2)).astype(int)

            # interpolate between the center trajectories for the two lanes
            for j in range(ind_start, ind_end+1):

                w = 1/(1 + np.exp(-5*(2*((j - ind_start)/(ind_end - ind_start))-1)))
                p = (1-w) * center_traj[i][j] + w * center_traj[i+1][j]
                ref_traj[:, j] = p

    return ref_traj


def desired_velocity(vel):
    """compute a desired velocity for each time step"""

    # select initial and final velocity
    v_init = vel[0][0]

    if vel[-1][0] <= v_init <= vel[-1][1]:
        v_end = v_init
    elif v_init < vel[-1][0]:
        v_end = vel[-1][0]
    else:
        v_end = vel[-1][1]

    v = linear_interpolation(v_init, v_end, len(vel))

    # loop until desired velocity profile if contained in valid velocity set
    v = velocity_recursive(v, vel)

    return v

def velocity_recursive(v, vel):
    """recursive function to refine the velocity profile"""

    # loop over all time steps
    dmax = 0
    ind = None

    for i in range(len(v)):

        if vel[i][0] > v[i]:
            d = vel[i][0] - v[i]
            if d > dmax:
                ind = i
                dmax = d
                vd = vel[i][0]
        elif vel[i][1] < v[i]:
            d = v[i] - vel[i][1]
            if d > dmax:
                ind = i
                dmax = d
                vd = vel[i][1]

    # recursively refine the velocity profile
    if ind is not None:
        v1 = linear_interpolation(v[0], vd, ind + 1)
        v1 = velocity_recursive(v1, vel[:ind+1])

        v2 = linear_interpolation(vd, v[-1], len(v) - ind)
        v2 = velocity_recursive(v2, vel[ind:])

        v = v1[:-1] + v2

    return v


def linear_interpolation(x_start, x_end, length):
    """linear interpolation between x_start and x_end"""

    d = (x_end - x_start)/(length-1)

    return [x_start + d * i for i in range(length)]

def cost_reference_trajectory(ref_traj, area, offset):
    """compute cost based on the distance between the reachable set and the desired reference trajectory"""

    p = Point(ref_traj[area['step']][0] - offset, ref_traj[area['step']][1])

    if area['space'].contains(p):
        return 0
    else:
        return area['space'].exterior.distance(p)

def offsets_lanelet_sequence(seq, lanelets):
    """determine shift in position when changing to a successor lanelet for the given lanelet sequence"""

    offset = [0]

    for i in range(len(seq) - 1):
        if seq[i + 1] in lanelets[seq[i]].successor:
            offset.append(offset[i] + lanelets[seq[i]].distance[-1])
        else:
            offset.append(offset[i])

    return offset

def lanelet2global(space, plan, lanelets):
    """transform free space from lanelet coordinate system to global coordinate system"""

    space_xy = []

    # loop over all time steps
    for i in range(0, len(space)):

        # initialization
        lanelet = lanelets[plan[i]]

        lower = space[i].bounds[0]
        upper = space[i].bounds[2]

        left_vertices = []
        right_vertices = []
        pgons = []

        # catch the case where space exceeds over the lanelet bounds
        if lower < lanelet.distance[0]:
            for pd in lanelet.predecessor:
                d = lanelets[pd].distance[-1]
                pgons.append(lanelet2global([interval2polygon([d + lower, -1], [d, 1])], [pd], lanelets)[0])
            lower = 0

        if upper > lanelet.distance[-1]:
            d = upper - lanelet.distance[-1]
            for sc in lanelet.successor:
                pgons.append(lanelet2global([interval2polygon([0, -1], [d, 1])], [sc], lanelets)[0])
            upper = lanelet.distance[-1]

        # loop over the single segments of the lanelet
        for j in range(0, len(lanelet.distance)-1):

            if lower >= lanelet.distance[j] and lower <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d/np.linalg.norm(d) * (lower - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (lower - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

            if lower <= lanelet.distance[j] <= lanelet.distance[j+1]:

                p_left = lanelet.left_vertices[j]
                left_vertices.append(Point(p_left[0], p_left[1]))

                p_right = lanelet.right_vertices[j]
                right_vertices.append(Point(p_right[0], p_right[1]))

            if upper >= lanelet.distance[j] and upper <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

                break

        # construct the resulting polygon in the global coordinate system
        right_vertices.reverse()
        left_vertices.extend(right_vertices)

        pgon = Polygon(left_vertices)

        # unite with polygons from predecessor and successor lanelets
        for p in pgons:
            pgon = union_robust(pgon, p)

        space_xy.append(pgon)

    return space_xy

def union_robust(pgon1, pgon2):
    """robust union of two polygons removing small unwanted fragments"""

    # compute union using build-in function
    pgon = pgon1.union(pgon2)

    # bloat the polygon by a small cube
    pgon_bloat = pgon

    if isinstance(pgon, MultiPolygon):
        polygons = list(pgon.geoms)
    else:
        polygons = [pgon]

    for p in polygons:

        x, y = p.exterior.coords.xy
        V = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)

        B = interval2polygon([-0.1, -0.1], [0.1, 0.1])

        for i in range(V.shape[1]-1):
            tmp1 = translate(B, V[0, i], V[1, i])
            tmp2 = translate(B, V[0, i+1], V[1, i+1])
            pgon_bloat = pgon_bloat.union(tmp1.union(tmp2).convex_hull)

    # subtract the bloating again
    x, y = pgon_bloat.exterior.coords.xy
    V = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)

    B = interval2polygon([-0.11, -0.11], [0.11, 0.11])
    pgon_diff = pgon_bloat

    for i in range(V.shape[1] - 1):
        tmp1 = translate(B, V[0, i], V[1, i])
        tmp2 = translate(B, V[0, i + 1], V[1, i + 1])
        pgon_diff = pgon_diff.difference(tmp1.union(tmp2).convex_hull)

    return pgon_diff

def expand_node(node, x0, drive_area, lane_prev, ref_traj, change_goal, lanelets):
    """add value for the current step to a given node"""

    # add current lanelet to list of lanelets
    l = deepcopy(node.lanelets)
    l.append(node.x0[0]['lanelet'])

    # truncate drivable area
    for i in range(len(drive_area)):
        if drive_area[i]['step'] == x0[-1]['step']:
            drive_area = drive_area[:i+1]
            break

    # add driveable area to the list
    s = deepcopy(node.drive_area)
    s.append(drive_area)

    # create resulting node
    return Node(x0, l, s, lane_prev, ref_traj, change_goal, lanelets)


class Node:
    """class representing a single node for A*-search"""

    def __init__(self, x0, lanes, drive_area, lane_prev, ref_traj, change_goal, lanelets):
        """class constructor"""

        # store object properties
        self.x0 = x0
        self.lanelets = lanes
        self.drive_area = drive_area
        self.lane_prev = lane_prev
        self.cost = self.cost_function(ref_traj, change_goal, lanelets)

    def cost_function(self, ref_traj, change_goal, lanelets):
        """compute cost function value for the node"""

        # determine shift in position when changing to a successor lanelet
        offset = offsets_lanelet_sequence(self.lanelets, lanelets)

        # determine cost from deviation to the desired reference trajectory
        diff = np.inf * np.ones(len(ref_traj))

        for i in range(len(self.x0)):
            diff[self.x0[i]['step']] = 0

        for i in range(len(ref_traj)):

            # loop over all lanelets
            for j in range(len(self.drive_area)):

                diff_cur = np.inf
                p = Point(ref_traj[i][0] - offset[j], ref_traj[i][1])

                # loop over all time steps
                for set in self.drive_area[j]:

                    # compute distance from desired velocity profile
                    if set['step'] == i:
                        if set['space'].contains(p):
                            diff_cur = 0
                        else:
                            diff_cur = set['space'].exterior.distance(p)
                    elif set['step'] > i:
                        break

                # total distance -> minimum over the distance for all lanelets
                if diff_cur < diff[i]:
                    diff[i] = diff_cur

                if diff[i] == 0:
                    break

        diff = np.extract(diff < np.inf, diff)

        # determine number of lane changes
        lane_changes = 0

        for i in range(1, len(self.lanelets)):
            l = lanelets[self.lanelets[i-1]]
            if not self.lanelets[i] in l.successor:
                lane_changes = lane_changes + 1

        # number of expected lane changes to reach the goal set
        expect_changes = 0

        if self.lane_prev != 'final':
            expect_changes = change_goal[self.x0[0]['lanelet']]
            if len(self.lanelets) > 0 and not self.x0[0]['lanelet'] in lanelets[self.lanelets[-1]].successor:
                expect_changes = expect_changes + 1

        return W_LANE_CHANGE * (lane_changes + expect_changes) + W_DIST * np.sum(diff)
