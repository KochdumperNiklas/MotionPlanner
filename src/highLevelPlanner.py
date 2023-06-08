import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.affinity import affine_transform
from shapely.affinity import translate
from shapely.ops import nearest_points
from copy import deepcopy
from commonroad.scenario.obstacle import StaticObstacle
from commonroad.geometry.shape import ShapeGroup
from commonroad.visualization.util import collect_center_line_colors

# weighting factors for the cost function
W_LANE_CHANGE = 1000
W_DIST = 1

# safety distance to other cars
DIST_SAFE = 2

# minimum number of consecutive time steps required to perform a lane change
MIN_LANE_CHANGE = 2

# desired number of time steps for performing a lane change
DES_LANE_CHANGE = 20

# desired acceleration
A_DES = 1

def highLevelPlanner(scenario, planning_problem, param, priority=False):
    """decide on which lanelets to be at all points in time"""

    # extract required information from the planning problem
    param, lanelets, speed_limit = initialization(scenario, planning_problem, param)

    # compute free space on each lanelet for each time step
    free_space, partially_occupied = free_space_lanelet(lanelets, scenario, speed_limit, param)

    # compute distance to goal lanelet and number of required lane changes for each lanelet
    change_goal, dist_goal = distance2goal(lanelets, param)

    # compute the desired velocity profile over time
    vel_prof = velocity_profile(dist_goal, speed_limit, param)

    # compute the desired reference trajectory
    ref_traj = velocity2trajectory(vel_prof, param)

    # determine best sequence of lanelets to drive on
    seq = best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, partially_occupied, priority, param)

    # refine the plan: decide on which lanelet to be on for all time steps
    plan, space = refine_plan(seq, ref_traj, lanelets, param)

    # determine drivable space for lane changes
    space_all, time_lane = space_lane_changes(space, plan, lanelets, free_space, partially_occupied, param)

    # shrink space by intersecting with the forward reachable set
    space = reduce_space(space, plan, lanelets, param)

    # extract the safe velocity intervals at each time point
    vel = [(s.bounds[1], s.bounds[3]) for s in space]

    # compute a desired reference trajectory
    ref_traj, plan = reference_trajectory(plan, seq, space, vel_prof, time_lane, param, lanelets)

    # resolve issue with spaces consisting of multiple distinct polygons
    space_all = remove_multi_polyogns(space_all, ref_traj)

    return plan, vel, space_all, ref_traj


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

    pgon = interval2polygon([x0[0] - 0.01, x0[1] - 0.01], [x0[0] + 0.01, x0[1] + 0.01])
    x0_id = []
    x0_set = []

    for id in lanelets.keys():
        if lanelets[id].polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id.append(id)
            x0_space_start, x0_space_end, _, _ = projection_lanelet_centerline(lanelets[id], pgon)
            x0_set.append(0.5 * (x0_space_start + x0_space_end))

    param['x0_lane'] = x0_id
    param['x0_set'] = x0_set
    param['x0'] = x0
    param['orientation'] = planning_problem.initial_state.orientation

    # extract parameter for the goal set
    param['goal'] = []

    for i in range(len(planning_problem.goal.state_list)):

        goal_state = planning_problem.goal.state_list[i]
        param_ = {}
        shapes = 1

        if hasattr(goal_state, 'position') and isinstance(goal_state.position, ShapeGroup):
            shapes = len(goal_state.position.shapes)
        elif not planning_problem.goal.lanelets_of_goal_position is None:
            shapes = len(list(planning_problem.goal.lanelets_of_goal_position.values())[i])

        for j in range(shapes):

            param_['time_start'] = goal_state.time_step.start
            param_['time_end'] = goal_state.time_step.end

            if hasattr(goal_state, 'position'):

                if isinstance(goal_state.position, ShapeGroup):
                    set = get_shapely_object(goal_state.position.shapes[j])
                else:
                    set = get_shapely_object(goal_state.position)

                param_['space'] = set

                if planning_problem.goal.lanelets_of_goal_position is None:
                    for id in lanelets.keys():
                        if lanelets[id].polygon.shapely_object.intersects(set):
                            param_['lane'] = id
                else:
                    param_['lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[i][j]

                goal_space_start, goal_space_end, _, _ = projection_lanelet_centerline(lanelets[param_['lane']], set)

            else:

                if planning_problem.goal.lanelets_of_goal_position is None:
                    param_['lane'] = None
                    param_['space'] = None
                    goal_space_start = -100000
                    goal_space_end = 100000
                else:
                    param_['lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[i][j]
                    l = lanelets[param_['lane']]
                    param_['space'] = l.polygon.shapely_object
                    goal_space_start, goal_space_end, _, _ = projection_lanelet_centerline(l, l.polygon.shapely_object)

            if hasattr(goal_state, 'velocity'):
                v = goal_state.velocity
                param_['set'] = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])
            else:
                param_['set'] = interval2polygon([goal_space_start, param['v_min']], [goal_space_end, param['v_max']])

            param['goal'].append(deepcopy(param_))

    # determine the speed limit for each lanelet
    speed_limit = {}

    for id in lanelets.keys():
        limits = []
        for sign_id in lanelets[id].traffic_signs:
            sign = scenario.lanelet_network.find_traffic_sign_by_id(sign_id)
            for elem in sign.traffic_sign_elements:
                if elem.traffic_sign_element_id.name == "MAX_SPEED":
                    limits.append(float(elem.additional_values[0]))
        if len(limits) > 0:
            speed_limit[id] = min(limits)
        else:
            speed_limit[id] = None

    return param, lanelets, speed_limit

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

        if lanelet.polygon.shapely_object.is_valid:
            set = lanelet.polygon.shapely_object.intersection(pgon)
        else:
            lane = lanelet.polygon.shapely_object.buffer(0)
            set = lane.intersection(pgon)

        if isinstance(set, Polygon):
            res = True

    return res

def distance2goal(lanelets, param):
    """compute the distance to the target lanelet for each lanelet"""

    # initialize dictionaries
    default = {}
    for l in lanelets.keys():
        default[l] = np.inf

    change_all = deepcopy(default)
    dist_all = []

    # loop over all goal sets
    for goal in param['goal']:

        # catch the case where no goal lane is provided
        if goal['lane'] is None:

            default = {}
            for l in lanelets.keys():
                default[l] = 0

            dist = deepcopy(default)
            change = deepcopy(default)

        else:

            # initialize distance to goal lanelet
            dist = {goal['lane']: 0}
            change = {goal['lane']: 0}

            # initialize queue with goal lanelet
            queue = []
            queue.append(goal['lane'])

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

        # combine with values for other goal sets
        for l in lanelets.keys():
            change_all[l] = min(change_all[l], change[l])

        dist_all.append(deepcopy(dist))

    return change_all, dist_all

def velocity_profile(dist, speed_limit, param):
    """compute the desired velocity profile over time"""

    # select desired velocity
    vel_des = np.inf

    for id in param['x0_lane']:
        if not speed_limit[id] is None:
            vel_des = min(vel_des, speed_limit[id])

    for goal in param['goal']:
        if goal['lane'] is not None and speed_limit[goal['lane']] is not None:
            vel_des = min(vel_des, speed_limit[goal['lane']])

    if vel_des == np.inf:
        vel_des = param['v_init']

    # loop over all possible combinations of initial sets and goal sets
    val = np.inf

    for i in range(len(param['x0_lane'])):
        for j in range(len(param['goal'])):

            goal = param['goal'][j]

            # compute velocity profile
            vel_ = velocity_profile_single(dist[j], vel_des, i, goal, param)

            # select best velocity profile (= minimum distance to desired velocity)
            val_ = abs(vel_[-1] - vel_des)
            if val_ < val:
                vel = deepcopy(vel_)
                val = val_

    return vel

def velocity_profile_single(dist, vel_des, x0_ind, goal, param):
    """compute the desired velocity profile for a single combination of inital and goal set"""

    # calculate minimum and maximum velocity required at the end of the time horizon
    vel_min = goal['set'].bounds[1]
    vel_max = goal['set'].bounds[3]

    if goal['lane'] is not None:

        # calculate minimum and maximum distance from initial state to goal set
        dist_min = dist[param['x0_lane'][x0_ind]] - param['x0_set'][x0_ind] + goal['set'].bounds[0]
        dist_max = dist[param['x0_lane'][x0_ind]] - param['x0_set'][x0_ind] + goal['set'].bounds[2]

        # calculate minimum and maximum final velocities required to reach the goal set using a linear velocity profile
        vel_min_ = 2*dist_min/(goal['time_end']*param['time_step']) - param['v_init']
        vel_max_ = 2*dist_max/(goal['time_start']*param['time_step']) - param['v_init']

        # use a quadratic velocity profile if a linear one is not sufficient
        if vel_min_ > vel_max or vel_max_ < vel_min:
            if vel_max_ < vel_min:
                vel_final = vel_min
                x_final = dist_max
                t_final = goal['time_start']
            else:
                vel_final = vel_min
                x_final = dist_min
                t_final = goal['time_end']

            b = -5.0/6.0 * (x_final - 0.5*t_final*(param['v_init'] + vel_final))/t_final**3
            a = (vel_final - param['v_init'] - b * t_final ** 2)/t_final

            vel = [param['v_init'] + a * t + b * t**2 for t in range(param['steps']+1)]

            return vel
        else:
            vel_min = max(vel_min, vel_min_)
            vel_max = min(vel_max, vel_max_)

    acc = max(-A_DES, min(A_DES, (vel_des - param['v_init'])/(param['time_step'] * param['steps'])))
    vel = list(param['v_init'] + acc * param['time_step'] * np.arange(0, param['steps'] + 1))

    # calculate velocity profile
    if vel[-1] < vel_min:
        vel = [param['v_init'] + (vel_min - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]
    elif vel[-1] > vel_max:
        vel = [param['v_init'] + (vel_max - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]

    return vel

def free_space_lanelet(lanelets, scenario, speed_limit, param):
    """compute free space on each lanelet for all time steps"""

    obstacles = scenario.obstacles
    free_space_all = []
    tmp = [deepcopy([[] for i in range(0, param['steps']+1)]) for j in range(len(lanelets))]
    occupied_space = dict(zip(lanelets.keys(), deepcopy(tmp)))
    partially_occupied = dict(zip(lanelets.keys(), deepcopy(tmp)))

    # loop over all obstacles
    for obs in obstacles:

        # distinguish static and dynamic obstacles
        if isinstance(obs, StaticObstacle):

            # determine lanelets that are affected by the obstacle
            pgon = get_shapely_object(obs.obstacle_shape)

            if obs.initial_shape_lanelet_ids is None:
                intersecting_lanes = set()
                for id in lanelets.keys():
                    if intersects_lanelet(lanelets[id], pgon):
                        intersecting_lanes.add(id)
            else:
                intersecting_lanes = obs.initial_shape_lanelet_ids

            # loop over all affected lanelets
            for id in intersecting_lanes:

                l = lanelets[id]
                d_min = l.distance[0]
                d_max = l.distance[-1]

                # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                dist_min, dist_max, y_min, y_max = projection_lanelet_centerline(l, pgon)
                offset = 0.5 * param['length_max'] + DIST_SAFE

                # loop over all time steps
                for i in range(param['steps'] + 1):
                    space = (max(d_min, dist_min - offset), min(d_max, dist_max + offset))
                    occupied_space[id][i].append({'space': space, 'width': (y_min, y_max)})

                # check if occupied space extends to predecessor lanelet
                if dist_min - offset < 0:
                    for ind in l.predecessor:
                        for i in range(param['steps'] + 1):
                            d = lanelets[ind].distance[-1]
                            space = (d + dist_min - offset, d)
                            occupied_space[ind][i].append({'space': space, 'width': (y_min, y_max)})

                # check if occupied space extends to successor lanelet
                if dist_max + offset > l.distance[-1]:
                    for ind in l.successor:
                        for i in range(param['steps'] + 1):
                            d = dist_max + offset - l.distance[-1]
                            occupied_space[ind][i].append({space: (0, d), 'width': (y_min, y_max)})

        else:

            # determine lanelets that are affected by the obstacle
            if obs.prediction.shape_lanelet_assignment is None:
                intersecting_lanes = {}
                for o in obs.prediction.occupancy_set:
                    intersecting_lanes[o.time_step] = set()
                    pgon = get_shapely_object(o.shape)
                    for id in lanelets.keys():
                        if intersects_lanelet(lanelets[id], pgon):
                            intersecting_lanes[o.time_step].add(id)
            else:
                intersecting_lanes = obs.prediction.shape_lanelet_assignment

            # loop over all time steps
            for o in obs.prediction.occupancy_set:

                if o.time_step < param['steps'] + 1:

                    pgon = get_shapely_object(o.shape)

                    # loop over all affected lanelets
                    for id in intersecting_lanes[o.time_step]:

                        l = lanelets[id]
                        d_min = l.distance[0]
                        d_max = l.distance[-1]

                        # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                        dist_min, dist_max, y_min, y_max = projection_lanelet_centerline(l, pgon)
                        offset = 0.5 * param['length_max'] + DIST_SAFE
                        space = (max(d_min, dist_min - offset), min(d_max, dist_max + offset))
                        occupied_space[id][o.time_step].append({'space': space, 'width': (y_min, y_max)})

                        # check if occupied space extends to predecessor lanelet
                        if dist_min - offset < 0:
                            for ind in l.predecessor:
                                d = lanelets[ind].distance[-1]
                                space = (d + dist_min - offset, d)
                                occupied_space[ind][o.time_step].append({'space': space, 'width': (y_min, y_max)})

                        # check if occupied space extends to successor lanelet
                        if dist_max + offset > l.distance[-1]:
                            for ind in l.successor:
                                d = dist_max + offset - l.distance[-1]
                                occupied_space[ind][o.time_step].append({'space': (0, d), 'width': (y_min, y_max)})

    # unite occupied spaces that belong to the same time
    for id in lanelets.keys():

        # set speed-limit for the lanelet
        v_min = param['v_min']
        if speed_limit[id] is None:
            v_max = param['v_max']
        else:
            v_max = speed_limit[id]

        # compute width of the lanelet
        width_min, width_max = width_lanelet(lanelets[id])

        # loop over all time steps
        for j in range(len(occupied_space[id])):

            o = occupied_space[id][j]

            if len(o) > 0:

                list_new = []

                o.sort(key=lambda i: i['space'][0])
                lower = [i['space'][0] for i in o]
                upper = [i['space'][1] for i in o]
                width = [i['width'] for i in o]

                cnt = 0

                while cnt <= len(o) - 1:

                    ind = cnt
                    up = upper[cnt]
                    y_max = width[cnt][1]
                    y_min = width[cnt][0]

                    # unite with intersecting obstacles
                    for i in range(cnt + 1, len(o)):
                        if lower[i] < up:
                            ind = i
                            up = max(up, upper[i])
                            y_max = max(width[i][1], y_max)
                            y_min = min(width[i][0], y_min)

                    # check if occupied space is small enough to still drive on the lane
                    occupied = 'all'

                    if width_max - y_max > y_min - width_min:
                        if width_max - y_max > param['width'] + 0.3:
                            occupied = 'right'
                            w = (width_min, y_max - 0.2)
                    else:
                        if y_min - width_min > param['width'] + 0.3:
                            occupied = 'left'
                            w = (y_min + 0.2, width_max)

                    # add obstacle to the list
                    space = (lower[cnt], up)

                    if occupied == 'all':
                        list_new.append(space)
                    else:
                        pgon = interval2polygon([space[0], v_min-1], [space[1], v_max+1])
                        partially_occupied[id][j].append({'space': pgon, 'side': occupied, 'width': w})

                    cnt = ind + 1

                occupied_space[id][j] = deepcopy(list_new)

    # loop over all lanelets and compute free space
    for id in lanelets.keys():

        l = lanelets[id]
        free_space = []

        v_min = param['v_min']
        if speed_limit[id] is None:
            v_max = param['v_max']
        else:
            v_max = speed_limit[id]

        for o in occupied_space[id]:

            if len(o) > 0:

                space = []

                if o[0][0] > l.distance[0]:
                    pgon = interval2polygon([l.distance[0], v_min], [o[0][0], v_max])
                    space.append(pgon)

                for i in range(len(o)-1):
                    pgon = interval2polygon([o[i][1], v_min], [o[i+1][0], v_max])
                    space.append(pgon)

                if o[-1][1] < l.distance[-1]:
                    pgon = interval2polygon([o[-1][1], v_min], [l.distance[-1], v_max])
                    space.append(pgon)

            else:
                pgon = interval2polygon([l.distance[0], v_min], [l.distance[len(l.distance)-1], v_max])
                space = [pgon]

            free_space.append(space)

        free_space_all.append(free_space)

    free_space_all = dict(zip(lanelets.keys(), free_space_all))

    # remove free space for lanelets that are blocked by a red traffic light
    if len(scenario.lanelet_network.traffic_lights) > 0:

        # loop over all time steps
        for i in range(0, param['steps']):
            status = collect_center_line_colors(scenario.lanelet_network, scenario.lanelet_network.traffic_lights, i)
            for l in status.keys():
                if status[l].value == 'red':
                    if not l in param['x0_lane']:
                        free_space_all[l][i] = []
                    for s in lanelets[l].predecessor:
                        pgon = interval2polygon([lanelets[s].distance[-1] - 0.5 * param['length_max'], -1000],
                                                [lanelets[s].distance[-1], 1000])
                        is_init = False
                        if s in param['x0_lane']:
                            for j in range(len(param['x0_lane'])):
                                if param['x0_lane'] == s:
                                    x0 = param['x0_set']
                                    if pgon.bounds[0] < x0 < pgon.bounds[2]:
                                        is_init = True
                                    break
                        if not is_init:
                            cnt = len(free_space_all[s][i])
                            for j in range(len(free_space_all[s][i])-1, -1, -1):
                                if free_space_all[s][i][j].intersects(pgon):
                                    if pgon.contains(free_space_all[s][i][j]):
                                        cnt = j
                                    else:
                                        free_space_all[s][i][j] = free_space_all[s][i][j].difference(pgon)
                                        break
                            free_space_all[s][i] = free_space_all[s][i][0:cnt]

    return free_space_all, partially_occupied


def projection_lanelet_centerline(lanelet, pgon):
    """project a polygon to the center line of the lanelet to determine the occupied space"""

    # intersect polygon with lanelet
    if lanelet.polygon.shapely_object.is_valid:
        o_int = lanelet.polygon.shapely_object.intersection(pgon)
    else:
        lane = lanelet.polygon.shapely_object.buffer(0)
        o_int = lane.intersection(pgon)

    vx, vy = o_int.exterior.coords.xy
    V = np.stack((vx, vy))

    # initialize minimum and maximum range of the projection
    dist_max = -np.inf
    dist_min = np.inf
    y_max = -np.inf
    y_min = np.inf

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

            # compute width of the lanelet that is occupied
            V_ = np.array([[-diff[0, 1], diff[0, 0]]]) @ (V - np.transpose(lanelet.center_vertices[[i], :]))

            y_max = max(y_max, max(V_[0]))
            y_min = min(y_min, min(V_[0]))

    return dist_min, dist_max, y_min, y_max

def width_lanelet(lanelet):
    """compute the width of a lanelet"""

    width_max = -np.inf
    width_min = np.inf

    # loop over all centerline segments
    for i in range(0, len(lanelet.distance) - 1):

        # compute lanelet width
        d = lanelet.right_vertices[[i], :] - lanelet.center_vertices[[i], :]
        width_min = min(width_min, -np.linalg.norm(d))

        d = lanelet.left_vertices[[i], :] - lanelet.center_vertices[[i], :]
        width_max = max(width_max, np.linalg.norm(d))

    return width_min, width_max

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

def best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, partially_occupied, priority, param):
    """determine the best sequences of lanelets to drive on that reach the goal state"""

    min_cost = None
    is_priority = False

    # create initial nodes
    queue = []
    v = param['v_init']

    for i in range(len(param['x0_lane'])):

        space = interval2polygon([param['x0_set'][i] - 0.01, v - 0.01], [param['x0_set'][i] + 0.01, v + 0.01])
        x0 = {'step': 0, 'space': space, 'lanelet': param['x0_lane'][i]}
        queue.append(Node([x0], [], [], 'none', ref_traj, change_goal, lanelets))

    # loop until queue empty -> all possible lanelet sequences have been explored
    while len(queue) > 0:

        # sort the queue
        queue.sort(key=lambda i: i.cost)

        # remove nodes with costs higher than the current minimum cost for reaching the goal set
        if min_cost is not None and (not priority or is_priority):
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

        drive_area, left, right, suc, x0 = compute_drivable_area(lanelet, node.x0, free_space, prev, node.lane_prev,
                                                                 partially_occupied, param)

        # check if goal set has been reached
        for i in range(len(param['goal'])):

            goal = param['goal'][i]

            if goal['lane'] is None or lanelet.lanelet_id == goal['lane']:

                final_sets = []

                for d in drive_area:
                    if goal['time_start'] <= d['step'] <= goal['time_end'] and d['space'].intersects(goal['set']):
                        final_sets.append({'space': d['space'].intersection(goal['set']), 'step': d['step']})

                if len(final_sets) > 0:
                    node_temp = expand_node(node, final_sets, drive_area, 'final', ref_traj, change_goal, lanelets)
                    if min_cost is None or node_temp.cost < min_cost or (priority and i == 0 and not is_priority):
                        min_cost = node_temp.cost
                        final_node = deepcopy(node_temp)
                        if i == 0:
                            is_priority = True

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

def compute_drivable_area(lanelet, x0, free_space, prev, lane_prev, partially_occupied, param):
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

                    space_ = space.intersection(space_left)
                    space_ = intersection_partially_occupied(space_, partially_occupied, i + 1, lanelet, 'left')

                    if not space_.is_empty:
                        left = add_transition(left, space_, i+1, lanelet.adj_left, param)

        # check if it is possible to make a lane change to the right
        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

            for space_right in free_space[lanelet.adj_right][i + 1]:
                if space.intersects(space_right) and not(lane_prev == 'right' and cnt_prev < len(prev) and
                                   prev[cnt_prev]['step'] == i+1 and space_right.intersects(prev[cnt_prev]['space'])):

                    space_ = space.intersection(space_right)
                    space_ = intersection_partially_occupied(space_, partially_occupied, i + 1, lanelet, 'right')

                    if not space_.is_empty:
                        right = add_transition(right, space_, i+1, lanelet.adj_right, param)

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

    cnt = 0

    # create first element
    x0_new = [{'space': space, 'step': time_step, 'lanelet': lanelet}]

    # loop over all possible transitions from the previous lanelet
    if x0[0]['step'] >= time_step:

        for i in range(time_step, param['steps']):

            # compute forward reachable set
            space = reach_set_forward(space, param)

            # check if it is possible to lane change in from previous lanelet
            lane_change = False

            if cnt+1 < len(x0)-1 and x0[cnt+1]['step'] == i+1 and x0[cnt+1]['space'].intersects(space):
                space = space.union(x0[cnt+1]['space'])
                space = space.convex_hull
                cnt = cnt + 1
                lane_change = True

            # intersect with free space
            for fs in free_space[lanelet][time_step]:
                if fs.intersects(space):
                    space = fs.intersection(space)

            # add new transition
            if lane_change:
                x0_new.append({'space': space, 'step': i, 'lanelet': lanelet})

    return x0_new

def intersection_partially_occupied(space, partially_occupied, step, lanelet, side):
    """remove partially occupied space from space for lane change"""

    # remove partially occupied space for current lanelet
    for o in partially_occupied[lanelet.lanelet_id][step]:
        if o['side'] == side and space.intersects(o['space']):
            space = space.difference(o['space'])

    # remove partially occupied space for the lanelet after the lane change
    if side == 'left':
        for o in partially_occupied[lanelet.adj_left][step]:
            if o['side'] == 'right' and space.intersects(o['space']):
                space = space.difference(o['space'])
    else:
        for o in partially_occupied[lanelet.adj_right][step]:
            if o['side'] == 'left' and space.intersects(o['space']):
                space = space.difference(o['space'])

    return space

def velocity2trajectory(vel_prof, param):
    """compute the reference trajectory for the given velocity profile"""

    ref_traj = []
    x = param['x0_set'][0]

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
                    if space_prev.bounds[0] <= 0:
                        space_ = translate(space_prev, dist, 0)
                        space_ = space_.intersection(seq.drive_area[i - 1][cnt]['space'])
                        if not isinstance(space_, Polygon):
                            space_ = space_.convex_hull
                        transitions.append({'space': space_, 'step': j})
                else:
                    if space[j].intersects(seq.drive_area[i - 1][cnt]['space']):
                        space_ = space[j].intersection(seq.drive_area[i - 1][cnt]['space'])
                        if not isinstance(space_, Polygon):
                            space_ = space_.convex_hull
                        transitions.append({'space': space_, 'step': j})
                cnt = cnt - 1

            # catch special case where time steps do not overlap
            if i > 0 and is_successor and j == seq.drive_area[i][0]['step'] and seq.drive_area[i-1][-1]['step'] == j-1:
                space_ = translate(space_prev, dist, 0)
                transitions.append({'space': space_, 'step': j})

        # select the best transition to take to the previous lanelet
        if i > 0:

            min_cost = np.inf

            for t in transitions:
                cost = cost_reference_trajectory(ref_traj, t, offset[i-1])
                if cost < min_cost:
                    time_step = t['step']
                    space_ = t['space']
                    min_cost = cost

            space[time_step] = space_
            plan[time_step] = seq.lanelets[i-1]

    return plan, space

def space_lane_changes(space, plan, lanelets, free_space, partially_occupied, param):
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

    # add free space on the same lanelet to drive on
    for i in range(0, len(plan)):
        for f in free_space[plan[i]][i]:
            if f.intersects(space[i]):
                pgon = lanelet2global([f], [plan[i]], lanelets)
                space_glob[i] = union_robust(space_glob[i], pgon[0])

    # add space from left and right lanelet for the beginning since the initial position is not necessarily in lanelet
    for i in range(min(10, len(plan))):

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

    # if initial state is on multiple lanelets, add the space on these lanelets to the free space
    for i in range(len(param['x0_lane'])):
        l = param['x0_lane'][i]
        if l != plan[0]:
            cnt = 0
            while cnt < len(plan) and plan[cnt] == plan[0]:
                for f in free_space[l][cnt]:
                    if f.bounds[0] <= param['x0_set'][i] <= f.bounds[2]:
                        pgon = lanelet2global([f], [l], lanelets)
                        if pgon[0].intersects(space_glob[cnt]):
                            space_glob[cnt] = union_robust(space_glob[cnt], pgon[0])
                cnt = cnt + 1

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

    # remove partially occupied space from the free space
    lanes = np.unique(plan)

    for i in range(len(space_glob)):
        for l in lanes:
            for p in partially_occupied[l][i]:

                p1 = translate(p['space'], -0.5*param['length_max'], 0)
                p2 = translate(p['space'], 0.5*param['length_max'], 0)

                sp = lanelet2global([p1.intersection(p2)], [l], lanelets, p['width'])[0]

                if sp.intersects(space_glob[i]):
                    space_glob[i] = space_glob[i].difference(sp)

    return space_glob, time

def remove_multi_polyogns(space, ref_traj):
    """select the best connected space for the case that space consists of multiple disconnected polygons"""

    # loop over all time steps
    for i in range(len(space)):

        if space[i].geom_type == 'MultiPolygon':

            polygons = list(space[i].geoms)
            p = Point(ref_traj[0, i], ref_traj[1, i])

            # loop over all disconnected polygons to select the one that contains the reference trajectory
            for poly in polygons:
                if poly.contains(p):
                    space[i] = poly
                    break

    return space

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

def reference_trajectory(plan, seq, space, vel_prof, time_lane, param, lanelets):
    """compute a desired reference trajectory"""

    # compute suitable velocity profile
    x, v = trajectory_position_velocity(space, plan, vel_prof, lanelets, param)

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

            if d >= lanelet.distance[-1]:
                if j < len(lanes) - 1 and lanes[j + 1] in lanelet.successor:
                    dist = dist + lanelet.distance[-1]
                    step = i
                    break

            for k in range(1, len(lanelet.distance)):
                if d <= lanelet.distance[k] + 1e-10:
                    p1 = lanelet.center_vertices[k-1, :]
                    p2 = lanelet.center_vertices[k, :]
                    p = p1 + (p2 - p1) * (d - lanelet.distance[k-1])/(lanelet.distance[k] - lanelet.distance[k-1])
                    center_traj[j][i] = np.transpose(p)
                    break

    # store reference trajectory (without considering lane changes)
    ref_traj = np.zeros((2, len(plan)))
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
        if (lanelets[plan[ind[i]]].adj_right is not None and lanelets[plan[ind[i]]].adj_right == plan[ind[i]+1]) or \
                (lanelets[plan[ind[i]]].adj_left is not None and lanelets[plan[ind[i]]].adj_left == plan[ind[i] + 1]):

            # compute start and end time step for the lane change
            ind_start = max(time_lane[i][0]+1, ind[i] - np.floor(DES_LANE_CHANGE/2)).astype(int)
            ind_end = min(time_lane[i][-1]-1, ind[i] + np.floor(DES_LANE_CHANGE/2)).astype(int)

            # interpolate between the center trajectories for the two lanes
            for j in range(ind_start, ind_end+1):

                w = 1/(1 + np.exp(-5*(2*((j - ind_start)/(ind_end - ind_start))-1)))
                p = (1-w) * center_traj[i][j] + w * center_traj[i+1][j]
                ref_traj[:, j] = p

    return ref_traj, plan

def trajectory_position_velocity(space, plan, vel_prof, lanelets, param):
    """compute the desired space-velocity trajectory"""

    # initialization
    dt = param['time_step']
    a_max = param['a_max']

    v = np.asarray(vel_prof)
    x = np.zeros(v.shape)

    for i in range(len(param['x0_lane'])):
        if param['x0_lane'][i] == plan[0]:
            x[0] = param['x0_set'][i]
            break

    dist = 0

    # loop over all time steps
    for i in range(len(plan)-1):

        # shift space if moving on to a successor lanelet
        if plan[i+1] != plan[i] and plan[i+1] in lanelets[plan[i]].successor:
            dist = dist + lanelets[plan[i]].distance[-1]

        space[i+1] = translate(space[i+1], dist, 0)

        # compute the next position when driving with the desired velocity
        a = (v[i+1] - v[i])/dt
        x[i+1] = x[i] + v[i]*dt + 0.5 * a * dt**2

        # check if driving the desired velocity profile is feasible
        if not space[i+1].contains(Point(x[i+1], v[i+1])):

            # determine the best feasible acceleration
            p1 = (x[i] + v[i]*dt + 0.5 * a_max * dt**2, v[i] + a_max * dt)
            p2 = (x[i] + v[i]*dt - 0.5 * a_max * dt**2, v[i] - a_max * dt)
            pgon = space[i+1].intersection(LineString([p1, p2]))

            if not pgon.is_empty:
                p = nearest_points(pgon, Point(x[i+1], v[i+1]))[0]
            elif space[i+1].exterior.distance(Point(p1[0], p1[1])) < 1e-10:
                p = Point(p1[0], p1[1])
            elif space[i+1].exterior.distance(Point(p2[0], p2[1])) < 1e-10:
                p = Point(p2[0], p2[1])
            else:
                p = nearest_points(space[i + 1], LineString([p1, p2]))[0]
                if space[i+1].exterior.distance(p) > 1e-10:
                    raise Exception("Space not driveable!")

            x[i+1] = p.x
            v[i+1] = p.y

    return x, v

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

def lanelet2global(space, plan, lanelets, width=None):
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
                pgons.append(lanelet2global([interval2polygon([d + lower, -1], [d, 1])], [pd], lanelets, width)[0])
            lower = 0

        if upper > lanelet.distance[-1]:
            d = upper - lanelet.distance[-1]
            for sc in lanelet.successor:
                pgons.append(lanelet2global([interval2polygon([0, -1], [d, 1])], [sc], lanelets, width)[0])
            upper = lanelet.distance[-1]

        # loop over the single segments of the lanelet
        for j in range(0, len(lanelet.distance)-1):

            if lower >= lanelet.distance[j] and lower <= lanelet.distance[j+1]:

                frac = (lower - lanelet.distance[j])/(lanelet.distance[j+1] - lanelet.distance[j])

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d * frac
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d * frac
                right_vertices.append(Point(p_right[0], p_right[1]))

            if lower <= lanelet.distance[j] <= upper:

                p_left = lanelet.left_vertices[j]
                left_vertices.append(Point(p_left[0], p_left[1]))

                p_right = lanelet.right_vertices[j]
                right_vertices.append(Point(p_right[0], p_right[1]))

            if upper >= lanelet.distance[j] and upper <= lanelet.distance[j+1]:

                frac = (upper - lanelet.distance[j]) / (lanelet.distance[j + 1] - lanelet.distance[j])

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d * frac
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d * frac
                right_vertices.append(Point(p_right[0], p_right[1]))

                break

        # restrict to the specified width
        if width is not None:
            for i in range(len(left_vertices)):
                left = np.array([left_vertices[i].x, left_vertices[i].y])
                right = np.array([right_vertices[i].x, right_vertices[i].y])
                center = 0.5*(left + right)
                d = left - right
                d = d / np.linalg.norm(d)
                left = center + width[0]*d
                right = center + width[1]*d
                left_vertices[i] = Point(left[0], left[1])
                right_vertices[i] = Point(right[0], right[1])

        # construct the resulting polygon in the global coordinate system
        right_vertices.reverse()
        left_vertices.extend(right_vertices)

        pgon = Polygon(left_vertices)

        if not pgon.is_valid:
            pgon = pgon.buffer(0)

        # unite with polygons from predecessor and successor lanelets
        for p in pgons:
            pgon = union_robust(pgon, p)

        space_xy.append(pgon)

    return space_xy

def union_robust(pgon1, pgon2):
    """robust union of two polygons removing small unwanted fragments"""

    # compute union using build-in function
    pgon = pgon1.union(pgon2)

    # convert to a list of polygons in case the polygons are disconnected
    if not isinstance(pgon, Polygon):
        polygons = []
        for p in list(pgon.geoms):
            if isinstance(p, Polygon):
                polygons.append(p)
    else:
        polygons = [pgon]

    # bloat the polygon by a small cube
    pgon_bloat = pgon

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
