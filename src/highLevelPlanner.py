import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.affinity import affine_transform
from shapely.affinity import translate
from shapely.ops import nearest_points, triangulate
from copy import deepcopy
from commonroad.scenario.obstacle import StaticObstacle
from commonroad.geometry.shape import ShapeGroup
from commonroad.visualization.util import collect_center_line_colors
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

def highLevelPlanner(scenario, planning_problem, param, weight_lane_change=1000, weight_velocity=1,
                     weight_safe_distance=10, minimum_safe_distance=1, minimum_steps_lane_change=5,
                     desired_steps_lane_change=20, desired_acceleration=1, desired_velocity='speed_limit',
                     compute_free_space=True, goal_set_priority=False):
    """decide on which lanelets to be at all points in time"""

    # store algorithm settings
    param['weight_lane_change'] = weight_lane_change
    param['weight_velocity'] = weight_velocity
    param['weight_safe_distance'] = weight_safe_distance
    param['minimum_safe_distance'] = minimum_safe_distance
    param['minimum_steps_lane_change'] = minimum_steps_lane_change
    param['desired_steps_lane_change'] = desired_steps_lane_change
    param['desired_acceleration'] = desired_acceleration
    param['desired_velocity'] = desired_velocity
    param['compute_free_space'] = compute_free_space
    param['goal_set_priority'] = goal_set_priority

    # extract required information from the planning problem
    param, lanelets, speed_limit, dist_init = initialization(scenario, planning_problem, param)

    # compute free space on each lanelet for each time step
    free_space, partially_occupied, safe_dist = free_space_lanelet(lanelets, scenario, speed_limit, dist_init, param)

    # compute distance to goal lanelet and number of required lane changes for each lanelet
    change_goal, dist_goal = distance2goal(lanelets, param)

    # compute the desired velocity profile over time
    vel_prof = velocity_profile(dist_goal, speed_limit, param)

    # compute the desired reference trajectory
    ref_traj = velocity2trajectory(vel_prof, param)

    # determine best sequence of lanelets to drive on
    seq = best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, dist_goal, partially_occupied, safe_dist, param)

    # refine the plan: decide on which lanelet to be on for all time steps
    plan, space, offset = refine_plan(seq, ref_traj, lanelets, safe_dist, partially_occupied, param)

    # determine time steps in which a lane change is possible
    time_lane = time_steps_lane_change(space, plan, lanelets, free_space)

    # determine free space in the global coordinate frame
    space_glob = free_space_global(space, plan, time_lane, lanelets, free_space, partially_occupied, param)

    # shrink space by intersecting with the forward reachable set
    space = reduce_space(space, plan, lanelets, offset, param)

    # extract the safe velocity intervals at each time point
    vel = [(s.bounds[1], s.bounds[3]) for s in space]

    # compute a desired reference trajectory
    ref_traj, plan = reference_trajectory(plan, free_space, space, vel_prof, time_lane, safe_dist, param, lanelets, partially_occupied)

    # resolve issue with spaces consisting of multiple distinct polygons
    space_glob = remove_multi_polyogns(space_glob, ref_traj)

    return plan, vel, space_glob, ref_traj


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

            param_['time_start'] = goal_state.time_step.start - planning_problem.initial_state.time_step
            param_['time_end'] = goal_state.time_step.end - planning_problem.initial_state.time_step

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
                            goal_space_start, goal_space_end, _, _ = projection_lanelet_centerline(
                                                                                        lanelets[param_['lane']], set)
                            param_['set'] = construct_goal_set(goal_state, goal_space_start, goal_space_end, param)
                            param['goal'].append(deepcopy(param_))
                else:
                    param_['lane'] = list(planning_problem.goal.lanelets_of_goal_position.values())[i][j]
                    goal_space_start, goal_space_end, _, _ = projection_lanelet_centerline(lanelets[param_['lane']], set)
                    param_['set'] = construct_goal_set(goal_state, goal_space_start, goal_space_end, param)
                    param['goal'].append(deepcopy(param_))

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

                param_['set'] = construct_goal_set(goal_state, goal_space_start, goal_space_end, param)
                param['goal'].append(deepcopy(param_))

    # determine number of time steps
    param['steps'] = 0

    for g in param['goal']:
        param['steps'] = np.maximum(param['steps'], g['time_end'])

    # determine distance from initial point for each lanelet
    dist_init = distance2init(lanelets, param)

    # determine the speed limit for each lanelet
    speed_limit = {}
    interpreter = TrafficSigInterpreter(scenario.scenario_id.country_name, scenario.lanelet_network)

    for id in lanelets.keys():

        speed_limit[id] = 100

        # get legal speed limit
        limit = interpreter.speed_limit(frozenset([id]))

        if limit is not None:
            speed_limit[id] = np.minimum(speed_limit[id], limit)

        # set artifical speed limit based on the dynamic constraints of the car (to not drive too fast into curves)
        limit = speed_limit_dynamics(lanelets[id], param)
        t = np.maximum((param['v_init'] - limit) / param['a_max'], 0)

        if id not in param['x0_lane'] and param['v_init'] * t - 0.5*param['a_max']*t**2 < dist_init[id]:
            speed_limit[id] = np.minimum(speed_limit[id], limit)

    return param, lanelets, speed_limit, dist_init

def construct_goal_set(goal_state, goal_space_start, goal_space_end, param):
    """construct the space for the goal set (in position-velocity space)"""

    if hasattr(goal_state, 'velocity'):
        v = goal_state.velocity
        space = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])
    else:
        space = interval2polygon([goal_space_start, param['v_min']], [goal_space_end, param['v_max']])

    return space

def speed_limit_dynamics(lanelet, param):
    """compute maximum speed for a lanelet based on kinematic constraints (Kamm's circle)"""

    # compute orientation in the middle of each time step
    traj = lanelet.center_vertices.T
    orientation = np.zeros((traj.shape[1],))

    for i in range(1, traj.shape[1]):
        diff = traj[:, i] - traj[:, i - 1]
        orientation[i] = np.arctan2(diff[1], diff[0])

    # avoid jumps by 2*pi
    for i in range(len(orientation) - 1):
        n = round(abs(orientation[i + 1] - orientation[i]) / (2 * np.pi))
        if orientation[i + 1] - orientation[i] > 3:
            orientation[i + 1] = orientation[i + 1] - n * 2 * np.pi
        elif orientation[i] - orientation[i + 1] > 3:
            orientation[i + 1] = orientation[i + 1] + n * 2 * np.pi

    # interpolate to obtain orientation at the time points
    orientation[0] = orientation[1]
    for i in range(1, len(orientation) - 1):
        orientation[i] = 0.5 * (orientation[i] + orientation[i + 1])

    # compute maximum velocity to not violate Kamm's circle constraint at 0 acceleration
    dx = np.diff(lanelet.distance)
    dphi = np.diff(orientation)

    ind = [i for i in range(len(dphi)) if abs(dphi[i]) > 1e-5]

    if len(ind) > 0:
        v = np.sqrt(np.abs(param['a_max'] * dx[ind]/dphi[ind]))
    else:
        v = 100

    return np.mean(v)

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

        if isinstance(set, Polygon) or isinstance(set.convex_hull, Polygon):
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

def distance2init(lanelets, param):
    """compute the distance from the initial point for each lanelet"""

    # initialize distance to goal lanelet
    dist = {}
    queue = []

    for i in range(len(param['x0_lane'])):
        dist[param['x0_lane'][i]] = -param['x0_set'][i]
        queue.append(param['x0_lane'][i])

    # loop until distances for all lanelets have been assigned
    while len(queue) > 0:

        q = queue.pop(0)
        lanelet = lanelets[q]

        for s in lanelet.successor:
            tmp = dist[q] + lanelet.distance[-1]
            if not s in dist.keys():
                dist[s] = tmp
                queue.append(s)
            elif tmp < dist[s]:
                dist[s] = tmp
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if not lanelet.adj_left in dist.keys():
                dist[lanelet.adj_left] = dist[q]
                queue.append(lanelet.adj_left)
            elif dist[q] < dist[lanelet.adj_left]:
                dist[lanelet.adj_left] = dist[q]
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if not lanelet.adj_right in dist.keys():
                dist[lanelet.adj_right] = dist[q]
                queue.append(lanelet.adj_right)
            elif dist[q] < dist[lanelet.adj_right]:
                dist[lanelet.adj_right] = dist[q]
                queue.append(lanelet.adj_right)

    # add costs for lanelet from which it is impossible to reach the goal set
    for l in lanelets.keys():
        if l not in dist.keys():
            dist[l] = np.inf

    return dist

def velocity_profile(dist, speed_limit, param):
    """compute the desired velocity profile over time"""

    # select desired velocity
    if param['desired_velocity'] == 'init':
        vel_des = param['v_init']
    elif param['desired_velocity'] == 'speed_limit':
        vel_des = np.inf

        for id in param['x0_lane']:
            if not speed_limit[id] is None:
                vel_des = min(vel_des, speed_limit[id])

        for goal in param['goal']:
            if goal['lane'] is not None and speed_limit[goal['lane']] is not None:
                if vel_des == np.inf:
                    vel_des = speed_limit[goal['lane']]
                else:
                    vel_des = max(vel_des, speed_limit[goal['lane']])

        if vel_des == np.inf:
            vel_des = param['v_init']
    else:
        raise Exception('Wrong value for input argument "desired velocity"!. The valid values are "init" and "speed_limit".')

    # loop over all possible combinations of initial sets and goal sets
    val = np.inf

    for i in range(len(param['x0_lane'])):
        for j in range(len(param['goal'])):

            if dist[j][param['x0_lane'][i]] < np.inf:

                # compute velocity profile
                vel_ = velocity_profile_single(dist[j], vel_des, i, param['goal'][j], param)

                # select best velocity profile (= minimum distance to desired velocity)
                val_ = abs(vel_[-1] - vel_des)
                if val_ < val:
                    vel = deepcopy(vel_)
                    val = val_

    if val == np.inf:
        raise Exception('Goal set is not reachable!')

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

            b = 6.0 * (-x_final + 0.5*t_final*(param['v_init'] + vel_final))/t_final**3
            a = (vel_final - param['v_init'] - b * t_final ** 2)/t_final

            vel = [param['v_init'] + a * t + b * t**2 for t in range(param['steps']+1)]

            return vel
        else:
            vel_min = max(vel_min, vel_min_)
            vel_max = min(vel_max, vel_max_)

    acc = max(-param['desired_acceleration'],
              min(param['desired_acceleration'], (vel_des - param['v_init'])/(param['time_step'] * param['steps'])))
    vel = list(param['v_init'] + acc * param['time_step'] * np.arange(0, param['steps'] + 1))

    # calculate velocity profile
    if vel[-1] < vel_min:
        vel = [param['v_init'] + (vel_min - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]
    elif vel[-1] > vel_max:
        vel = [param['v_init'] + (vel_max - param['v_init']) * t/param['steps'] for t in range(param['steps']+1)]

    return vel

def free_space_lanelet(lanelets, scenario, speed_limit, dist_init, param):
    """compute free space on each lanelet for all time steps"""

    obstacles = scenario.obstacles
    free_space_all = []
    tmp = [deepcopy([[] for i in range(0, param['steps']+1)]) for j in range(len(lanelets))]
    occupied_space = dict(zip(lanelets.keys(), deepcopy(tmp)))
    occupied_dist = dict(zip(lanelets.keys(), deepcopy(tmp)))
    partially_occupied = dict(zip(lanelets.keys(), deepcopy(tmp)))

    # check which lanelets are reachable
    t_final = param['steps']*param['time_step']
    dist = param['v_init'] * t_final + 0.5*param['a_max']*t_final**2
    reachable_lanes = []

    for k in lanelets.keys():
        if dist_init[k] < dist:
            reachable_lanes.append(k)
            if dist_init[k] < 0 and -dist_init[k] < param['length_max'] + param['minimum_safe_distance']:
                d = param['length_max'] + param['minimum_safe_distance'] + dist_init[k]
                reachable_lanes = reachable_lanes + relevant_predecessors(lanelets, k, d)

    # precompute lanelet properties like width, etc.
    lanelet_props = []

    for k in lanelets.keys():
        lanelet_props.append(lanelet_properties(lanelets[k]))

    lanelet_props = dict(zip(lanelets.keys(), lanelet_props))

    # loop over all obstacles
    for obs in obstacles:

        # distinguish static and dynamic obstacles
        if isinstance(obs, StaticObstacle):

            # determine lanelets that are affected by the obstacle
            pgon = get_shapely_object(obs.occupancy_at_time(0).shape)

            if obs.initial_shape_lanelet_ids is None:
                intersecting_lanes = set()
                for id in lanelets.keys():
                    if intersects_lanelet(lanelets[id], pgon):
                        intersecting_lanes.add(id)
            else:
                intersecting_lanes = obs.initial_shape_lanelet_ids

            # loop over all affected lanelets
            lanes = (x for x in intersecting_lanes if x in reachable_lanes)

            for id in lanes:

                l = lanelets[id]
                d_min = l.distance[0]
                d_max = l.distance[-1]

                # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                dist_min, dist_max, y_min, y_max = projection_lanelet_centerline(l, pgon, properties=lanelet_props[id])
                offset = 0.5 * param['length_max'] + param['minimum_safe_distance']

                # loop over all time steps
                for i in range(param['steps'] + 1):
                    space = (max(d_min, dist_min - offset), min(d_max, dist_max + offset))
                    occupied_space[id][i].append({'space': space, 'width': (y_min, y_max), 'obs': pgon})

                # check if occupied space extends to predecessor lanelet
                if dist_min - offset < 0:
                    occupied_predecessor(lanelets, id, dist_min - offset, occupied_space,
                                         range(param['steps'] + 1), (y_min, y_max), pgon)

                # check if occupied space extends to successor lanelet
                if dist_max + offset > l.distance[-1]:
                    occupied_successor(lanelets, id, dist_max + offset - l.distance[-1], occupied_space,
                                       range(param['steps'] + 1), (y_min, y_max), pgon)

        else:

            # determine lanelets that are affected by the obstacle
            if not hasattr(obs.prediction, 'shape_lanelet_assignment') or \
                    obs.prediction.shape_lanelet_assignment is None:
                intersecting_lanes = {}
                for o in obs.prediction.occupancy_set:
                    intersecting_lanes[o.time_step] = set()
                    pgon = get_shapely_object(o.shape)
                    for id in lanelets.keys():
                        if intersects_lanelet(lanelets[id], pgon):
                            intersecting_lanes[o.time_step].add(id)
            else:
                intersecting_lanes = obs.prediction.shape_lanelet_assignment

            # compute velocity of the obstacle
            v = obstacle_velocity(obs, param)

            # loop over all time steps
            for o in obs.prediction.occupancy_set:

                if o.time_step < param['steps'] + 1:

                    pgon = get_shapely_object(o.shape)

                    # loop over all affected lanelets
                    lanes = (x for x in intersecting_lanes[o.time_step] if x in reachable_lanes)

                    for id in lanes:

                        l = lanelets[id]
                        d_min = l.distance[0]
                        d_max = l.distance[-1]
                        width_min = lanelet_props[id]['width_min']
                        width_max = lanelet_props[id]['width_max']

                        # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                        dist_min, dist_max, y_min, y_max = projection_lanelet_centerline(l, pgon,
                                                                                         properties=lanelet_props[id])
                        offset = 0.5 * param['length_max'] + param['minimum_safe_distance']
                        space = (max(d_min, dist_min - offset), min(d_max, dist_max + offset))
                        occupied_space[id][o.time_step].append({'space': space, 'width': (y_min, y_max), 'obs': pgon})

                        # check if occupied space extends to predecessor lanelet
                        if dist_min - offset < 0:
                            occupied_predecessor(lanelets, id, dist_min - offset, occupied_space,
                                                 [o.time_step], (y_min, y_max), pgon)

                        # check if occupied space extends to successor lanelet
                        if dist_max + offset > l.distance[-1]:
                            occupied_successor(lanelets, id, dist_max + offset - l.distance[-1], occupied_space,
                                               [o.time_step], (y_min, y_max), pgon)

                        # compute occupied space if the safe distance is respected
                        occupied, _ = occupied_side(width_max, width_min, y_max, y_min, param)

                        if occupied == 'all':

                            offset_safe = 0.5 * param['length_max'] + np.maximum(v[o.time_step], 2)
                            space = (dist_min - offset_safe, dist_max + offset_safe)
                            occupied_dist[id][o.time_step].append(space)

                            if dist_min - offset_safe < 0:
                                for ind in l.predecessor:
                                    d = lanelets[ind].distance[-1]
                                    space = (dist_min - offset_safe + d, dist_max + offset_safe + d)
                                    occupied_dist[ind][o.time_step].append(space)

                            if dist_max + offset_safe > l.distance[-1]:
                                for ind in l.successor:
                                    d = l.distance[-1]
                                    space = (dist_min - offset_safe - d, dist_max + offset_safe - d)
                                    occupied_dist[ind][o.time_step].append(space)

    # unite occupied spaces that belong to the same time
    for id in lanelets.keys():

        # set speed-limit for the lanelet
        v_min = param['v_min']
        if speed_limit[id] is None:
            v_max = param['v_max']
        else:
            v_max = speed_limit[id]

        # get width of the lanelet
        width_min = lanelet_props[id]['width_min']
        width_max = lanelet_props[id]['width_max']

        # loop over all time steps
        for j in range(len(occupied_space[id])):

            o = occupied_space[id][j]

            if len(o) > 0:

                # assign partially occupied space
                for o_ in o:
                    occupied, w = occupied_side(width_max, width_min, o_['width'][1], o_['width'][0], param)
                    if not occupied == 'all':
                        pgon = interval2polygon([o_['space'][0], v_min - 1], [o_['space'][1], v_max + 1])
                        partially_occupied[id][j].append({'space': pgon, 'side': occupied, 'width': w, 'obs': o_['obs']})

                # unite all intersecting obstacles
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
                    occupied, _ = occupied_side(width_max, width_min, y_max, y_min, param)

                    # add obstacle to the list
                    space = (lower[cnt], up)

                    if occupied == 'all':
                        list_new.append(space)

                    cnt = ind + 1

                occupied_space[id][j] = deepcopy(list_new)

    # loop over all lanelets and compute free space
    for id in lanelets.keys():

        l = lanelets[id]
        free_space = []

        # set maximum velocity for the lanelet
        v_min = param['v_min']
        if speed_limit[id] is None:
            v_max = param['v_max']
        else:
            v_max = speed_limit[id]

        # check if it is possible to slow down to the maximum velocity in time
        if v_max < param['v_init']:
            t = (param['v_init'] - v_max) / param['a_max']
            dist_break = param['v_init'] * t - 0.5 * param['a_max'] * t ** 2 + 0.2
            if dist_break > dist_init[id]:
                v_max = param['v_init'] + 0.2

        # loop over all obstacles
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
                if status[l].value == 'red' or status[l].value == 'yellow' or status[l].value == 'red_yellow':

                    # check if car is already on intersection or cannot stop in front of lanelet anymore
                    t = param['v_init'] / param['a_max']
                    dist_break = param['v_init']*t - 0.5 * param['a_max']*t**2 + 0.5 * param['length_max']

                    if l not in param['x0_lane'] and dist_init[l] > dist_break:

                        # remove free space on lanelets influenced by the traffic light
                        free_space_all[l][i] = []

                        # remove space from predecessor lanelets to make sure the car stops before the stopline
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
                                occupied_dist[s][i].append((pgon.bounds[0]-0.5, pgon.bounds[2]))

    # compute areas in which a safe distance to the surrounding traffic participants is satisfied
    safe_dist = area_safe_distance(free_space_all, occupied_dist)

    return free_space_all, partially_occupied, safe_dist

def occupied_side(width_max, width_min, y_max, y_min, param):
    """check which side of the lanelet is occupied"""

    occupied = 'all'
    w = None

    if width_max - y_max > y_min - width_min:
        if width_max - y_max > param['width'] + 0.3:
            occupied = 'right'
            w = (width_min, y_max + 0.2)
    else:
        if y_min - width_min > param['width'] + 0.3:
            occupied = 'left'
            w = (y_min - 0.2, width_max)

    return occupied, w


def occupied_predecessor(lanelets, id, dist_min, occupied_space, steps, width, pgon):
    """compute occupied space on all predecessor lanelets"""

    for ind in lanelets[id].predecessor:

        d = lanelets[ind].distance[-1]
        space = (d + dist_min, d)

        for i in steps:
            occupied_space[ind][i].append({'space': space, 'width': width, 'obs': pgon})

        if d + dist_min < 0:
            occupied_predecessor(lanelets, ind, d + dist_min, occupied_space, steps, width, pgon)

def occupied_successor(lanelets, id, dist_max, occupied_space, steps, width, pgon):
    """compute occupied space on all successor lanelets"""

    for ind in lanelets[id].successor:

        for i in steps:
            occupied_space[ind][i].append({'space': (0, dist_max), 'width': width, 'obs': pgon})

        if dist_max > lanelets[ind].distance[-1]:
            occupied_successor(lanelets, ind, dist_max - lanelets[ind].distance[-1], occupied_space, steps, width, pgon)

def relevant_predecessors(lanelets, id, dist):
    """compute a list of relevant predecessor lanelets the cars on which could potentially influence the current lane"""

    lanes = lanelets[id].predecessor

    for s in lanelets[id].predecessor:
        if lanelets[s].distance[-1] < dist:
            lanes = lanes + relevant_predecessors(lanelets, s, dist - lanelets[s].distance[-1])

    return lanes

def lanelet_properties(lanelet):
    """precompute additional properties for the lanelet"""

    data = {}

    # construct lanelet polygon
    if lanelet.polygon.shapely_object.is_valid:
        data['polygon'] = lanelet.polygon.shapely_object
    else:
        data['polygon'] = lanelet.polygon.shapely_object.buffer(0)

    # construct polygons for the lanelet segments
    segments = []

    for i in range(len(lanelet.distance) - 1):
        segments.append(Polygon(np.concatenate((lanelet.right_vertices[[i], :], lanelet.right_vertices[[i + 1], :],
                                                lanelet.left_vertices[[i + 1], :], lanelet.left_vertices[[i], :]))))

    data['segments'] = segments

    # compute directions for curvilinear coordinate system
    directions = []
    orthogonal = []

    for i in range(len(lanelet.distance)-1):
        diff = lanelet.center_vertices[i + 1, :] - lanelet.center_vertices[i, :]
        diff = np.expand_dims(diff / np.linalg.norm(diff), axis=0)
        directions.append(diff)
        orthogonal.append(np.array([[-diff[0, 1], diff[0, 0]]]))

    data['directions'] = directions
    data['orthogonal'] = orthogonal

    # compute width of the lanelet
    width_min, width_max = width_lanelet(lanelet)

    data['width_min'] = width_min
    data['width_max'] = width_max

    return data

def projection_lanelet_centerline(lanelet, pgon, properties=None):
    """project a polygon to the center line of the lanelet to determine the occupied space"""

    # get lanelet properties
    if properties is None:
        properties = lanelet_properties(lanelet)

    # intersect polygon with lanelet
    o_int = properties['polygon'].intersection(pgon)

    if not isinstance(o_int, Polygon):
        o_int = o_int.convex_hull
        if isinstance(o_int, Polygon):
            vx, vy = o_int.exterior.coords.xy
        else:
            vx, vy = o_int.coords.xy
    else:
        vx, vy = o_int.exterior.coords.xy

    V = np.stack((vx, vy))

    # initialize minimum and maximum range of the projection
    dist_max = -np.inf
    dist_min = np.inf
    y_max = -np.inf
    y_min = np.inf

    # loop over all centerline segments
    for i in range(len(lanelet.distance)-1):

        # check if space intersects the current lanelet segment
        seg = properties['segments'][i]

        if seg.intersects(o_int):

            # project the vertices of the polygon onto the centerline
            V_ = properties['directions'][i] @ (V - np.transpose(lanelet.center_vertices[[i], :]))

            # update ranges for the projection
            dist_max = max(dist_max, max(V_[0]) + lanelet.distance[i])
            dist_min = min(dist_min, min(V_[0]) + lanelet.distance[i])

            # compute width of the lanelet that is occupied
            V_ = properties['orthogonal'][i] @ (V - np.transpose(lanelet.center_vertices[[i], :]))

            y_max = max(y_max, max(V_[0]))
            y_min = min(y_min, min(V_[0]))

    return dist_min, dist_max, y_min, y_max

def obstacle_velocity(obs, param):
    """compute the velocity of a dynamic obstacle"""

    v = np.zeros((obs.prediction.occupancy_set[-1].time_step+1, ))

    for i in range(len(obs.prediction.occupancy_set)-1):
        occ1 = obs.prediction.occupancy_set[i]
        occ2 = obs.prediction.occupancy_set[i+1]
        c1 = get_center(occ1.shape)
        c2 = get_center(occ2.shape)
        dx = np.linalg.norm(c2-c1)
        dt = (occ2.time_step-occ1.time_step)*param['time_step']
        v[occ1.time_step] = dx/dt

    if len(v) > 1:
        v[0] = v[1]
        v[-1] = v[-2]

    return v

def get_center(shape):
    """get the center of a CommonRoad shape object"""

    if hasattr(shape, 'center'):
        return shape.center
    else:
        c = []
        for s in shape.shapes:
            c.append(s.center)
        return np.mean(np.asarray(c), axis=0)

def area_safe_distance(free_space, occupied_space):
    """compute the areas in which the safe distance constraint is satisfied"""

    safe_dist = deepcopy(free_space)

    for i in free_space.keys():
        for j in range(len(free_space[i])):
            for k in range(len(free_space[i][j])):
                f = free_space[i][j][k]
                obs_right = f.bounds[2]
                obs_left = f.bounds[0]
                for o in occupied_space[i][j]:
                    if intersects_interval(o, (f.bounds[0], f.bounds[2])):
                        if 0.5*(o[0] + o[1]) < f.bounds[0]:
                            obs_left = np.maximum(obs_left, o[1])
                        else:
                            obs_right = np.minimum(obs_right, o[0])

                safe_dist[i][j][k] = {'l': f.bounds[0], 'u': f.bounds[2], 'l_safe': obs_left, 'u_safe': obs_right}

    return safe_dist

def intersects_interval(int1, int2):
    """check if two intervals intersect"""

    if int1[0] <= int2[0]:
        if int2[0] <= int1[1]:
            return True
    else:
        if int1[0] <= int2[1]:
            return True

    return False

def intersects_polygon(pgon1, pgon2):
    """check if two polygons intersect"""

    # quick check based on the interval bounds
    if not intersects_interval((pgon1.bounds[0], pgon1.bounds[2]), (pgon2.bounds[0], pgon2.bounds[2])) or \
            not intersects_interval((pgon1.bounds[1], pgon1.bounds[3]), (pgon2.bounds[1], pgon2.bounds[3])):
        return False
    else:
        try:
            return pgon1.intersects(pgon2)
        except:
            if isinstance(pgon1, Polygon) and isinstance(pgon2, Polygon):
                if pgon1.intersection(pgon2).area > 0:
                    return True
                else:
                    return False
            else:
                if isinstance(pgon1,Polygon):
                    pgon1 = [pgon1]
                else:
                    pgon1 = pgon1.geoms
                if isinstance(pgon2,Polygon):
                    pgon2 = [pgon2]
                else:
                    pgon2 = pgon2.geoms

                for p1 in pgon1:
                    for p2 in pgon2:
                        if intersects_polygon(p1, p2):
                            return True

                return False

def union_polygon(pgon1, pgon2):
    """robustly compute the union of two polygons"""

    if isinstance(pgon1, Polygon) and isinstance(pgon2, Polygon):
        return pgon1.union(pgon2)
    else:
        try:
            return pgon1.union(pgon2)
        except:
            if isinstance(pgon1, Polygon):
                pgon1 = [pgon1]
            else:
                pgon1 = pgon1.geoms
            if isinstance(pgon2, Polygon):
                pgon2 = [pgon2]
            else:
                pgon2 = pgon2.geoms

            pgon = pgon1[0]

            for i in range(1, len(pgon1)):
                pgon = pgon.union(pgon1[i])

            for i in range(len(pgon2)):
                pgon = pgon.union(pgon2[i])

        return pgon


def safe_distance_violation(space, safe_distance):
    """check how much the given set violates the safe distance constraint"""

    # determine the gap the car is currently in
    l = space.bounds[0]
    u = space.bounds[2]

    for entry in safe_distance:
        if entry['l'] <= l <= entry['u'] or entry['l'] <= u <= entry['u']:
            l_safe = entry['l_safe']
            u_safe = entry['u_safe']
            l_ = entry['l']
            u_ = entry['u']
            break

    # compute violation
    if l_safe < u_safe:             # possible to satisfy the safe distance constraint

        if intersects_interval((l_safe, u_safe), (l, u)):
            val = 0
        else:
            val = min([abs(l-l_safe), abs(l-u_safe), abs(u-l_safe), abs(u-u_safe)])

    else:                           # not possible to satisfy the safe distance constraint

        if l_safe < l_ and u_safe > u_:
            m = 0.5*(l_safe + u_safe)
            dist = abs(l_safe - m)
        elif l_safe < l_:
            m = l_
            dist = abs(l_safe - m)
        else:
            m = u_
            dist = abs(u_safe - m)

        if l <= m <= u:
            val = dist
        else:
            val = dist + np.minimum(abs(u - m), abs(l - m))

    return val

def width_lanelet(lanelet):
    """compute the width of a lanelet"""

    width_max = np.inf
    width_min = -np.inf

    # loop over all centerline segments
    for i in range(0, len(lanelet.distance) - 1):

        # compute lanelet width
        d = lanelet.right_vertices[[i], :] - lanelet.center_vertices[[i], :]
        width_min = max(width_min, -np.linalg.norm(d))

        d = lanelet.left_vertices[[i], :] - lanelet.center_vertices[[i], :]
        width_max = min(width_max, np.linalg.norm(d))

    return width_min, width_max

def interval2polygon(lb, ub):
    """convert an interval given by lower and upper bound to a polygon"""

    return Polygon([(lb[0], lb[1]), (lb[0], ub[1]), (ub[0], ub[1]), (ub[0], lb[1])])


def reduce_space(space, plan, lanelets, offset, param):
    """reduce the space of the drivable area by intersecting with the forward reachable set"""

    # loop over all time steps
    for i in range(len(space)-1):

        # compute forward reachable set
        space_ = reach_set_forward(space[i], param)

        # shift set if moving on to a successor lanelet
        if not plan[i+1] == plan[i] and plan[i+1] != lanelets[plan[i]].adj_left and plan[i+1] != lanelets[plan[i]].adj_right:
            dist = offset[plan[i+1]] - offset[plan[i]]
            space_ = translate(space_, -dist, 0)

        # intersect forward reachable set with drivable area
        space[i+1] = space[i+1].intersection(space_)

    return space

def lanelet_orientation(lanelet, x0, reference_orientation):
    """compute the orientation of the lanelet at the given point"""

    # get lanelet segment that corresponds to the point
    for i in range(len(lanelet.distance)):
        if i == len(lanelet.distance)-1:
            diff = lanelet.center_vertices[-1, :] - lanelet.center_vertices[-2, :]
        elif x0 > lanelet.distance[i]:
            diff = lanelet.center_vertices[i+1, :] - lanelet.center_vertices[i, :]

    # compute orientation
    o = np.arctan2(diff[1], diff[0])

    # consider 2*pi periodicallity for the orientation
    tmp = o - reference_orientation
    o = reference_orientation + np.mod(tmp + np.pi, 2 * np.pi) - np.pi

    return o

def order_initial_sets(lanelets, dist_goal, param):
    """compute the best order for exploring the potential initial sets"""

    if len(param['x0_lane']) == 1:
        return [0]

    # sort possible initial lanelets according to a heursitic based on distance to goal set and initial orientation
    dist = []

    for i in range(len(param['x0_lane'])):
        tmp = np.inf
        for j in range(len(dist_goal)):
            if param['goal'][j]['lane'] == param['x0_lane'][i]:
                tmp = -10
            else:
                tmp = min(tmp, dist_goal[j][param['x0_lane'][i]])
            o = lanelet_orientation(lanelets[param['x0_lane'][i]], param['x0_set'][i], param['orientation'])
        dist.append(tmp + 10 * abs(o - param['orientation']))

    order = np.argsort(dist)

    return order

def best_lanelet_sequence(lanelets, free_space, ref_traj, change_goal, dist_goal, partially_occupied, safe_dist, param):
    """determine the best sequences of lanelets to drive on that reach the goal state"""

    min_cost = None
    is_priority = False

    # sort possible initial lanelets
    order = order_initial_sets(lanelets, dist_goal, param)

    # loop over all initial nodes
    final_node = None

    for j in order:

        # create initial node
        v = param['v_init']
        space = interval2polygon([param['x0_set'][j] - 0.01, v - 0.01], [param['x0_set'][j] + 0.01, v + 0.01])
        x0 = {'step': 0, 'space': space, 'lanelet': param['x0_lane'][j]}
        queue = [Node([x0], [], [], 'none', ref_traj, change_goal, lanelets, safe_dist, param)]

        # loop until queue empty -> all possible lanelet sequences have been explored
        while len(queue) > 0:

            # sort the queue
            queue.sort(key=lambda i: i.cost)

            # remove nodes with costs higher than the current minimum cost for reaching the goal set
            if min_cost is not None and (not param['goal_set_priority'] or is_priority):
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

            drive_area, left, right, suc, x0 = compute_drivable_area(lanelet, lanelets, node.x0, free_space, prev,
                                                                     node.lane_prev, partially_occupied, param)

            # check if goal set has been reached
            for i in range(len(param['goal'])):

                goal = param['goal'][i]

                if goal['lane'] is None or lanelet.lanelet_id == goal['lane']:

                    final_sets = []

                    for d in drive_area:
                        if goal['time_start'] <= d['step'] <= goal['time_end'] and d['space'].intersects(goal['set']):
                            space_fin = d['space'].intersection(goal['set'])
                            if space_fin.area > 0:
                                final_sets.append({'space': space_fin, 'step': d['step']})

                    if len(final_sets) > 0:
                        node_temp = expand_node(node, final_sets, drive_area, 'final',
                                                ref_traj, change_goal, lanelets, safe_dist, param)
                        if min_cost is None or node_temp.cost < min_cost or \
                                (param['goal_set_priority'] and i == 0 and not is_priority):
                            min_cost = node_temp.cost
                            final_node = deepcopy(node_temp)
                            if i == 0:
                                is_priority = True

            # create child nodes
            for entry in x0:
                queue.append(expand_node(node, entry, drive_area, node.lane_prev,
                                         ref_traj, change_goal, lanelets, safe_dist, param))

            for entry in left:
                if len(entry) > param['minimum_steps_lane_change']:
                    queue.append(expand_node(node, entry, drive_area, 'right',
                                             ref_traj, change_goal, lanelets, safe_dist, param))

            for entry in right:
                if len(entry) > param['minimum_steps_lane_change']:
                    queue.append(expand_node(node, entry, drive_area, 'left',
                                             ref_traj, change_goal, lanelets, safe_dist, param))

            for entry in suc:
                queue.append(expand_node(node, entry, drive_area, 'none',
                                         ref_traj, change_goal, lanelets, safe_dist, param))

        # abort if a solution has been found
        if not final_node is None:
            return final_node

    raise Exception("Failed to find a feasible solution!")

def compute_drivable_area(lanelet, lanelets, x0, free_space, prev, lane_prev, partially_occupied, param):
    """compute the drivable area for a single lanelet"""

    # initialization
    cnt = 0
    cnt_prev = 0

    for i in range(len(prev)):
        if prev[i]['step'] <= x0[cnt]['step']:
            cnt_prev = cnt_prev + 1

    drive_area = [x0[cnt]]
    successors = []
    left = []
    right = []
    x0_new = []
    space_suc = dict(zip(lanelet.successor, [None for i in range(len(lanelet.successor))]))

    # loop over all time steps up to the final time
    for i in range(x0[cnt]['step'], param['steps']):

        # compute reachable set using maximum acceleration and deceleration
        space = reach_set_forward(drive_area[len(drive_area)-1]['space'], param)

        # unite with set resulting from doing a lane-change from the predecessor lanelet
        if cnt+1 < len(x0)-1 and x0[cnt+1]['step'] == i+1:
            space = union_polygon(space, x0[cnt+1]['space'])
            cnt = cnt + 1

        # avoid intersection with lanelet for moving on to a successor lanelet
        if space.bounds[2] >= lanelet.distance[-1] and len(free_space[lanelet.lanelet_id][i+1]) > 0 and \
                free_space[lanelet.lanelet_id][i+1][-1].bounds[2] >= lanelet.distance[-1] - 1e-5:
            for k in space_suc.keys():
                if len(free_space[k][i+1]) > 0 and free_space[k][i+1][0].bounds[0] < 1e-5:
                    if space_suc[k] is None:
                        space_suc[k] = translate(space, -lanelet.distance[-1], 0)
                    else:
                        space_suc[k] = reach_set_forward(space_suc[k], param)
                else:
                    space_suc[k] = None
        else:
            for k in space_suc.keys():
                space_suc[k] = None

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
        for suc in lanelet.successor:
            if space_suc[suc] is not None:
                if space_suc[suc].intersects(free_space[suc][i+1][0]):
                    free = union_free_space(translate(space_lanelet[-1], -lanelet.distance[-1], 0), free_space[suc][i+1][0])
                    if free.intersects(space_suc[suc]):
                        space_suc[suc] = free.intersection(space_suc[suc])
                        successors = add_transition(successors, space_suc[suc], i + 1, suc, param)
                else:
                    space_suc[suc] = None

        if finished:
            break

        # check if it is possible to make a lane change to the left
        if not lanelet.adj_left is None and lanelet.adj_left_same_direction and \
                abs(lanelet.distance[-1] - lanelets[lanelet.adj_left].distance[-1]) < param['minimum_safe_distance']:

            for space_left in free_space[lanelet.adj_left][i+1]:
                if space.intersects(space_left) and not(lane_prev == 'left' and cnt_prev < len(prev) and
                                  prev[cnt_prev]['step'] == i+1 and space_left.intersects(prev[cnt_prev]['space'])):

                    space_ = space.intersection(space_left)
                    space_ = intersection_partially_occupied(space_, partially_occupied, i + 1, lanelet, 'left')

                    if not space_.is_empty:
                        left = add_transition(left, space_, i+1, lanelet.adj_left, param)

        # check if it is possible to make a lane change to the right
        if not lanelet.adj_right is None and lanelet.adj_right_same_direction and \
                abs(lanelet.distance[-1] - lanelets[lanelet.adj_right].distance[-1]) < param['minimum_safe_distance']:

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

    A = [1, dt, 0, 1, 0, 0]
    B = LineString([[-0.5 * dt ** 2 * a_max, -dt * a_max], [0.5 * dt ** 2 * a_max, dt * a_max]])

    space_new = affine_transform(space, A)
    space_new = minkowski_sum_polygon_line(space_new, B)

    """space1 = affine_transform(space, [1, dt, 0, 1, 0.5 * dt ** 2 * a_max, dt * a_max])
    space2 = affine_transform(space, [1, dt, 0, 1, -0.5 * dt ** 2 * a_max, -dt * a_max])

    space_new = space1.union(space2)
    space_new = space_new.convex_hull"""

    return space_new

def reach_set_backward(space, param):
    """compute the backward reachable set for one time step using maximum acceleration and deceleration"""

    dt = param['time_step']
    a_max = param['a_max']

    A = [1, -dt, 0, 1, 0, 0]
    B = LineString([[-0.5 * dt ** 2 * a_max, dt * a_max], [0.5 * dt ** 2 * a_max, -dt * a_max]])

    space_new = affine_transform(space, A)
    space_new = minkowski_sum_polygon_line(space_new, B)

    """space1 = affine_transform(space, [1, -dt, 0, 1, -0.5 * dt ** 2 * a_max, dt * a_max])
    space2 = affine_transform(space, [1, -dt, 0, 1, 0.5 * dt ** 2 * a_max, -dt * a_max])

    space_new = space1.union(space2)
    space_new = space_new.convex_hull"""

    return space_new

def minkowski_sum_polygon_line(pgon, line):
    """compute the Minkowski sum between a polygon and a line"""

    # get center and difference vector for the line
    m_line = 0.5 * np.array([line.coords.xy[0][0] + line.coords.xy[0][1], line.coords.xy[1][0] + line.coords.xy[1][1]])
    d_line = 0.5 * np.array([line.coords.xy[0][1] - line.coords.xy[0][0], line.coords.xy[1][1] - line.coords.xy[1][0]])

    # shift polygon by line center
    pgon = translate(pgon, m_line[0], m_line[1])

    # convert to a list of polygons in case the polygons are disconnected
    if not isinstance(pgon, Polygon):
        polygons = []
        for p in list(pgon.geoms):
            if isinstance(p, Polygon):
                polygons.append(p)
    else:
        polygons = [pgon]

    # loop over all polygons
    res = []

    for pgon in polygons:

        # get vertices
        V = np.concatenate((np.expand_dims(pgon.exterior.xy[0], axis=0),
                            np.expand_dims(pgon.exterior.xy[1], axis=0)), axis=0)

        # split into lower and upper boundary
        tmp = np.array([[-d_line[1], d_line[0]]]) @ V
        ind1 = np.minimum(np.argmin(tmp), np.argmax(tmp))
        ind2 = np.maximum(np.argmin(tmp), np.argmax(tmp))

        V1 = V[:, ind1:ind2+1]
        V2 = np.concatenate((V[:, ind2:], V[:, 0:ind1+1]), axis=1)

        # shift the two parts by +/- the line vector
        pgon_shift = translate(pgon, d_line[0], d_line[1])

        V1_ = V1 + np.expand_dims(d_line, axis=1) @ np.ones((1, V1.shape[1]))
        V2_ = V2 - np.expand_dims(d_line, axis=1) @ np.ones((1, V2.shape[1]))
        V = np.concatenate((V1_, V2_), axis=1)
        pgon1 = Polygon(list(V.T))

        V1_ = V1 - np.expand_dims(d_line, axis=1) @ np.ones((1, V1.shape[1]))
        V2_ = V2 + np.expand_dims(d_line, axis=1) @ np.ones((1, V2.shape[1]))
        V = np.concatenate((V1_, V2_), axis=1)
        pgon2 = Polygon(list(V.T))

        # select the correct part to construct the resulting polygon
        if not pgon1.is_valid and not pgon2.is_valid:
            tri = triangulate(pgon)
            pgon_res = minkowski_sum_polygon_line(tri[0], line)
            for i in range(len(tri)):
                pgon_res = pgon_res.union(minkowski_sum_polygon_line(tri[i], line))
        else:
            if not pgon1.is_valid:
                pgon_res = pgon2
            elif not pgon2.is_valid:
                pgon_res = pgon1
            else:
                if pgon1.intersection(pgon_shift).area > pgon2.intersection(pgon_shift).area:
                    pgon_res = pgon1
                else:
                    pgon_res = pgon2

        # unit with parallel polygons if possible
        if len(res) > 0:
            found = False
            for i in range(len(res)):
                if res[i].intersects(pgon_res):
                    res[i] = res[i].union(pgon_res)
                    found = True
                    break
            if not found:
                res.append(pgon_res)
        else:
            res.append(pgon_res)

    # construct the resuling set (potentially a multi-polygon)
    if len(res) == 1:
        res = res[0]
    else:
        res = MultiPolygon(res)

    if not res.is_valid:
        return res.buffer(0)
    else:
        return res

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
                intersects_polygon(space_new, last_entry['space']):
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
        if i < len(vel_prof)-1:
            x = x + vel_prof[i] * param['time_step'] + 0.5*(vel_prof[i+1]-vel_prof[i])*param['time_step']
        else:
            x = x + vel_prof[i] * param['time_step']

    return ref_traj

def refine_plan(seq, ref_traj, lanelets, safe_dist, partially_occupied, param):
    """refine the plan by deciding on which lanelets to be on at which points in time"""

    # determine shift in position when changing to a successor lanelet
    offset = offsets_lanelet_sequence(seq.lanelets, lanelets)

    # select best final set from all intersections with the goal set
    min_cost = np.inf

    for fin in seq.x0:

        cost_vel = cost_reference_trajectory(ref_traj, fin, offset[-1])
        cost_safe_dist = safe_distance_violation(fin['space'], safe_dist[seq.lanelets[-1]][fin['step']])
        cost = param['weight_velocity'] * cost_vel + param['weight_safe_distance'] * cost_safe_dist

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

        # make sure that the first set at least enters the loop once
        first_set = False

        if i == len(seq.lanelets) - 1 and time_step - 1 == seq.drive_area[i][0]['step'] - 1:
            time_step = time_step + 1
            space_first = space[-1]
            first_set = True

        # make sure that also lanelets with a single time step enter the loop
        if not first_set and time_step - 1 == seq.drive_area[i][0]['step'] - 1:
            time_step = time_step + 1
            space_first = space[time_step-1]
            first_set = True

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

            # check if space is empty
            if len(space) > j+1 and space[j+1].area == 0:
                break

            # propagate set one time step backward in time
            if first_set:
                space[j] = space_first
            else:
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
                        if space_.intersects(seq.drive_area[i - 1][cnt]['space']):
                            space_ = space_.intersection(seq.drive_area[i - 1][cnt]['space'])
                            if not isinstance(space_, Polygon):
                                space_ = space_.convex_hull
                            transitions.append({'space': space_, 'step': j})
                else:
                    if space[j].intersects(seq.drive_area[i - 1][cnt]['space']):
                        space_ = space[j].intersection(seq.drive_area[i - 1][cnt]['space'])
                        if not isinstance(space_, Polygon):
                            space_ = space_.convex_hull
                        if lanelets[seq.lanelets[i-1]].adj_left == seq.lanelets[i]:
                            space_ = intersection_partially_occupied(space_, partially_occupied, j,
                                                                     lanelets[seq.lanelets[i-1]], 'left')
                        elif lanelets[seq.lanelets[i-1]].adj_right == seq.lanelets[i]:
                            space_ = intersection_partially_occupied(space_, partially_occupied, j,
                                                                     lanelets[seq.lanelets[i - 1]], 'right')
                        if isinstance(space_, Polygon):
                            if not space_.is_empty:
                                transitions.append({'space': space_, 'step': j})
                        else:
                            for p in space_.geoms:
                                transitions.append({'space': p, 'step': j})
                cnt = cnt - 1

            # catch special case where time steps do not overlap
            if i > 0 and is_successor and j == seq.drive_area[i][0]['step'] and seq.drive_area[i-1][-1]['step'] == j-1:
                space_ = translate(reach_set_backward(space_prev, param), dist, 0)
                space_ = space_.intersection(seq.drive_area[i-1][-1]['space'])
                transitions.append({'space': space_, 'step': j-1})

            # catch special case where one stays on the same lanelet
            if i > 0 and seq.lanelets[i] == seq.lanelets[i-1] and j == seq.drive_area[i][0]['step'] and \
                    seq.drive_area[i-1][-1]['step'] == j and not space[j].intersects(seq.drive_area[i-1][-1]['space']):
                space_ = reach_set_backward(space[j], param)
                if space_.intersects(seq.drive_area[i-1][-2]['space']):
                    space_ = space_.intersection(seq.drive_area[i - 1][-2]['space'])
                    transitions.append({'space': space_, 'step': j - 1})

            # catch special case where time steps where a lane change is possible do not overlap
            if i > 0 and is_successor and j == seq.drive_area[i][0]['step'] and len(transitions) == 0:
                space_ = translate(reach_set_backward(space_prev, param), dist, 0)
                space_ = space_.intersection(seq.drive_area[i - 1][cnt]['space'])
                transitions.append({'space': space_, 'step': seq.drive_area[i - 1][cnt]['step']})

        # select the best transition to take to the previous lanelet
        if i > 0:

            if len(transitions) == 1 or \
                    (not is_successor and time_step - transitions[-1]['step'] < param['minimum_steps_lane_change']):
                time_step = transitions[-1]['step']
                space_ = transitions[-1]['space']
            else:

                min_cost = np.inf

                for t in transitions:
                    if is_successor or time_step - t['step'] >= param['minimum_steps_lane_change']:

                        cost_vel = cost_reference_trajectory(ref_traj, t, offset[i-1])
                        cost_safe_dist1 = safe_distance_violation(t['space'], safe_dist[seq.lanelets[i-1]][t['step']])
                        if is_successor:
                            t_space = translate(t['space'], -dist, 0)
                        else:
                            t_space = t['space']
                        if t_space.bounds[0] > 0:
                            cost_safe_dist2 = safe_distance_violation(t_space, safe_dist[seq.lanelets[i]][t['step']])
                        else:
                            cost_safe_dist2 = 0
                        cost = param['weight_velocity'] * cost_vel + \
                               param['weight_safe_distance'] * np.maximum(cost_safe_dist1, cost_safe_dist2)

                        if cost < min_cost:
                            time_step = t['step']
                            space_ = t['space']
                            min_cost = cost

            space[time_step] = space_
            plan[time_step] = seq.lanelets[i-1]

    # potentially remove the previously added auxiliary set
    if len(space) > len(plan):
        space = space[:-1]

    # compute space offset for lane changes to successor lanelets
    offset = {seq.lanelets[0]: 0}

    for i in range(1, len(seq.lanelets)):
        if lanelets[seq.lanelets[i]] in lanelets[seq.lanelets[i-1]].successor or seq.lanelets[i] == seq.lanelets[i-1]:
            offset[seq.lanelets[i]] = offset[seq.lanelets[i-1]]
        else:
            offset[seq.lanelets[i]] = offset[seq.lanelets[i-1]] + lanelets[seq.lanelets[i-1]].distance[-1]

    return plan, space, offset

def free_space_global(space, plan, time, lanelets, free_space, partially_occupied, param):
    """compute the drivable space in the global coordinate frame"""

    # check if the free space around the reference trajectory should be computed
    if not param['compute_free_space']:
        return None

    # determine indices for all lane changes
    plan = np.asarray(plan)
    ind = np.where(plan[:-1] != plan[1:])[0]

    # get types for all lane changes
    types = []

    for i in range(len(ind)):
        l = lanelets[plan[ind[i]]]
        if plan[ind[i]+1] in l.successor:
            types.append('successor')
        elif plan[ind[i]+1] == l.adj_right:
            types.append('right')
        else:
            types.append('left')

    if len(ind) == 0:
        time = [[]]
        ind = [len(plan)]

    # loop over all time steps
    space_glob = deepcopy(space)
    cnt = 0

    for i in range(len(plan)):

        l = lanelets[plan[i]]
        space_int = (space[i].bounds[0], space[i].bounds[2])
        partial = []

        # get free space on the current lanelet
        for f in free_space[plan[i]][i]:
            if intersects_interval((f.bounds[0], f.bounds[2]), space_int):
                lower = f.bounds[0] - 0.5*param['length_max']
                upper = f.bounds[2] + 0.5*param['length_max']
                pgon = lanelet2global(l, lower, upper)

                for p in partially_occupied[plan[i]][i]:
                    partial = partial + [p['obs']]
                break

        space_glob[i] = Polygon(pgon.T)

        if not space_glob[i].is_valid:
            space_glob[i] = space_glob[i].buffer(0)

        # get free space on the right lanelet
        pgon_right = None
        width = 0

        if not l.adj_right is None and l.adj_right_same_direction and \
                (i <= 10 or (i in time[cnt] and types[cnt] == 'right')):
            for f in free_space[l.adj_right][i]:
                if intersects_interval((f.bounds[0], f.bounds[2]), space_int):
                    tmp = np.minimum(f.bounds[2], space_int[1]) - np.maximum(f.bounds[0], space_int[0])
                    if tmp > width:
                        pgon_right = deepcopy(f)

            if pgon_right is not None:
                lower_right = pgon_right.bounds[0] - 0.5 * param['length_max']
                upper_right = pgon_right.bounds[2] + 0.5 * param['length_max']
                pgon_right = lanelet2global(lanelets[l.adj_right], lower_right, upper_right)
                pgon_right = unite_lanelets_right(pgon, lower, upper, pgon_right, lower_right, upper_right, l)
                space_glob[i] = space_glob[i].union(pgon_right)

                for p in partially_occupied[l.adj_right][i]:
                    partial = partial + [p['obs']]

        # get free space on the left lanelet
        pgon_left = None
        width = 0

        if not l.adj_left is None and l.adj_left_same_direction and \
                (i <= 10 or (i in time[cnt] and types[cnt] == 'left')):
            for f in free_space[l.adj_left][i]:
                if intersects_interval((f.bounds[0], f.bounds[2]), space_int):
                    tmp = np.minimum(f.bounds[2], space_int[1]) - np.maximum(f.bounds[0], space_int[0])
                    if tmp > width:
                        pgon_left = deepcopy(f)

            if pgon_left is not None:
                lower_left = pgon_left.bounds[0] - 0.5 * param['length_max']
                upper_left = pgon_left.bounds[2] + 0.5 * param['length_max']
                pgon_left = lanelet2global(lanelets[l.adj_left], lower_left, upper_left)
                pgon_left = unite_lanelets_right(pgon_left, lower_left, upper_left, pgon, lower, upper, l)
                space_glob[i] = space_glob[i].union(pgon_left)

                for p in partially_occupied[l.adj_left][i]:
                    partial = partial + [p['obs']]

        # get free space on the successor lanelets
        pgon_suc = None
        width = np.maximum(0, upper - l.distance[-1])

        if width > 0:
            for s in l.successor:
                lower_suc = 0
                upper_suc = width
                if len(free_space[s][i]) > 0 and free_space[s][i][0].bounds[0] < 1e-5:
                    upper_suc = np.maximum(free_space[s][i][0].bounds[2] + 0.5 * param['length_max'], upper_suc)
                pgon_tmp = lanelet2global(lanelets[s], lower_suc, upper_suc,
                                          suc=width > lanelets[s].distance[-1], lanelets=lanelets)
                if pgon_suc is None:
                    pgon_suc = Polygon(pgon_tmp.T)
                else:
                    pgon_suc = pgon_suc.union(Polygon(pgon_tmp.T))

                for p in partially_occupied[s][i]:
                    partial = partial + [p['obs']]

            if pgon_suc is not None:
                pgon_suc = unite_lanelets_successor(pgon, pgon_suc)
                space_glob[i] = space_glob[i].union(pgon_suc)

        # get free space on the predecessor lanelets
        pgon_pre = None

        if lower < 0:
            for p in l.predecessor:
                lower_pre = lower + lanelets[p].distance[-1]
                upper_pre = lanelets[p].distance[-1]
                if len(free_space[p][i]) > 0 and free_space[p][i][-1].bounds[2] > lanelets[p].distance[-1]-1e-5:
                    lower_pre = np.minimum(free_space[p][i][-1].bounds[0] - 0.5 * param['length_max'], lower_pre)
                pgon_tmp = lanelet2global(lanelets[p], lower_pre, upper_pre,
                                          pred=lower + lanelets[p].distance[-1] < 0, lanelets=lanelets)
                if pgon_pre is None:
                    pgon_pre = Polygon(pgon_tmp.T)
                else:
                    pgon_pre = pgon_pre.union(Polygon(pgon_tmp.T))

                for pr in partially_occupied[p][i]:
                    partial = partial + [pr['obs']]

            if pgon_pre is not None:
                pgon_pre = unite_lanelets_predecessor(pgon, pgon_pre)
                space_glob[i] = space_glob[i].union(pgon_pre)

        # if initial state is on multiple lanelets, add the space on these lanelets to the free space
        if plan[i] == plan[0]:
            for j in range(len(param['x0_lane'])):
                id = param['x0_lane'][j]
                if id != plan[0]:
                    for f in free_space[id][i]:
                        if f.bounds[0] <= param['x0_set'][j] <= f.bounds[2]:
                            lower_x0 = f.bounds[0] - 0.5 * param['length_max']
                            upper_x0 = f.bounds[2] + 0.5 * param['length_max']
                            pgon = lanelet2global(lanelets[id], lower_x0, upper_x0)
                            pgon = Polygon(pgon.T)
                            if not pgon.is_valid:
                                pgon = pgon.buffer(0)
                            space_glob[i] = space_glob[i].union(pgon)

                            for p in partially_occupied[id][i]:
                                partial = partial + [p['obs']]

        # remove partially occupied space from the free space
        for p in partial:
            if intersects_polygon(p, space_glob[i]):
                space_glob[i] = space_glob[i].difference(p)

        # increase counter
        if len(time[cnt]) > 0:
            if i == time[cnt][-1] and cnt < len(time)-1:
                cnt = cnt + 1
        else:
            if i == ind[cnt] and cnt < len(time)-1:
                cnt = cnt + 1

    return space_glob

def unite_lanelets_successor(pgon, pgon_suc):
    """unite a lanelet with its successor lanelet"""

    n = int(pgon.shape[1] / 2)

    # get vertices of the successor polygon in the correct order
    vx, vy = pgon_suc.exterior.coords.xy
    V = np.stack((vx, vy))

    if pgon_suc.exterior.is_ccw:
        V = np.fliplr(V)

    V = remove_duplicate_columns(V)

    if all(V[:, 0] - V[:, -1] < 1e-5):
        V = V[:, :-1]

    # combine with the polygon for the current lanelet
    ind1 = np.argmin(np.sum((pgon[:, [n - 1]] - V)**2, axis=0))
    V = np.concatenate((V[:, ind1:], V[:, :ind1]), axis=1)
    ind2 = np.argmin(np.sum((pgon[:, [n]] - V) ** 2, axis=0))

    p = np.concatenate((pgon[:, 0:n], V[:, :ind2], pgon[:, n:]), axis=1)

    # construct polygon object
    pgon = Polygon(p.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon

def unite_lanelets_predecessor(pgon, pgon_pre):
    """unite a lanelet with its predecessor lanelet"""

    n = int(pgon.shape[1] / 2)

    # get vertices of the predecessor polygon in the correct order
    vx, vy = pgon_pre.exterior.coords.xy
    V = np.stack((vx, vy))

    if pgon_pre.exterior.is_ccw:
        V = np.fliplr(V)

    V = remove_duplicate_columns(V)

    if all(V[:, 0] - V[:, -1] < 1e-5):
        V = V[:, :-1]

    # combine with the polygon for the current lanelet
    ind1 = np.argmin(np.sum((pgon[:, [-1]] - V)**2, axis=0))
    V = np.concatenate((V[:, ind1:], V[:, :ind1]), axis=1)
    ind2 = np.argmin(np.sum((pgon[:, [0]] - V) ** 2, axis=0))

    p = np.concatenate((V[:, :ind2+1], pgon[:, 1:-1]), axis=1)

    # construct polygon object
    pgon = Polygon(p.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon

def unite_lanelets_right(pgon, lower, upper, pgon_right, lower_right, upper_right, lanelet):
    """unite a lanelet with its right neighbor"""

    n = int(pgon.shape[1] / 2)
    n_right = int(pgon_right.shape[1] / 2)

    # handle left side of the resulting polygon
    if upper >= lanelet.distance[-1] and upper_right >= lanelet.distance[-1]:
        p = pgon[:, 0:n+1]
        index = n_right
    elif upper > upper_right:
        point, ind = closest_line_segment(pgon[:, n:], pgon_right[:, [n_right-1]])
        p = np.concatenate((pgon[:, :ind + n + 1], point), axis=1)
        index = n_right-1
    else:
        point, ind = closest_line_segment(pgon_right[:, 0:n_right], pgon[:, [n]])
        p = np.concatenate((pgon[:, 0:n+1], point), axis=1)
        index = ind+1

    # concatenate with the right side of the resulting polygon
    if lower <= 0 and lower_right <= 0:
        p = np.concatenate((p, pgon_right[:, index:]), axis=1)
    elif lower < lower_right:
        p = np.concatenate((p, pgon_right[:, index:]), axis=1)
        point, ind = closest_line_segment(pgon[:, n:], pgon_right[:, [0]])
        p = np.concatenate((p, point, pgon[:, ind+n+1:]), axis=1)
    else:
        p = np.concatenate((p, pgon_right[:, index:]), axis=1)
        point, ind = closest_line_segment(pgon_right[:, 0:n_right], pgon[:, [-1]])
        p = np.concatenate((p, pgon_right[:, 0:ind+1], point), axis=1)

    p = remove_duplicate_columns(p)

    # construct polygon object
    pgon = Polygon(p.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon

def remove_duplicate_columns(M):
    """remove all duplicate neighbouring columns from a matrix"""

    ind = [0]
    cnt = 0

    for i in range(1, M.shape[1]):
        if any(abs(M[:, cnt] - M[:, i]) > 1e-5):
            cnt = cnt + 1
            ind.append(i)

    return M[:, ind]

def closest_line_segment(line, point):
    """compute the line segment that is closest to the current point"""

    dist = np.inf

    # loop over all line segments
    for i in range(line.shape[1]-1):
        dist_, p_ = distance_line_point(line[:, i:i+2], point)
        if dist_ < dist:
            dist = dist_
            ind = i
            p = p_

    if dist == np.inf:
        p = line[:, [0]]
        ind = 0

    return p, ind

def distance_line_point(line, point):
    """compute the minimum distance between a line and a point"""

    # project point onto line
    diff = line[:, 0] - line[:, 1]

    if np.linalg.norm(np.expand_dims(diff, axis=1)) > 1e-5:

        c = np.array([[diff[1]], [-diff[0]]])
        d = c.T @ line[:, [0]]

        A = np.eye(2) - c @ c.T
        b = d * c

        p_proj = A @ point + b

        dist_proj = np.linalg.norm(point - p_proj)
    else:
        dist_proj = np.inf

    # compute distances
    dist1 = np.linalg.norm(point - line[:, [0]])
    dist2 = np.linalg.norm(point - line[:, [1]])

    # return point with minimum distance
    if dist1 < dist2:
        if dist1 < dist_proj:
            return dist1, line[:, [0]]
        else:
            return dist_proj, p_proj
    else:
        if dist2 < dist_proj:
            return dist2, line[:, [1]]
        else:
            return dist_proj, p_proj

def time_steps_lane_change(space, plan, lanelets, free_space):
    """compute the time steps in which a lane change is possible"""

    # determine indices for all lane changes
    plan = np.asarray(plan)
    ind = np.where(plan[:-1] != plan[1:])[0]

    time = [[] for i in range(len(ind))]

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
                            time[i].append(j)
                else:
                    break

            else:
                intersection = False

                for f in free_space[lanelet_2.lanelet_id][j]:
                    if f.intersects(space[j]):
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
                            time[i].append(j)
                else:
                    break

            else:
                intersection = False

                for f in free_space[lanelet_1.lanelet_id][j]:
                    if f.intersects(space[j]):
                        intersection = True
                        time[i].append(j)

                if not intersection:
                    break

        # sort time steps where lane change is possible
        time[i] = np.unique(np.asarray(time[i]))
        time[i] = np.sort(time[i])

    return time

def remove_multi_polyogns(space, ref_traj):
    """select the best connected space for the case that space consists of multiple disconnected polygons"""

    # catch case where space has not been computed
    if space is None:
        return space

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

def reference_trajectory(plan, free_space, space, vel_prof, time_lane, safe_dist, param, lanelets, partially_occupied):
    """compute a desired reference trajectory"""

    # compute suitable velocity profile
    x, v = trajectory_position_velocity(deepcopy(space), plan, vel_prof, lanelets, safe_dist, param)
    x, v = improve_trajectory_position_velocity(deepcopy(space), plan, x, v, lanelets, safe_dist, param)

    # update plan (= lanelet-time-assignment)
    plan = np.asarray(plan)
    lanes = [plan[0]]

    for i in range(1, len(plan)):
        if plan[i] != plan[i-1]:
            lanes.append(plan[i])

    dist = 0

    ind = np.where(plan[:-1] != plan[1:])[0]
    ind = [-1] + ind.tolist() + [len(plan) - 1]

    for i in range(len(lanes)-1):
        if lanes[i+1] in lanelets[lanes[i]].successor:
            for j in range(ind[i]+1, ind[i+2]+1):
                if x[j] - dist <= lanelets[lanes[i]].distance[-1]:
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
    orthogonal = [deepcopy(tmp) for i in range(len(lanes))]
    center_orientation = [deepcopy(tmp) for i in range(len(lanes))]
    shifts = []
    nonempty = []
    dist_plan = np.zeros((len(plan),))

    for j in range(len(lanes)):

        # loop over all time steps
        for i in range(step, len(x)):

            d = x[i] - dist
            lanelet = lanelets[lanes[j]]
            partial = partially_occupied[lanes[j]][i]

            if d >= lanelet.distance[-1] - 1e-5:
                if j < len(lanes) - 1 and lanes[j + 1] in lanelet.successor and plan[i] != lanes[j]:
                    dist = dist + lanelet.distance[-1]
                    step = i
                    break

            dist_plan[i] = dist

            for k in range(1, len(lanelet.distance)):
                if d <= lanelet.distance[k] + 1e-10:
                    if len(nonempty) == 0 or nonempty[-1] != j:
                        nonempty.append(j)
                    p1 = lanelet.center_vertices[k-1, :]
                    p2 = lanelet.center_vertices[k, :]
                    p = p1 + (p2 - p1) * (d - lanelet.distance[k-1])/(lanelet.distance[k] - lanelet.distance[k-1])
                    center_traj[j][i] = np.transpose(p)
                    diff = p2 - p1
                    orthogonal[j][i] = np.array([[-diff[1], diff[0]]])/np.linalg.norm(diff)
                    center_orientation[j][i] = np.arctan2(diff[1], diff[0])
                    for par in partial:
                        if par['space'].bounds[0] <= d <= par['space'].bounds[2]:
                            add_shift(shifts, par, lanelet, i, j, k)
                    break

    # shift center trajectory for partially occupied space
    center_traj = correction_partially_occupied(center_traj, orthogonal, lanes, plan, lanelets, shifts)

    # correct the center trajectory to avoid collisions with lanelet boundaries for lanelets with high curvature
    """for j in range(len(center_traj)):
        center_traj[j] = correct_centerline(center_traj[j], lanelets[lanes[j]], param)"""

    # store reference trajectory (without considering lane changes)
    ref_traj = np.zeros((2, len(plan)))
    orientation = np.zeros((len(plan), ))
    cnt = 0

    for i in range(0, len(plan)):
        if len(center_traj[nonempty[cnt]][i]) == 0:
            ref_traj[:, i] = center_traj[nonempty[cnt+1]][i]
            orientation[i] = center_orientation[nonempty[cnt+1]][i]
        else:
            ref_traj[:, i] = center_traj[nonempty[cnt]][i]
            orientation[i] = center_orientation[nonempty[cnt]][i]
        if i < len(plan)-1 and plan[i] != plan[i+1]:
            cnt = cnt + 1

    # loop over all lane changes
    lane_changes = []

    for i in range(len(ind)):

        # check if it is a lane change or just a change onto a successor lanelet
        if (lanelets[plan[ind[i]]].adj_right is not None and lanelets[plan[ind[i]]].adj_right == plan[ind[i]+1]) or \
                (lanelets[plan[ind[i]]].adj_left is not None and lanelets[plan[ind[i]]].adj_left == plan[ind[i] + 1]):

            # compute start and end time step for the lane change
            ind_start_ = max(time_lane[i][0]+1, ind[i] - np.floor(param['desired_steps_lane_change']/2)).astype(int)
            ind_end_ = min(time_lane[i][-1], ind[i] + np.floor(param['desired_steps_lane_change']/2)).astype(int)

            ind_start = ind[i]
            ind_end = ind[i]

            for j in range(ind[i], ind_start_-1, -1):
                found1 = False
                found2 = False
                for f in free_space[plan[ind[i]]][j]:
                    if f.bounds[0] <= x[j] - dist_plan[j] <= f.bounds[2]:
                        found1 = True
                        break
                for f in free_space[plan[ind[i]+1]][j]:
                    if f.bounds[0] <= x[j] - dist_plan[j] <= f.bounds[2]:
                        found2 = True
                        break
                if found1 and found2 and len(center_traj[i][j]) > 0 and len(center_traj[i+1][j]) > 0:
                    ind_start = j
                else:
                    break

            for j in range(ind[i], ind_end_+1):
                found1 = False
                found2 = False
                for f in free_space[plan[ind[i]]][j]:
                    if f.bounds[0] <= x[j] - dist_plan[j] <= f.bounds[2]:
                        found1 = True
                        break
                for f in free_space[plan[ind[i]+1]][j]:
                    if f.bounds[0] <= x[j] - dist_plan[j] <= f.bounds[2]:
                        found2 = True
                        break
                if found1 and found2 and len(center_traj[i][j]) > 0 and len(center_traj[i+1][j]) > 0:
                    ind_end = j
                else:
                    break

            lane_changes.append({'ind': np.arange(ind_start, ind_end+1), 'lanes': [plan[ind[i]], plan[ind[i]+1]]})

            # interpolate between the center trajectories for the two lanes
            if ind_end - ind_start > 0:
                for j in range(ind_start, ind_end+1):

                    w = 1/(1 + np.exp(-5*(2*((j - ind_start)/(ind_end - ind_start))-1)))
                    p = (1-w) * center_traj[i][j] + w * center_traj[i+1][j]
                    ref_traj[:, j] = p

    # extend reference trajectory by velocity and orientation
    velocity = velocity_reference_trajectory(ref_traj, param)
    orientation = orientation_reference_trajectory(ref_traj, orientation, param)

    ref_traj = np.concatenate((ref_traj, np.expand_dims(velocity, axis=0), np.expand_dims(orientation, axis=0)), axis=0)

    # correct orientation of the reference trajectory to avoid collisions
    ref_traj = correct_orientation(ref_traj, lanelets, plan, lanes, lane_changes, param)

    return ref_traj, plan

def add_shift(shifts, par, lanelet, i, j, k):
    """add a shift to the referenece trajectory in order to avoid intersections with partially occupied space"""

    if par['side'] == 'left':
        w_lane = np.linalg.norm(lanelet.right_vertices[k, :] - lanelet.center_vertices[k, :])
        w = 0.5*(par['width'][0] - w_lane)
        if len(shifts) == 0 or not shifts[-1]['lane'] == j or \
                not shifts[-1]['ind'][-1] in [i, i-1] or not shifts[-1]['side'] == 'left':
            shifts.append({'lane': j, 'w': w, 'side': 'left', 'ind': [i], 'end': True, 'start': True})
        else:
            shifts[-1]['w'] = min(shifts[-1]['w'], w)
            shifts[-1]['ind'].append(i)
    else:
        w_lane = np.linalg.norm(lanelet.left_vertices[k, :] - lanelet.center_vertices[k, :])
        w = 0.5 * (par['width'][1] + w_lane)
        if len(shifts) == 0 or not shifts[-1]['lane'] == j or \
                not shifts[-1]['ind'][-1] in [i, i-1] or not shifts[-1]['side'] == 'right':
            shifts.append({'lane': j, 'w': w, 'side': 'right', 'ind': [i], 'end': True, 'start': True})
        else:
            shifts[-1]['w'] = max(shifts[-1]['w'], w)
            shifts[-1]['ind'].append(i)

def correction_partially_occupied(center_traj, orthogonal, lanes, plan, lanelets, shifts):
    """correct the reference trajectory at areas where the lane is partially occupied"""

    # number of time steps for interpolation
    interp = 10

    # combine connected correction shifts
    for i in range(len(shifts)):
        if i + 1 < len(shifts) and shifts[i]['ind'][-1]+1 == shifts[i+1]['ind'][0] and \
                lanes[shifts[i]['lane']] in lanelets[lanes[shifts[i+1]['lane']]].predecessor:
            shifts[i]['end'] = False
            shifts[i+1]['start'] = False
            shifts[i]['interp_end'] = 0
            shifts[i+1]['interp_start'] = 0
        elif i + 1 < len(shifts) and lanes[shifts[i]['lane']] in lanelets[lanes[shifts[i+1]['lane']]].predecessor:
            val = int((shifts[i + 1]['ind'][0] - shifts[i]['ind'][-1])/2)
            shifts[i]['interp_end'] = val
            shifts[i + 1]['interp_start'] = val

    # loop over all shifts
    for s in shifts:

        # get number of time steps for interpolation before and after the shift
        interp_start = interp
        interp_end = interp

        if 'interp_start' in s.keys():
            interp_start = min(interp, s['interp_start'])

        if 'interp_end' in s.keys():
            interp_end = min(interp, s['interp_end'])

        # extend shifted area if it is close to ths start or end
        s['ind'] = np.unique(s['ind'])
        index = [i for i in range(len(plan)) if len(center_traj[s['lane']][i]) > 0]

        if s['ind'][0] - interp_start < index[0] and (s['lane'] == 0 or
                                                lanes[s['lane']-1] not in lanelets[lanes[s['lane']]].predecessor):
            s['ind'] = list(np.arange(index[0], s['ind'][-1] + 1))
        if s['ind'][-1] + interp_end > index[-1] and (s['lane'] == len(lanes)-1 or
                                                  lanes[s['lane']+1] not in lanelets[lanes[s['lane']]].successor):
            s['ind'] = list(np.arange(s['ind'][0], index[-1] + 1))

        # shift reference trajectory
        for i in s['ind']:
            d = orthogonal[s['lane']][i]
            center_traj[s['lane']][i] = center_traj[s['lane']][i] + d * s['w']

        # interpolation before the shifted area
        if s['start']:

            # find minimum index for interpolation before the shifted area
            min_index = index[0]
            cnt = s['lane']
            while cnt > 0 and lanes[cnt - 1] in lanelets[lanes[cnt]].predecessor:
                cnt = cnt - 1
                ind = [i for i in range(len(plan)) if len(center_traj[cnt][i]) > 0]
                if len(ind) > 0:
                    min_index = ind[0]
            min_index = max(min_index, s['ind'][0] - interp_start)

            # interpolation before the shifted area
            if s['ind'][0] - min_index > 0:
                cnt = s['lane']
                for i in range(s['ind'][0]-1, min_index-1, -1):
                    while len(center_traj[cnt][i]) == 0 and cnt > 0:
                        cnt = cnt - 1
                    d = orthogonal[cnt][i]
                    p = center_traj[cnt][i] + d * s['w']
                    w = 1 / (1 + np.exp(-5 * (2 * ((i - min_index) / (s['ind'][0] - min_index)) - 1)))
                    center_traj[cnt][i] = w * p + (1-w) * center_traj[cnt][i]

        # interpolation after the shifted area
        if s['end']:

            # find maximum index for interpolation after the shifted area
            max_index = index[-1]
            cnt = s['lane']
            while cnt + 1 < len(lanes) and lanes[cnt+1] in lanelets[lanes[cnt]].successor:
                cnt = cnt + 1
                ind = [i for i in range(len(plan)) if len(center_traj[cnt][i]) > 0]
                if len(ind) > 0:
                    max_index = ind[-1]
            max_index = min(max_index, s['ind'][-1] + interp_end)

            # interpolation after the shifted area
            if max_index - s['ind'][-1] - 1 > 0:
                cnt = s['lane']
                for i in range(s['ind'][-1] + 1, max_index+1):
                    while len(center_traj[cnt][i]) == 0 and cnt < len(center_traj)-1:
                        cnt = cnt + 1
                    d = orthogonal[cnt][i]
                    p = center_traj[cnt][i] + d * s['w']
                    w = 1 / (1 + np.exp(-5 * (2 * ((i - s['ind'][-1] - 1) / (max_index - s['ind'][-1] - 1)) - 1)))
                    center_traj[cnt][i] = (1 - w) * p + w * center_traj[cnt][i]

    return center_traj

def correct_orientation(ref_traj, lanelets, plan, lanes, lane_changes, param):
    """correct the orientation of the reference trajectory to avoid collisions with the lanelet boundaries"""

    # construct polygons for all lanes
    pgons_lanes = {}
    left = lanelets[lanes[0]].left_vertices.T
    right = lanelets[lanes[0]].right_vertices.T
    ind = [lanes[0]]

    for i in range(1, len(lanes)):
        left_ = lanelets[lanes[i]].left_vertices.T
        right_ = lanelets[lanes[i]].right_vertices.T
        if lanes[i] in lanelets[lanes[i-1]].successor:
            left = np.concatenate((left, left_), axis=1)
            right = np.concatenate((right, right_), axis=1)
            ind.append(lanes[i])
        else:
            pgon = Polygon(np.concatenate((left, np.fliplr(right)), axis=1).T)
            pgon = add_predecessor(pgon, ind, lanelets, 0, param)
            pgon = add_successor(pgon, ind, lanelets, 0, param)
            for j in ind:
                pgons_lanes[j] = pgon
            ind = [lanes[i]]
            left = left_
            right = right_

    pgon = Polygon(np.concatenate((left, np.fliplr(right)), axis=1).T)
    pgon = add_predecessor(pgon, ind, lanelets, 0, param)
    pgon = add_successor(pgon, ind, lanelets, 0, param)
    for j in ind:
        pgons_lanes[j] = pgon

    # construct polygons for all lane changes
    pgons = []

    for l in lane_changes:
        if lanelets[l['lanes'][0]].adj_right == l['lanes'][1]:
            pgon = Polygon(np.concatenate((lanelets[l['lanes'][0]].left_vertices.T,
                                           np.fliplr(lanelets[l['lanes'][1]].right_vertices.T)), axis=1).T)
        else:
            pgon = Polygon(np.concatenate((lanelets[l['lanes'][1]].left_vertices.T,
                                           np.fliplr(lanelets[l['lanes'][0]].right_vertices.T)), axis=1).T)
        pgon = add_predecessor(pgon, l['lanes'], lanelets, 0, param)
        pgon = add_successor(pgon, l['lanes'], lanelets, 0, param)
        pgons.append(pgon)

    # polygon for the car
    car = interval2polygon([-param['length']/2, -param['width']/2], [param['length']/2, param['width']/2])

    # loop over all time steps
    for i in range(ref_traj.shape[1]):

        phi = ref_traj[3, i]
        x = ref_traj[0, i]
        y = ref_traj[1, i]
        pgon = pgons_lanes[plan[i]]

        for j in range(len(lane_changes)):
            if i in lane_changes[j]['ind']:
                pgon = pgons[j]
                break

        if not pgon.is_valid:
            pgon = pgon.buffer(0)

        dphi = 0.1
        cnt = 0
        phi_ = phi
        first = True

        while not pgon.contains(affine_transform(car, [np.cos(phi_), -np.sin(phi_), np.sin(phi_), np.cos(phi_), x, y])):

            if abs(phi_ - phi) > np.pi:
                dphi = 0.5*dphi
                cnt = 1
                if dphi < 1e-3:
                    if first:
                        ref_traj[0:2, i] = correct_position(pgon, car, ref_traj[:, i])
                        x = ref_traj[0, i]
                        y = ref_traj[1, i]
                        first = False
                    else:
                        raise Exception('Reference trajectory generation failed!')

            phi_ = phi + cnt * dphi

            if cnt > 0:
                cnt = -cnt
            else:
                cnt = -cnt + 1

        ref_traj[3, i] = phi_

    return ref_traj

def add_successor(pgon, ind, lanelets, length, param):
    """unite lanelet sequence by the successor lanelets"""

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    for s in lanelets[ind[-1]].successor:
        left = np.concatenate((lanelets[ind[-1]].left_vertices.T, lanelets[s].left_vertices.T), axis=1)
        right = np.concatenate((lanelets[ind[-1]].right_vertices.T, lanelets[s].right_vertices.T), axis=1)
        tmp = Polygon(np.concatenate((left, np.fliplr(right)), axis=1).T)
        if not tmp.is_valid:
            tmp = tmp.buffer(0)
        pgon = pgon.union(tmp)
        if length + lanelets[s].distance[-1] < param['length_max']/2:
            pgon = add_successor(pgon, [s], lanelets, length + lanelets[s].distance[-1], param)

    return pgon

def add_predecessor(pgon, ind, lanelets, length, param):
    """unite lanelet sequence by the predecessor lanelets"""

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    for p in lanelets[ind[0]].predecessor:
        left = np.concatenate((lanelets[p].left_vertices.T, lanelets[ind[0]].left_vertices.T), axis=1)
        right = np.concatenate((lanelets[p].right_vertices.T, lanelets[ind[0]].right_vertices.T), axis=1)
        tmp = Polygon(np.concatenate((left, np.fliplr(right)), axis=1).T)
        if not tmp.is_valid:
            tmp = tmp.buffer(0)
        pgon = pgon.union(tmp)
        if length + lanelets[p].distance[-1] < param['length_max']/2:
            pgon = add_predecessor(pgon, [p], lanelets, length + lanelets[p].distance[-1], param)

    return pgon

def orientation_reference_trajectory(ref_traj, default, param):
    """compute orientation for the reference trajectory"""

    # compute orientation in the middle of each time step
    orientation = np.zeros((ref_traj.shape[1], ))
    orientation[0] = param['orientation']

    for i in range(1, ref_traj.shape[1]):
        diff = ref_traj[:, i] - ref_traj[:, i - 1]
        if np.linalg.norm(diff)/param['time_step'] < 0.01:
            orientation[i] = default[i]
        else:
            orientation[i] = np.arctan2(diff[1], diff[0])

    # avoid jumps by 2*pi
    for i in range(len(orientation) - 1):
        n = round(abs(orientation[i + 1] - orientation[i]) / (2 * np.pi))
        if orientation[i + 1] - orientation[i] > 3:
            orientation[i + 1] = orientation[i + 1] - n * 2 * np.pi
        elif orientation[i] - orientation[i + 1] > 3:
            orientation[i + 1] = orientation[i + 1] + n * 2 * np.pi

    # interpolate to obtain orientation at the time points
    for i in range(1, len(orientation)-1):
        orientation[i] = 0.5*(orientation[i] + orientation[i+1])

    return orientation

def velocity_reference_trajectory(ref_traj, param):
    """comptue the velocity for the reference trajectory"""

    # compute velocity
    velocity = np.zeros((ref_traj.shape[1], ))
    velocity[0] = param['v_init']

    for i in range(1, ref_traj.shape[1]):
        if i < ref_traj.shape[1]-1:
            v1 = np.linalg.norm(ref_traj[:, i + 1] - ref_traj[:, i]) / param['time_step']
            v2 = np.linalg.norm(ref_traj[:, i] - ref_traj[:, i-1]) / param['time_step']
            velocity[i] = 0.5*(v1 + v2)
        else:
            velocity[i] = np.linalg.norm(ref_traj[:, i] - ref_traj[:, i-1]) / param['time_step']

    return velocity

def correct_centerline(traj, lanelet, param):
    """correct the centerline (since middle is often not the best line for collision avoidance)"""

    # find start and end index
    index = [i for i in range(len(traj)) if len(traj[i]) > 0]

    if len(index) == 1:
        return traj

    # compute orientation
    orientation = np.zeros((len(index, )))

    for i in range(len(index)):
        if i == 0:
            diff = traj[index[i + 1]] - traj[index[i]]
            orientation[i] = np.arctan2(diff[1], diff[0])
        elif i == len(index)-1:
            diff = traj[index[i]] - traj[index[i-1]]
            orientation[i] = np.arctan2(diff[1], diff[0])
        else:
            diff1 = traj[index[i + 1]] - traj[index[i]]
            diff2 = traj[index[i]] - traj[index[i-1]]
            orientation[i] = 0.5*(np.arctan2(diff1[1], diff1[0]) + np.arctan2(diff2[1], diff2[0]))

    # transform center line
    car = interval2polygon([-param['length']/2, -100], [param['length']/2, 100])
    left_bound = LineString(lanelet.left_vertices)
    right_bound = LineString(lanelet.right_vertices)

    for i in range(len(index)):
        phi = orientation[i]
        pgon = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), traj[index[i]][0], traj[index[i]][1]])
        int_left = translate(pgon.intersection(left_bound), -traj[index[i]][0], -traj[index[i]][1])
        int_right = translate(pgon.intersection(right_bound), -traj[index[i]][0], -traj[index[i]][1])
        ub = min(-np.sin(phi) * np.asarray(int_left.coords.xy[0]) + np.cos(phi) * np.asarray(int_left.coords.xy[1]))
        lb = max(-np.sin(phi) * np.asarray(int_right.coords.xy[0]) + np.cos(phi) * np.asarray(int_right.coords.xy[1]))
        traj[index[i]] = traj[index[i]] + np.array([-np.sin(phi), np.cos(phi)]) * 0.5*(lb + ub)

    return traj

def correct_position(pgon, car, x):
    """correct the position of the reference trajectory so that the car does not intersect the lanelet boundaries"""

    phi = x[3]
    pgon_car = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), x[0], x[1]])

    pgon_road = interval2polygon((pgon.bounds[0]-100, pgon.bounds[1]-100), (pgon.bounds[2]+100, pgon.bounds[3]+100))
    pgon_road = pgon_road.difference(pgon)

    pgon_int = pgon_car.intersection(pgon_road)
    pgon_int = affine_transform(pgon_int, [np.cos(phi), np.sin(phi), -np.sin(phi), np.cos(phi), 0, 0])

    diff = pgon_int.bounds[3] - pgon_int.bounds[1]
    x1 = x[0:2] + np.array([np.sin(phi), -np.cos(phi)]) * (diff + 0.01)
    x2 = x[0:2] - np.array([np.sin(phi), -np.cos(phi)]) * (diff + 0.01)

    pgon1 = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), x1[0], x1[1]])
    pgon2 = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), x2[0], x2[1]])

    if pgon1.intersection(pgon).area > pgon2.intersection(pgon).area:
        return x1
    else:
        return x2

def trajectory_position_velocity(space, plan, vel_prof, lanelets, safe_dist, param):
    """compute the desired space-velocity trajectory"""

    # initialization
    dt = param['time_step']
    a_max = param['a_max']

    space_orig = deepcopy(space)

    v = np.asarray(vel_prof[:len(plan)])
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
            elif distance_border(space[i+1], Point(p1[0], p1[1])) < 1e-10:
                p = Point(p1[0], p1[1])
            elif distance_border(space[i+1], Point(p2[0], p2[1])) < 1e-10:
                p = Point(p2[0], p2[1])
            else:
                p = nearest_points(space[i + 1], LineString([p1, p2]))[0]
                if distance_border(space[i+1], p) > 1e-10:
                    raise Exception("Space not driveable!")

            x[i+1] = p.x
            v[i+1] = p.y

    return x, v

def improve_trajectory_position_velocity(space, plan, x, v, lanelets, safe_dist, param):
    """compute the desired space-velocity trajectory"""

    # initialization
    dt = param['time_step']
    a_max = param['a_max']
    space_orig = deepcopy(space)
    dist = 0
    corrections = []

    # loop over all time steps
    for i in range(len(plan) - 1):

        # shift space if moving on to a successor lanelet
        if plan[i + 1] != plan[i] and plan[i + 1] in lanelets[plan[i]].successor:
            dist = dist + lanelets[plan[i]].distance[-1]

        space[i + 1] = translate(space[i + 1], dist, 0)

        # check if point satisfies safe distance to other cars and correct it if not
        for s in safe_dist[plan[i + 1]][i + 1]:
            if s['l'] <= space_orig[i + 1].bounds[0] and s['u'] >= space_orig[i + 1].bounds[2]:
                if s['l_safe'] < s['u_safe']:
                    if s['l_safe'] + dist <= x[i + 1] <= s['u_safe'] + dist:
                        x_des = x[i + 1]
                    elif x[i + 1] > s['u_safe'] + dist:
                        x_des = s['u_safe'] + dist
                        type = 'upper'
                    else:
                        x_des = s['l_safe'] + dist
                        type = 'lower'
                else:
                    x_des = 0.5 * (s['l_safe'] + s['u_safe']) + dist
                    type = 'both'
                break

        if x[i + 1] != x_des:
            factor = param['weight_velocity'] / (param['weight_velocity'] + param['weight_safe_distance'])
            x[i + 1] = x_des + factor * (x[i + 1] - x_des)
            if type == 'upper':
                corrections.append({'index': i+1, 'upper': x[i+1], 'lower': space[i+1].bounds[0]})
            elif type == 'lower':
                corrections.append({'index': i + 1, 'upper': space[i + 1].bounds[2], 'lower': x[i + 1]})
            else:
                corrections.append({'index': i+1, 'upper': x[i+1], 'lower': x[i+1]})

    # update velocity profile
    if len(corrections) > 0:

        x_prev = deepcopy(x)

        # detemermine best linear velocity profile
        a_lower = -param['a_max']
        a_upper = param['a_max']

        for c in corrections:
            t = c['index'] * dt
            a_upper = min(a_upper, 2 * (c['upper'] - x[0] - v[0] * t) / t ** 2)
            a_lower = max(a_lower, 2 * (c['lower'] - x[0] - v[0] * t) / t ** 2)

        t = np.arange(len(v)) * dt

        if a_upper > a_lower:

            # select velocity profile closest to the original one
            v_upper = v[0] + a_upper * t
            v_lower = v[0] + a_lower * t

            if np.mean(abs(v_lower - v)) > np.mean(abs(v_upper - v)):
                v = v_upper
                a = a_upper
            else:
                v = v_lower
                a = a_lower

        else:
            a = 0.5 * (a_lower + a_upper)
            v = v[0] + 0.5 * a * t

        for i in range(1, len(x)):
            x[i] = x[0] + v[0] * (i * dt) + 0.5 * a * (i * dt) ** 2

        if corrections[-1]['index'] < len(x):
            ind = corrections[-1]['index']
            t = (len(x) - ind) * dt
            a = 2 * (x_prev[-1] - x[ind] - v[ind] * t) / t ** 2
            a = max(-param['a_max'], min(a, param['a_max']))
            for i in range(ind + 1, len(x)):
                t = (i - ind) * dt
                v[i] = v[ind] + a * t
                x[i] = x[ind] + v[ind] * t + 0.5 * a * t ** 2

        # check if driving the desired velocity profile is feasible
        if not space[-1].contains(Point(x[-1], v[-1])):
            p = nearest_points(space[-1], Point(x[-1], v[-1]))[0]
            x[-1] = p.x
            v[-1] = p.y

        for i in range(len(space)-1, 0, -1):

            # determine the best feasible acceleration
            p1 = (x[i] - v[i] * dt - 0.5 * a_max * dt ** 2, v[i] + a_max * dt)
            p2 = (x[i] - v[i] * dt + 0.5 * a_max * dt ** 2, v[i] - a_max * dt)
            pgon = space[i-1].intersection(LineString([p1, p2]))

            if not pgon.is_empty:
                p = nearest_points(pgon, Point(x[i-1], v[i-1]))[0]
            elif distance_border(space[i - 1], Point(p1[0], p1[1])) < 1e-10:
                p = Point(p1[0], p1[1])
            elif distance_border(space[i - 1], Point(p2[0], p2[1])) < 1e-10:
                p = Point(p2[0], p2[1])
            else:
                p = nearest_points(space[i - 1], LineString([p1, p2]))[0]
                if distance_border(space[i - 1], p) > 1e-10:
                    raise Exception("Space not driveable!")

            x[i - 1] = p.x
            v[i - 1] = p.y

    return x, v

def distance_border(space, point):
    """compute the distance of a point from the polytope boundary"""

    if isinstance(space, Polygon):
        return space.exterior.distance(point)
    else:
        dist = np.inf
        for p in space.geoms:
            dist = min(dist, p.exterior.distance(point))
        return dist

def cost_reference_trajectory(ref_traj, area, offset):
    """compute cost based on the distance between the reachable set and the desired reference trajectory"""

    p = Point(ref_traj[area['step']][0] - offset, ref_traj[area['step']][1])

    if area['space'].contains(p):
        dist = 0
    else:
        dist = area['space'].exterior.distance(p)

    return dist

def offsets_lanelet_sequence(seq, lanelets):
    """determine shift in position when changing to a successor lanelet for the given lanelet sequence"""

    offset = [0]

    for i in range(len(seq) - 1):
        if seq[i + 1] in lanelets[seq[i]].successor:
            offset.append(offset[i] + lanelets[seq[i]].distance[-1])
        else:
            offset.append(offset[i])

    return offset

def lanelet2global(lanelet, lower, upper, width=None, pred=False, suc=False, lanelets=None):
    """transform free space from lanelet coordinate system to global coordinate system"""

    left_vertices = []
    right_vertices = []

    # catch the case where space exceeds over the lanelet bounds
    lower_ = lower
    upper_ = upper

    if lower < lanelet.distance[0]:
        lower = 0

    if upper > lanelet.distance[-1]:
        upper = lanelet.distance[-1]

    # loop over the single segments of the lanelet
    for j in range(0, len(lanelet.distance) - 1):

        if lower >= lanelet.distance[j] and lower <= lanelet.distance[j + 1]:
            frac = (lower - lanelet.distance[j]) / (lanelet.distance[j + 1] - lanelet.distance[j])

            d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
            p_left = lanelet.left_vertices[j] + d * frac
            left_vertices.append(p_left)

            d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
            p_right = lanelet.right_vertices[j] + d * frac
            right_vertices.append(p_right)

        if lower <= lanelet.distance[j] <= upper:
            p_left = lanelet.left_vertices[j]
            left_vertices.append(p_left)

            p_right = lanelet.right_vertices[j]
            right_vertices.append(p_right)

        if upper >= lanelet.distance[j] and upper <= lanelet.distance[j + 1]:
            frac = (upper - lanelet.distance[j]) / (lanelet.distance[j + 1] - lanelet.distance[j])

            d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
            p_left = lanelet.left_vertices[j] + d * frac
            left_vertices.append(p_left)

            d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
            p_right = lanelet.right_vertices[j] + d * frac
            right_vertices.append(p_right)

            break

    # restrict to the specified width
    if width is not None:
        for i in range(len(left_vertices)):
            center = 0.5 * (left_vertices[i] + right_vertices[i])
            d = left_vertices[i] - right_vertices[i]
            d = d / np.linalg.norm(d)
            left_vertices[i] = center + width[0] * d
            right_vertices[i] = center + width[1] * d

    # construct the resulting polygon in the global coordinate system
    right_vertices.reverse()
    left_vertices.extend(right_vertices)
    pgon = np.asarray(left_vertices).T

    # consider successor lanelets
    if suc and upper_ > lanelet.distance[-1]:
        pgon_suc = None
        for s in lanelet.successor:
            pgon_tmp = lanelet2global(lanelets[s], 0, upper_ - lanelet.distance[-1])
            if pgon_suc is None:
                pgon_suc = Polygon(pgon_tmp.T)
            else:
                pgon_suc = pgon_suc.union(Polygon(pgon_tmp.T))
        if pgon_suc is not None:
            pgon = unite_lanelets_successor(pgon, pgon_suc)
            vx, vy = pgon.exterior.xy
            pgon = np.stack((vx, vy))

    # consider predessesor lanelets
    if pred and lower_ < 0:
        pgon_pre = None
        for p in lanelet.predecessor:
            pgon_tmp = lanelet2global(lanelets[p], lanelets[p].distance[-1] + lower_, lanelets[p].distance[-1])
            if pgon_pre is None:
                pgon_pre = Polygon(pgon_tmp.T)
            else:
                pgon_pre = pgon_pre.union(Polygon(pgon_tmp.T))
        if pgon_pre is not None:
            pgon = unite_lanelets_predecessor(pgon, pgon_pre)
            vx, vy = pgon.exterior.xy
            pgon = np.stack((vx, vy))

    return pgon

def union_free_space(pgon, pgon_suc):
    """unite the free space on the current and on the successor lanelet"""

    if pgon.bounds[3] == pgon_suc.bounds[3]:
        res = Polygon([[pgon.bounds[0], pgon.bounds[1]],
                       [pgon.bounds[0], pgon.bounds[3]],
                       [pgon_suc.bounds[2], pgon.bounds[3]],
                       [pgon_suc.bounds[2], pgon.bounds[1]]])
    else:
        res = Polygon([[pgon.bounds[0], pgon.bounds[1]],
                       [pgon.bounds[0], pgon.bounds[3]],
                       [pgon.bounds[2], pgon.bounds[3]],
                       [pgon_suc.bounds[0], pgon_suc.bounds[3]],
                       [pgon_suc.bounds[2], pgon_suc.bounds[3]],
                       [pgon_suc.bounds[2], pgon_suc.bounds[1]]])

    return res

def expand_node(node, x0, drive_area, lane_prev, ref_traj, change_goal, lanelets, safe_dist, param):
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
    return Node(x0, l, s, lane_prev, ref_traj, change_goal, lanelets, safe_dist, param)


class Node:
    """class representing a single node for A*-search"""

    def __init__(self, x0, lanes, drive_area, lane_prev, ref_traj, change_goal, lanelets, safe_dist, param):
        """class constructor"""

        # store object properties
        self.x0 = x0
        self.lanelets = lanes
        self.drive_area = drive_area
        self.lane_prev = lane_prev
        self.cost = self.cost_function(ref_traj, change_goal, lanelets, safe_dist, param)

    def cost_function(self, ref_traj, change_goal, lanelets, safe_dist, param):
        """compute cost function value for the node"""

        # determine shift in position when changing to a successor lanelet
        offset = offsets_lanelet_sequence(self.lanelets, lanelets)

        # determine cost from deviation to the desired reference trajectory
        diff = np.inf * np.ones(len(ref_traj))

        if self.lane_prev != 'final':
            for i in range(len(self.x0)):
                diff[self.x0[i]['step']] = 0

        for i in range(len(ref_traj)):

            # loop over all lanelets
            for j in range(len(self.drive_area)):

                diff_cur = np.inf
                p = Point(ref_traj[i][0] - offset[j], ref_traj[i][1])

                # loop over all time steps
                for set in self.drive_area[j]:

                    # compute distance from desired reference trajectory
                    if set['step'] == i:
                        if set['space'].contains(p):
                            diff_cur = 0
                        elif isinstance(set['space'], Polygon):
                            diff_cur = set['space'].exterior.distance(p)
                        else:
                            diff_cur = np.inf
                            for pgon in set['space'].geoms:
                                diff_cur = np.minimum(diff_cur, pgon.exterior.distance(p))
                    elif set['step'] > i:
                        break

                # total distance -> minimum over the distance for all lanelets
                if diff_cur < diff[i]:
                    diff[i] = diff_cur

                if diff[i] == 0:
                    break

        diff = np.extract(diff < np.inf, diff)

        # determine cost from violation of the safe distance
        viol = np.inf * np.ones(len(ref_traj))

        if self.lane_prev != 'final':
            for i in range(len(self.x0)):
                viol[self.x0[i]['step']] = 0

        for i in range(len(ref_traj)):

            # loop over all lanelets
            for j in range(len(self.drive_area)):

                viol_cur = np.inf
                l = self.lanelets[j]

                # loop over all time steps
                for set in self.drive_area[j]:

                    # compute safe distance violation for current set
                    if set['step'] == i:
                        viol_cur = safe_distance_violation(set['space'], safe_dist[l][i])
                    elif set['step'] > i:
                        break

                # total violation -> minimum over the violations for all lanelets
                if viol_cur < viol[i]:
                    viol[i] = viol_cur

                if viol[i] == 0:
                    break

        viol = np.extract(viol < np.inf, diff)

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

        return param['weight_lane_change'] * (lane_changes + expect_changes) + \
                            param['weight_velocity'] * np.sum(diff) + param['weight_safe_distance'] * np.sum(viol)
