import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import LineString
from commonroad.geometry.shape import Rectangle
from shapely.ops import nearest_points
from copy import deepcopy
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

from auxiliary.overlappingLanelets import overlappingLanelets

def prediction(scenario, horizon, x_ego, overlapping_lanelets=None, most_likely=True):
    """predict the future positions of the surrounding vehicles"""

    trajectories = []

    # determine lanelets that overlap
    if overlapping_lanelets is None:
        overlapping_lanelets = overlappingLanelets(scenario)

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    ids = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(ids, scenario.lanelet_network.lanelets))

    # extract the initial positions for all vehicles
    vehicles = []

    for obs in scenario.dynamic_obstacles:
        s = obs.initial_state
        r = obs.obstacle_shape
        if hasattr(s, 'acceleration'):
            acc = s.acceleration
        else:
            acc = 0
        vehicles.append({'width': r.width, 'length': r.length, 'x': s.position[0], 'y': s.position[1],
                         'velocity': s.velocity, 'orientation': s.orientation, 'acceleration': acc, 'obj': obs})

    # delete all dynamic obstacles
    for v in vehicles:
        scenario.remove_obstacle(v['obj'])

    # find all possible lanelets for the ego-vehicle
    lanes_ego = scenario.lanelet_network.find_lanelet_by_position([x_ego[0:2]])
    lanes_ego = lanes_ego[0]

    dist_ego = []
    for lane in lanes_ego:
        l = lanelets[lane]
        dist_ego.append(distance_on_lanelet(l, x_ego[0], x_ego[1]))

    # compute maximum distance that can be travelled by the ego vehicle
    r_ego = x_ego[2] * scenario.dt * horizon + 0.5 * 9 * (scenario.dt * horizon)**2

    # loop over all surrounding vehicles
    for v in vehicles:

        # predict x-position of the vehicle under the assumption of constant velocity
        vel = v['velocity'] + scenario.dt * v['acceleration'] * np.linspace(0, horizon, num=horizon+1)
        x = np.zeros((len(vel)))

        for i in range(len(vel)-1):
            if vel[i] > 0:
                x[i+1] = x[i] + vel[i]*scenario.dt + 0.5*v['acceleration']*scenario.dt**2
            else:
                x[i+1] = x[i]

        # check if the vehicle can potentially intervene with the ego vehicle
        if np.sqrt((v['x'] - x_ego[0])**2 + (v['y'] - x_ego[1])**2) > r_ego + x[-1]:
            continue

        # initialize queue with all lanelets corresponding to the initial state of the vehicle
        queue = []
        x0 = CustomState(position=np.array([v['x'], v['y']]), velocity=v['velocity'],
                         orientation=v['orientation'], time_step=0)

        if most_likely:
            lanes = scenario.lanelet_network.find_most_likely_lanelet_by_state([x0])
        else:
            lanes = scenario.lanelet_network.find_lanelet_by_position([np.array([v['x'], v['y']])])
            lanes = lanes[0]

        for lane in lanes:
            l = lanelets[lane]
            dist = distance_on_lanelet(l, v['x'], v['y'])
            queue.append({'states': [x0], 'dist': -dist, 'lanelets': [l.lanelet_id], 'lanelet': l.lanelet_id})

        # loop until queue is empty
        while len(queue) > 0:

            node = queue.pop(0)
            l = lanelets[node['lanelet']]
            states = deepcopy(node['states'])
            lanes = deepcopy(node['lanelets'])

            # loop over all time steps
            for i in range(len(node['states']), horizon):

                dist = x[i] - node['dist']

                # check if the vehicle would crash into the ego-vehicle from behind
                abort = False

                for j in range(len(lanes_ego)):
                    if l.lanelet_id == lanes_ego[j] and dist_ego[j] > dist > dist_ego[j] - 5 - v['length']/2:
                        abort = True
                        break

                if abort:
                    trajectories.append({'trajectory': deepcopy(states), 'lanelets': deepcopy(lanes), 'vehicle': v})
                    break

                # create child nodes from the successor lanelets
                if dist > l.distance[-1]:
                    for s in l.successor:
                        queue.append({'states': states, 'dist': node['dist'] + l.distance[-1], 'lanelets': lanes, 'lanelet': s})
                    break

                # compute states along the current lanelet
                else:
                    for j in range(len(l.distance)-1):
                        if l.distance[j + 1] > dist:
                            d = l.center_vertices[j+1, :] - l.center_vertices[j, :]
                            d = d/np.linalg.norm(d)
                            phi = np.arctan2(d[1], d[0])
                            pos = l.center_vertices[j, :] + d * (dist - l.distance[j])
                            states.append(CustomState(position=pos, velocity=v['velocity'], orientation=phi, time_step=i))
                            lanes.append(l.lanelet_id)
                            break

                # check if trajectory length reached horizon -> finished
                if i == horizon - 1:
                    trajectories.append({'trajectory': deepcopy(states), 'lanelets': deepcopy(lanes), 'vehicle': v})

    # add all trajectories to the CommonRoad traffic scenario
    for traj in trajectories:

        t = traj['trajectory']
        v = traj['vehicle']
        l = traj['lanelets']

        # create the trajectory of the obstacle, starting at time step 0
        dynamic_obstacle_trajectory = Trajectory(0, t)

        # determine lanelets intersected by the vehicle shape
        dynamic_obstacle_shape = Rectangle(width=v['width'], length=v['length'])
        shape_lanelet_assign = {}

        for i in range(len(t)):
            shape_lanelet_assign[i] = {l[i]}
            shape = dynamic_obstacle_shape.rotate_translate_local(t[i].position, t[i].orientation)
            if lanelets[l[i]].polygon.shapely_object.contains(shape.shapely_object):
                for o in overlapping_lanelets[l[i]]:
                    if lanelets[o].polygon.shapely_object.intersects(shape.shapely_object):
                        shape_lanelet_assign[i].add(o)
            else:
                lanes = scenario.lanelet_network.find_lanelet_by_shape(shape)
                for lane in lanes:
                    shape_lanelet_assign[i].add(lane)

        # create the prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape,
                                                           shape_lanelet_assignment=shape_lanelet_assign)

        # generate the dynamic obstacle according to the specification
        dynamic_obstacle_id = scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id, dynamic_obstacle_type, dynamic_obstacle_shape,
                                           t[0], dynamic_obstacle_prediction)

        # add dynamic obstacle to the scenario
        scenario.add_objects(dynamic_obstacle)

    return scenario


def distance_on_lanelet(l, x, y):
    """compute distance of a point along the given lanelet"""

    dist = np.inf

    # loop over all lanelet segments
    for i in range(len(l.distance)-1):

        line = LineString([(l.center_vertices[i, 0], l.center_vertices[i, 1]),
                           (l.center_vertices[i+1, 0], l.center_vertices[i+1, 1])])

        if line.distance(Point(x, y)) < dist:
            p = nearest_points(line, Point(x, y))[0]
            val = l.distance[i] + np.sqrt((p.x - l.center_vertices[i, 0])**2 + (p.y - l.center_vertices[i, 1])**2)
            dist = line.distance(Point(x, y))

    return val
