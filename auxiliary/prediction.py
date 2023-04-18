import numpy as np
from shapely.geometry import Point
from shapely.geometry import LineString
from commonroad.geometry.shape import Rectangle
from shapely.ops import nearest_points
from copy import deepcopy
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

def prediction(vehicles, scenario, horizon, most_likely=True):
    """predict the future positions of the surrounding vehicles"""

    trajectories = []

    # loop over all surrounding vehicles
    for v in vehicles:

        # predict x-position of the vehicle under the assumption of constant velocity
        x = scenario.dt * v['velocity'] * np.linspace(0, horizon)

        # initialize queue with all lanelets corresponding to the initial state of the vehicle
        queue = []
        x0 = CustomState(position=np.array([v['x'], v['y']]), velocity=v['velocity'],
                         orientation=v['orientation'], time_step=0)

        if most_likely:
            lanes = scenario.lanelet_network.find_most_likely_lanelet_by_state([CustomState(
                position=np.array([v['x'], v['y']]), velocity=v['velocity'], orientation=v['orientation'],
                time_step=0)])
        else:
            lanes = scenario.lanelet_network.find_lanelet_by_position([np.array([v['x'], v['y']])])
            lanes = lanes[0]

        for lane in lanes:
            l = scenario.lanelet_network.find_lanelet_by_id(lane)
            dist = distance_on_lanelet(l, v['x'], v['y'])
            queue.append({'states': [x0], 'dist': -dist, 'lanelet': l.lanelet_id})

        # loop until queue is empty
        while len(queue) > 0:

            node = queue.pop(0)
            l = scenario.lanelet_network.find_lanelet_by_id(node['lanelet'])
            states = deepcopy(node['states'])

            # loop over all time steps
            for i in range(len(node['states']), horizon):

                dist = x[i] - node['dist']

                # create child nodes from the successor lanelets
                if dist > l.distance[-1]:
                    for s in l.successor:
                        queue.append({'states': states, 'dist': node['dist'] + l.distance[-1], 'lanelet': s})
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
                            break

                # check if trajectory length reached horizon -> finished
                if i == horizon - 1:
                    trajectories.append(deepcopy(states))

    # add all trajectories to the CommonRoad traffic scenario
    for t in trajectories:

        # create the trajectory of the obstacle, starting at time step 0
        dynamic_obstacle_trajectory = Trajectory(0, t)

        # create the prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_shape = Rectangle(width=v['width'], length=v['length'])
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

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
