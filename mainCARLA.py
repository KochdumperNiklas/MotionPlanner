#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import carla
from carla import *
import random
import pygame
import queue

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import os
import pickle
import cv2
import time
from copy import deepcopy

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization
from auxiliary.prediction import prediction
from auxiliary.overlappingLanelets import overlappingLanelets

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.scenario.traffic_sign import TrafficLight, TrafficLightCycleElement, TrafficLightState
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from commonroad.visualization.util import collect_center_line_colors
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType

import warnings
warnings.filterwarnings("ignore")

MAP = 'Town01'                      # CARLA map
PLANNER = 'Optimization'            # motion planner ('Automaton' or 'Optimization')
HORIZON = 3                         # planning horizon (in seconds)
REPLAN = 0.3                        # time after which the trajectory is re-planned (in seconds)
FREQUENCY = 25                     # control frequency (in Hertz)
SENSORRANGE = 100                   # sensor range of the car (in meters)
CARS = 100                          # number of cars in the map
REAL_DYNAMICS = True                # use real car dynamics from the CARLA vehicle model
VISUALIZE = True                    # visualize the planned trajectory
VIDEO = True                        # create a video


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', FREQUENCY)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():

    # initialize CARLA
    actor_list = []
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(3000)
    client.load_world(MAP)

    # load parameter for the car
    param = vehicleParameter()
    param['a_max'] = 6
    param['wheelbase'] = 2.7

    # load maneuver automaton
    filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
    MA = pickle.load(filehandler)

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('auxiliary', MAP + '.xml')).open()

    # determine overlapping laneletstrajectories
    overlaps = overlappingLanelets(scenario)

    # create random planning problem
    spawn_points = world.get_map().get_spawn_points()
    start = random.choice(spawn_points)
    initial_state = InitialState(position=np.array([start.location.x, -start.location.y]),
                                 velocity=0, orientation=-np.deg2rad(start.rotation.yaw), yaw_rate=0,
                                 slip_angle=0, time_step=0)
    goal = random.choice(spawn_points)
    id = scenario.lanelet_network.find_lanelet_by_position([np.array([goal.location.x, -goal.location.y])])
    l = scenario.lanelet_network.find_lanelet_by_id(id[0][0])
    goal_state = CustomState(position=l.polygon, time_step=Interval(0, 0))
    goal_region = GoalRegion([goal_state], lanelets_of_goal_position=None)
    planning_problem = PlanningProblemSet([PlanningProblem(1, initial_state, goal_region)])

    # plan route
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    candidate_holder = route_planner.plan_routes()
    route = candidate_holder.retrieve_first_route()

    # initialization
    state = planning_problem.initial_state
    x0 = np.concatenate((state.position, np.array([state.velocity, state.orientation])))
    start_pose = Transform(Location(x=x0[0], y=-x0[1], z=0.1), Rotation(pitch=0.0, yaw=-np.rad2deg(x0[3]), roll=0.0))
    lanelet = route.list_ids_lanelets[0]
    cnt_init = 0
    traj = {'x': [x0], 'u': [], 't': [0]}
    comp_time = {'high': [], 'all': []}
    t = REPLAN
    plt.figure(figsize=(7, 7))
    horizon = int(np.round(HORIZON/scenario.dt))

    if VIDEO:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(os.path.join('data', 'videoCARLA.mp4'), fourcc, FREQUENCY, (800+600, 600))

    # set weather
    w = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon,
         carla.WeatherParameters.WetCloudyNoon, carla.WeatherParameters.SoftRainNoon,
         carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon,
         carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset,
         carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.SoftRainSunset,
         carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset]
    index = np.random.randint(0, len(w)-1)
    world.set_weather(w[index])

    # create traffic
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    for i in range(0, CARS):
        point = random.choice(spawn_points)
        if (point.location.x - x0[0])**2 + (-point.location.y - x0[1])**2 > 10**2:
            world.try_spawn_actor(random.choice(vehicle_blueprints), point)
    for v in world.get_actors().filter('*vehicle*'):
        v.set_autopilot(True, traffic_manager.get_port())

    try:

        # select vehicle model
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(REAL_DYNAMICS)
        param['length'] = 2*vehicle.bounding_box.extent.x
        param['width'] = 2*vehicle.bounding_box.extent.y

        # select camara view
        camera_rgb = world.spawn_actor(blueprint_library.find('sensor.camera.rgb'),
                                       carla.Transform(carla.Location(x=-5.5, z=2.8),
                                                       carla.Rotation(pitch=-15)), attach_to=vehicle)
        actor_list.append(camera_rgb)

        # main control loop
        with CarlaSyncMode(world, camera_rgb, fps=FREQUENCY) as sync_mode:
            while True:

                if should_quit():
                    return

                # advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # plan a new trajectory
                if t < REPLAN:

                    t = t + 1/FREQUENCY

                else:

                    t = 0

                    # get current position of the ego vehicle
                    transform = vehicle.get_transform()
                    velocity = vehicle.get_velocity()
                    velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                    x0 = np.array([transform.location.x, -transform.location.y, velocity, -np.deg2rad(transform.rotation.yaw)])

                    # check if goal has been reached
                    if (goal.location.x - x0[0])**2 + (goal.location.y + x0[1])**2 < 10**2:
                        return

                    # predict the future positions of the other vehicles
                    scenario_ = deepcopy(scenario)
                    vehicles = []

                    for v in world.get_actors().filter('*vehicle*'):
                        transform = v.get_transform()
                        if (transform.location.x - x0[0])**2 + (-transform.location.y - x0[1])**2 < SENSORRANGE**2 and \
                            (transform.location.x - x0[0]) ** 2 + (-transform.location.y - x0[1]) ** 2 > 0.1:
                            orientation = np.deg2rad(transform.rotation.yaw)
                            vel = v.get_velocity()
                            velocity = np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
                            acc = v.get_acceleration()
                            acceleration = np.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2) * np.sign(acc.x*vel.x + acc.y*vel.y)
                            length = 2*v.bounding_box.extent.x
                            width = 2*v.bounding_box.extent.y
                            if length == 0:
                                length = 2
                            if width == 0:
                                width = 1

                            state = CustomState(position=np.array([transform.location.x, -transform.location.y]),
                                                velocity=velocity, orientation=-orientation, time_step=0, acceleration=acceleration)
                            shape = Rectangle(width=width, length=length)
                            dynamic_obstacle = DynamicObstacle(scenario.generate_object_id(), ObstacleType.CAR, shape,
                                                               state)
                            scenario_.add_objects(dynamic_obstacle)
                            vehicles.append(Rectangle(center=state.position, orientation=state.orientation,
                                                      width=width, length=length))

                    scenario_ = prediction(scenario_, horizon, x0, overlapping_lanelets=overlaps)

                    # consider red traffic lights
                    points_all = []
                    for traffic_light in world.get_actors().filter('*traffic_light*'):
                        loc = traffic_light.get_location()
                        if str(traffic_light.get_state()) == 'Red' or str(traffic_light.get_state()) == 'Yellow' \
                                and (loc.x - x0[0])**2 + (-loc.y - x0[1])**2 < SENSORRANGE**2:
                            points = []
                            for wp in traffic_light.get_affected_lane_waypoints():
                                points.append(np.array([wp.transform.location.x, -wp.transform.location.y]))
                            points_all = points_all + points
                            lanesPrev = scenario.lanelet_network.find_lanelet_by_position([points[-1]])
                            lanesPrev_ = []
                            for lane in list(np.unique(np.asarray(lanesPrev))):
                                l = scenario.lanelet_network.find_lanelet_by_id(lane)
                                d1 = (l.center_vertices[-1, 0] - points[-1][0])**2 + \
                                     (l.center_vertices[-1, 1] - points[-1][1])**2
                                d2 = (l.center_vertices[-2, 0] - points[-1][0]) ** 2 + \
                                     (l.center_vertices[-2, 1] - points[-1][1]) ** 2
                                if d1 < d2:
                                    lanesPrev_ = lanesPrev_ + l.successor
                                else:
                                    lanesPrev_.append(lane)
                            lanes = []
                            for lane in list(np.unique(np.asarray(lanesPrev_))):
                                try:
                                    l = scenario.lanelet_network.find_lanelet_by_id(lane)
                                    lanes = lanes + l.predecessor
                                except:
                                    test = 1
                            lanes = list(np.unique(np.asarray(lanes)))
                            for lane in lanes:
                                try:
                                    l = scenario.lanelet_network.find_lanelet_by_id(lane)
                                    if not l.adj_left is None and l.adj_left_same_direction:
                                        lanes = lanes + [l.adj_left]
                                    if not l.adj_right is None and l.adj_right_same_direction:
                                        lanes = lanes + [l.adj_right]
                                except:
                                    test = 1
                                lanes = list(np.unique(np.asarray(lanes)))
                            if len(lanes) > 0:
                                cycle = TrafficLightCycleElement(TrafficLightState('red'), horizon * scenario.dt)
                                traffic_light = TrafficLight(scenario_.generate_object_id(), [cycle])
                                _ = scenario_.lanelet_network.add_traffic_light(traffic_light, lanes)

                    # get lanelet for the current state
                    lane = scenario.lanelet_network.find_lanelet_by_id(lanelet)
                    if not lane.polygon.shapely_object.contains(Point(x0[0], x0[1])):
                        for i in range(cnt_init, len(route.list_ids_lanelets)):
                            l = scenario.lanelet_network.find_lanelet_by_id(route.list_ids_lanelets[i])
                            if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
                                cnt_init = i
                                lanelet = l.lanelet_id

                    # create motion planning problem
                    goal_states = []
                    lanelets_of_goal_position = {}

                    dist_max = x0[2] * scenario.dt * horizon + 0.5 * param['a_max'] * (scenario.dt * horizon) ** 2
                    dist = 0

                    for i in range(cnt_init, len(route.list_ids_lanelets)):
                        goal_id = route.list_ids_lanelets[i]
                        goal_lane = scenario.lanelet_network.find_lanelet_by_id(goal_id)
                        goal_states.append(
                            CustomState(time_step=Interval(horizon, horizon), position=goal_lane.polygon))
                        lanelets_of_goal_position[len(goal_states) - 1] = [goal_id]
                        if i > cnt_init:
                            dist = dist + goal_lane.distance[-1]
                        if dist > dist_max:
                            break

                    goal_region = GoalRegion(goal_states, lanelets_of_goal_position=lanelets_of_goal_position)
                    initial_state = InitialState(position=x0[0:2], velocity=x0[2], orientation=x0[3], yaw_rate=0,
                                                 slip_angle=0, time_step=0)
                    planning_problem = PlanningProblemSet([PlanningProblem(1, initial_state, goal_region)])

                    # store planning problem
                    data = {'scenario': scenario_, 'problem': planning_problem}
                    filehandler = open(os.path.join('auxiliary', 'CARLAdata.obj'), 'wb')
                    pickle.dump(data, filehandler)

                    # solve motion planning problem
                    start_time = time.time()
                    try:
                        plan, vel, space, ref_traj = highLevelPlanner(scenario_, planning_problem, param, compute_free_space=False, 
                                                                      minimum_safe_distance=0.5, improve_velocity_profile=True)
                    except Exception as e:
                        print(e)
                        return
                    comp_time['high'].append(time.time() - start_time)

                    if PLANNER == 'Automaton':
                        x, u = lowLevelPlannerManeuverAutomaton(scenario_, planning_problem, param, plan, vel, space,
                                                                ref_traj, MA)
                    elif PLANNER == 'Optimization':
                        x, u, controller = lowLevelPlannerOptimization(scenario_, planning_problem, param, plan, vel,
                                                                       space, ref_traj, feedback_control=True,
                                                                       R_diff=np.diag([0, 0.1]))
                    else:
                        raise Exception('Motion planner not supported! The available planners are "Automaton" and "Optimization".')

                    comp_time['all'].append(time.time() - start_time)

                # update car position and orientation
                if REAL_DYNAMICS:
                    transform = vehicle.get_transform()
                    velocity = vehicle.get_velocity()
                    velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

                    x_meas = np.expand_dims(np.array([transform.location.x, -transform.location.y,
                                                      velocity, -np.deg2rad(transform.rotation.yaw)]), axis=1)

                    traj['x'].append(x_meas[:, 0])

                    x_meas[0, 0] = x_meas[0, 0] - np.cos(x_meas[3, 0]) * param['b']
                    x_meas[1, 0] = x_meas[1, 0] - np.sin(x_meas[3, 0]) * param['b']

                    u = controller.get_control_input(t, x_meas)

                    acc = u[0] / 6.17
                    steer = -u[1]

                    if abs(velocity) < 0.5:
                        steer = 0
                        if all(ref_traj[2, :] < 0.5) and acc > 0:
                            acc = 0

                    if acc > 0:
                        vehicle.apply_control(carla.VehicleControl(throttle=acc, brake=0, steer=steer, manual_gear_shift=True, gear=2))
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=0, brake=-acc, steer=steer, manual_gear_shift=True, gear=2))

                    traj['u'].append(np.array([acc, steer]))
                    traj['t'].append(traj['t'][-1] + 1 / FREQUENCY)

                else:
                    x0 = x[:, np.floor(t/param['time_step'])]
                    pose = Transform(Location(x=x0[0], y=-x0[1], z=0.3),
                                     Rotation(pitch=0.0, yaw=-np.rad2deg(x0[3]), roll=0.0))
                    vehicle.set_transform(pose)

                # draw the display.
                draw_image(display, image_rgb)
                pygame.display.flip()

                # visualize the planned trajectory
                if VISUALIZE:

                    plt.cla()
                    rnd = MPRenderer()
                    canvas = FigureCanvasAgg(rnd.f)

                    cnt_time = int(np.floor(t/param['time_step']))
                    rnd.draw_params.time_begin = cnt_time
                    rnd.draw_params.time_end = horizon
                    rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.draw_arrow = False
                    rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.radius = 0
                    scenario.draw(rnd)
                    planning_problem.draw(rnd)

                    # plot traffic lights
                    if len(scenario_.lanelet_network.traffic_lights) > 0:
                        status = collect_center_line_colors(scenario_.lanelet_network,
                                                            scenario_.lanelet_network.traffic_lights, 0)
                        settings = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, facecolor='r')
                        for l in status.keys():
                            lane = scenario_.lanelet_network.find_lanelet_by_id(l)
                            for j in range(len(lane.distance) - 1):
                                center = 0.5 * (lane.center_vertices[j, :] + lane.center_vertices[j + 1, :])
                                d = lane.center_vertices[j + 1, :] - lane.center_vertices[j, :]
                                r = Rectangle(length=np.linalg.norm(d), width=0.6, center=center,
                                              orientation=np.arctan2(d[1], d[0]))
                                r.draw(rnd, settings)

                    # plot prediction for the other vehicles
                    for d in scenario_.dynamic_obstacles:
                        for j in range(len(d.prediction.trajectory.state_list), 0, -1):
                            s = d.prediction.trajectory.state_list[j - 1]
                            if s.time_step >= cnt_time:
                                if s.time_step == cnt_time:
                                    settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0,
                                                           facecolor='#1d7eea')
                                else:
                                    settings = ShapeParams(opacity=0.2, edgecolor="k", linewidth=0.0,
                                                           facecolor='#1d7eea')
                                r = Rectangle(length=d.obstacle_shape.length, width=d.obstacle_shape.width,
                                              center=s.position,
                                              orientation=s.orientation)
                                r.draw(rnd, settings)
                        if d.prediction.trajectory.state_list[-1].time_step < cnt_time:
                            s = d.prediction.trajectory.state_list[-1]
                            settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='#1d7eea')
                            r = Rectangle(length=d.obstacle_shape.length, width=d.obstacle_shape.width,
                                          center=s.position, orientation=s.orientation)
                            r.draw(rnd, settings)

                    for v in vehicles:
                        found = False
                        for d in scenario_.dynamic_obstacles:
                            if np.linalg.norm(d.initial_state.position - v.center) < 0.1:
                                found = True
                                break
                        if not found:
                            settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='#1d7eea')
                            v.draw(rnd, settings)

                    # plot planned trajectory
                    for j in range(x.shape[1] - 1, -1, -1):
                        if j >= cnt_time:
                            if j == cnt_time:
                                settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='#d95558')
                            else:
                                settings = ShapeParams(opacity=0.2, edgecolor="k", linewidth=0.0, facecolor='#d95558')
                            r = Rectangle(length=param['length'], width=param['width'],
                                          center=np.array([x[0, j], x[1, j]]),
                                          orientation=x[3, j])
                            r.draw(rnd, settings)

                    rnd.render()
                    plt.xlim([x[0, cnt_time] - 40, x[0, cnt_time] + 40])
                    plt.ylim([x[1, cnt_time] - 40, x[1, cnt_time] + 40])
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    plt.pause(0.01)

                # create video
                if VIDEO:

                    # get image from CARLA
                    array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                    frame1 = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
                    frame1 = frame1[:, :, :3]

                    # extract image of planned trajectory from plotted figure
                    canvas.draw()
                    buf = canvas.buffer_rgba()
                    frame2 = np.asarray(buf)
                    frame2 = frame2[53:653, 57:657, :]
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

                    # combine images
                    frame = np.concatenate((frame1, frame2), axis=1)

                    # add frame to video
                    video.write(frame)


    finally:

        # write video to file
        if VIDEO:
            cv2.destroyAllWindows()
            video.release()

        # write computation times to file
        filehandler = open(os.path.join('data', 'computationTime.obj'), 'wb')
        pickle.dump(comp_time, filehandler)

        # write trajectory to file
        filehandler = open(os.path.join('data', 'trajectory.obj'), 'wb')
        pickle.dump(traj, filehandler)

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
