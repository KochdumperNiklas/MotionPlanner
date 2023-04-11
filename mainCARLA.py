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
import numpy as np
import os
import pickle
from copy import deepcopy

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from auxiliary.prediction import prediction

from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route

import warnings
warnings.filterwarnings("ignore")

HORIZON = 30
REPLAN = 5
SENSORRANGE = 100
MAP = 'Town01'
VISUALIZE = True


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
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
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
    client.set_timeout(2.0)
    world = client.get_world()
    client.load_world(MAP)

    # load parameter for the car
    param = vehicleParameter()

    # load maneuver automaton
    filehandler = open('./maneuverAutomaton/maneuverAutomaton.obj', 'rb')
    MA = pickle.load(filehandler)

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('auxiliary', MAP + '.xml')).open()

    # plan route
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    candidate_holder = route_planner.plan_routes()
    route = candidate_holder.retrieve_first_route()

    # initialization
    state = planning_problem.initial_state
    x0 = np.concatenate((state.position, np.array([state.velocity, state.orientation])))
    start_pose = Transform(Location(x=x0[0], y=-x0[1], z=0.3), Rotation(pitch=0.0, yaw=-np.rad2deg(x0[3]), roll=0.0))
    lanelet = route.list_ids_lanelets[0]
    cnt_goal = 0
    cnt_init = 0
    cnt_time = REPLAN

    # create traffic
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()
    for i in range(0, 50):
        point = random.choice(spawn_points)
        if (point.location.x - x0[0])**2 + (point.location.y + x0[1])**2 > 10**2:
            world.try_spawn_actor(random.choice(vehicle_blueprints), point)
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.set_autopilot(True)

    try:

        # select vehicle model
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # select camara view
        camera_rgb = world.spawn_actor(blueprint_library.find('sensor.camera.rgb'),
                                       carla.Transform(carla.Location(x=-5.5, z=2.8),
                                                       carla.Rotation(pitch=-15)), attach_to=vehicle)
        actor_list.append(camera_rgb)

        # main control loop
        with CarlaSyncMode(world, camera_rgb, fps=10) as sync_mode:
            while True:
                if should_quit():
                    return

                # advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # plan a new trajectory
                if cnt_time < REPLAN:

                    cnt_time = cnt_time + 1

                else:

                    cnt_time = 0

                    # predict the future positions of the other vehicles
                    vehicles = []
                    for v in world.get_actors().filter('*vehicle*'):
                        transform = v.get_transform()
                        if (transform.location.x - x0[0])**2 + (transform.location.y + x0[1])**2 < SENSORRANGE**2 and \
                            (transform.location.x - x0[0]) ** 2 + (transform.location.y + x0[1]) ** 2 > 0.1:
                            orientation = np.deg2rad(transform.rotation.yaw)
                            velocity = v.get_velocity()
                            velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                            length = 2*v.bounding_box.extent.x
                            width = 2*v.bounding_box.extent.y
                            vehicles.append({'width': width, 'length': length, 'x': transform.location.x,
                                             'y': -transform.location.y, 'velocity': velocity,
                                             'orientation': -orientation})

                    scenario_ = prediction(vehicles, deepcopy(scenario), HORIZON)

                    # get lanelet for the current state
                    lane = scenario.lanelet_network.find_lanelet_by_id(lanelet)
                    if not lane.polygon.shapely_object.contains(Point(x0[0], x0[1])):
                        for i in range(cnt_init, len(route.list_ids_lanelets)):
                            l = scenario.lanelet_network.find_lanelet_by_id(route.list_ids_lanelets[i])
                            if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
                                cnt_init = i
                                lanelet = l.lanelet_id

                    # select goal set
                    lane_goal = scenario.lanelet_network.find_lanelet_by_id(route.list_ids_lanelets[cnt_goal])
                    point = 0.5 * (lane_goal.left_vertices[-1, :] + lane_goal.right_vertices[-1, :])
                    d = np.sqrt((x0[0] - point[0]) ** 2 + (x0[1] - point[1]) ** 2)
                    if d < 15:
                        cnt_goal = cnt_goal + 1

                    # create motion planning problem
                    goal_id = route.list_ids_lanelets[cnt_goal]
                    goal_lane = scenario.lanelet_network.find_lanelet_by_id(route.list_ids_lanelets[cnt_goal])
                    goal_state = CustomState(time_step=Interval(HORIZON, HORIZON), position=goal_lane.polygon)
                    goal_region = GoalRegion([goal_state], lanelets_of_goal_position={0: [goal_id]})
                    initial_state = InitialState(position=x0[0:2], velocity=x0[2], orientation=x0[3], yaw_rate=0,
                                                 slip_angle=0, time_step=0)
                    planning_problem = PlanningProblemSet([PlanningProblem(1, initial_state, goal_region)])

                    # solve motion planning problem
                    plan, vel, space, ref_traj = highLevelPlanner(scenario_, planning_problem, param)
                    x, u = lowLevelPlannerManeuverAutomaton(scenario_, planning_problem, param, plan, vel, space,
                                                            ref_traj, MA)

                # update car position and orientation
                x0 = x[:, cnt_time]
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

                    rnd.draw_params.time_begin = cnt_time
                    rnd.draw_params.time_end = HORIZON
                    rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.draw_arrow = False
                    rnd.draw_params.planning_problem_set.planning_problem.initial_state.state.radius = 0
                    scenario.draw(rnd)
                    planning_problem.draw(rnd)

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

                    # plot planned trajectory
                    for j in range(x.shape[1] - 1, -1, -1):
                        if j >= cnt_time:
                            if j == cnt_time:
                                settings = ShapeParams(opacity=1, edgecolor="k", linewidth=1.0, facecolor='r')
                            else:
                                settings = ShapeParams(opacity=0.2, edgecolor="k", linewidth=0.0, facecolor='r')
                            r = Rectangle(length=param['length'], width=param['width'],
                                          center=np.array([x[0, j], x[1, j]]),
                                          orientation=x[3, j])
                            r.draw(rnd, settings)

                    rnd.render()
                    plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
                    plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
                    ax = plt.gca()
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    plt.pause(0.01)


    finally:

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