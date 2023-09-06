import os
import numpy as np
from copy import deepcopy
from lxml import etree
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

from commonroad.scenario.scenario import Tag
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.scenario.traffic_sign import TrafficSign, TrafficSignElement, TrafficSignIDZamunda
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader

#from crdesigner.map_conversion.map_conversion_interface import opendrive_to_commonroad

def remove_small_lanelets(scenario):

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    id = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(id, scenario.lanelet_network.lanelets))
    index = []

    # loop over lanelets
    for id in lanelets.keys():

        # check if width of the lanelet is too small
        d = lanelets[id].left_vertices[0, :] - lanelets[id].right_vertices[0, :]
        w = np.linalg.norm(d)

        if w < 1:

            # reassign lanelet ids from the left
            """if lanelets[id].adj_left is not None:

                l = lanelets[lanelets[id].adj_left]
                lanelets[lanelets[id].adj_left] = Lanelet(l.left_vertices, l.center_vertices, lanelets[id].center_vertices,
                                                      l.lanelet_id,
                                                      predecessor=l.predecessor, successor=l.successor,
                                                      adjacent_left=l.adj_left,
                                                      adjacent_left_same_direction=l.adj_left_same_direction,
                                                      adjacent_right=lanelets[id].adj_right,
                                                      adjacent_right_same_direction=lanelets[id].adj_right_same_direction,
                                                      line_marking_left_vertices=l.line_marking_left_vertices,
                                                      line_marking_right_vertices=l.line_marking_right_vertices,
                                                      stop_line=l.stop_line,
                                                      lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                                                      user_bidirectional=l.user_bidirectional,
                                                      traffic_signs=l.traffic_signs, traffic_lights=l.traffic_lights)

            # reassign lanelet ids from the right
            if lanelets[id].adj_right is not None:

                l = lanelets[lanelets[id].adj_right]
                lanelets[lanelets[id].adj_right] = Lanelet(lanelets[id].center_vertices, l.center_vertices,
                                                           l.right_vertices, l.lanelet_id,
                                                      predecessor=l.predecessor, successor=l.successor,
                                                      adjacent_left=lanelets[id].adj_left,
                                                      adjacent_left_same_direction=lanelets[id].adj_left_same_direction,
                                                      adjacent_right=l.adj_right,
                                                      adjacent_right_same_direction=l.adj_right_same_direction,
                                                      line_marking_left_vertices=l.line_marking_left_vertices,
                                                      line_marking_right_vertices=l.line_marking_right_vertices,
                                                      stop_line=l.stop_line,
                                                      lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                                                      user_bidirectional=l.user_bidirectional,
                                                      traffic_signs=l.traffic_signs,
                                                      traffic_lights=l.traffic_lights)"""

            index.append(id)

    for i in index:
        del lanelets[i]

    network = LaneletNetwork.create_from_lanelet_list(list(lanelets.values()), cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def remove_short_lanelets(scenario, length):
    """remove all lanelets that are shorter than the given length"""

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    id = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(id, scenario.lanelet_network.lanelets))
    index = []

    # loop over lanelets
    for i in lanelets.keys():

        if lanelets[i].distance[-1] < length:

            id = lanelets[i].predecessor[0]
            l = lanelets[id]
            center = np.concatenate((l.center_vertices, lanelets[i].center_vertices), axis=0)
            left = np.concatenate((l.left_vertices, lanelets[i].left_vertices), axis=0)
            right = np.concatenate((l.right_vertices, lanelets[i].right_vertices), axis=0)

            lanelets[id] = Lanelet(left, center, right, l.lanelet_id,
                                  predecessor=l.predecessor, successor=lanelets[i].successor, adjacent_left=l.adj_left,
                                  adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                                  adjacent_right_same_direction=l.adj_right_same_direction,
                                  line_marking_left_vertices=l.line_marking_left_vertices,
                                  line_marking_right_vertices=l.line_marking_right_vertices,
                                  stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                                  user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                                  traffic_lights=l.traffic_lights)

            for id_ in lanelets[i].successor:


                l = deepcopy(lanelets[id_])

                lanelets[id_] = Lanelet(l.left_vertices, l.center_vertices, l.right_vertices, l.lanelet_id,
                                      predecessor=[lanelets[id].lanelet_id], successor=l.successor,
                                      adjacent_left=l.adj_left,
                                      adjacent_left_same_direction=l.adj_left_same_direction,
                                      adjacent_right=l.adj_right,
                                      adjacent_right_same_direction=l.adj_right_same_direction,
                                      line_marking_left_vertices=l.line_marking_left_vertices,
                                      line_marking_right_vertices=l.line_marking_right_vertices,
                                      stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                                      user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                                      traffic_lights=l.traffic_lights)

            index.append(i)

    for i in index:
        del lanelets[i]

    network = LaneletNetwork.create_from_lanelet_list(list(lanelets.values()), cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def remove_outer_lanelets(scenario):
    """remove all lanelelts that do not have a left and right neighbor"""

    # get list of lanelets
    network = scenario.lanelet_network
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    for l in lanelets:

        if l.adj_right is None or l.adj_left is None:
            network.remove_lanelet(l.lanelet_id, rtree=True)

    scenario.replace_lanelet_network(network)

    return scenario

def connect_successors(scenario):
    """make sure successor lanelets start at the same vertices as the previous lanelet"""

    # get list of lanelets
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    for i in range(len(lanelets)):

        l = lanelets[i]

        if len(l.successor) > 0:

            l_ = scenario.lanelet_network.find_lanelet_by_id(l.successor[0])

            if l.lanelet_id == 4 and l_.lanelet_id == 199:

                left = l.left_vertices
                right = l.right_vertices
                center = l.center_vertices
                left[-1, :] = l_.left_vertices[0, :]
                right[-1, :] = l_.right_vertices[0, :]
                center[-1, :] = l_.center_vertices[0, :]

                lanelets[i] = Lanelet(left, center, right, l.lanelet_id,
                                  predecessor=l.predecessor, successor=l.successor, adjacent_left=l.adj_left,
                                  adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                                  adjacent_right_same_direction=l.adj_right_same_direction,
                                  line_marking_left_vertices=l.line_marking_left_vertices,
                                  line_marking_right_vertices=l.line_marking_right_vertices,
                                  stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                                  user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs, traffic_lights=lanelets[0].traffic_lights)

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def make_lanelets_valid(scenario):
    """make lanelets valid by replacing wrong properties"""

    # get list of lanelets
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    for i in range(len(lanelets)):
        l = lanelets[i]
        lanelets[i] = Lanelet(l.left_vertices, l.center_vertices, l.right_vertices, l.lanelet_id,
                              predecessor=l.predecessor, successor=l.successor, adjacent_left=l.adj_left,
                              adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                              adjacent_right_same_direction=l.adj_right_same_direction,
                              line_marking_left_vertices=l.line_marking_left_vertices,
                              line_marking_right_vertices=l.line_marking_right_vertices,
                              stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                              user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs, traffic_lights=lanelets[0].traffic_lights)

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def remove_unreachable_lanelets(scenario, start):
    """remove all lanelets that are not reachable from the given starting lanelet"""

    # determine all reachable lanelets
    queue = [start]
    traversed = []

    while len(queue) > 0:

        id = queue.pop(0)
        traversed.append(id)
        l = scenario.lanelet_network.find_lanelet_by_id(id)

        for s in l.successor:
            if not s in traversed and not s in queue:
                queue.append(s)

        if not l.adj_left is None and l.adj_left_same_direction and not l.adj_left in traversed and not l.adj_left in queue:
            queue.append(l.adj_left)

        if not l.adj_right is None and l.adj_right_same_direction and not l.adj_right in traversed and not l.adj_right in queue:
            queue.append(l.adj_right)

    # remove all unreachable lanelets
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    lanelets_ = []
    for l in lanelets:
        if l.lanelet_id in traversed:
            lanelets_.append(l)

    network = LaneletNetwork.create_from_lanelet_list(lanelets_, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def remove_lanelets_without_successor(scenario):
    """remove all lanelets that do not have a successor or predecessor"""

    found = True

    while found:

        lanelets = []
        found = False

        for l in scenario.lanelet_network.lanelets:

            if len(l.successor) > 0 and len(l.predecessor) > 0:
                lanelets.append(l)
            else:
                found = True

        network = LaneletNetwork.create_from_lanelet_list(deepcopy(lanelets), cleanup_ids=True)
        scenario.replace_lanelet_network(network)

    return scenario

def mirror_lanelets(scenario, index):
    """mirror the coordinate given by index"""

    # construct tranformation matrix
    T = np.zeros((2, 2))
    T[0, 0] = 1
    T[1, 1] = 1
    T[index, index] = -T[index, index]

    # get list of lanelets
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    for j in range(len(lanelets)):

        l = lanelets[j]

        left = l.left_vertices @ T
        right = l.right_vertices @ T
        center = l.center_vertices @ T

        lanelets[j] = Lanelet(right, center, left, l.lanelet_id,
                              predecessor=l.predecessor, successor=l.successor, adjacent_left=l.adj_left,
                              adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                              adjacent_right_same_direction=l.adj_right_same_direction,
                              line_marking_left_vertices=l.line_marking_left_vertices,
                              line_marking_right_vertices=l.line_marking_right_vertices,
                              stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                              user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                              traffic_lights=lanelets[0].traffic_lights)

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def split_lanelet(scenario, point):
    """split the lanelet at the given point"""

    # get lanelet
    l = scenario.lanelet_network.find_lanelet_by_position([point])

    lanelets = scenario.lanelet_network.lanelets

    for i in range(len(lanelets)):
        if lanelets[i].lanelet_id == l[0][0]:
            index = i

    l = lanelets[index]

    # get distance on the lanelet
    dist = np.inf

    for i in range(len(l.distance) - 1):

        line = LineString([(l.center_vertices[i, 0], l.center_vertices[i, 1]),
                           (l.center_vertices[i + 1, 0], l.center_vertices[i + 1, 1])])

        if line.distance(Point(point[0], point[1])) < dist:
            p = nearest_points(line, Point(point[0], point[1]))[0]
            val = l.distance[i] + np.sqrt((p.x - l.center_vertices[i, 0]) ** 2 + (p.y - l.center_vertices[i, 1]) ** 2)
            dist = line.distance(Point(point[0], point[1]))
            ind = i

    """diff = l.center_vertices.T - np.expand_dims(np.asarray(point), axis=1) @ np.ones((1, len(l.distance)))
    dist = np.sum(diff**2, axis=0)
    ind = np.argmin(dist)"""

    # construct intermediate point
    factor = (val - l.distance[ind])/(l.distance[ind+1] - l.distance[ind])
    diff = l.center_vertices[ind+1, :]-l.center_vertices[ind, :]
    diff = diff/np.linalg.norm(diff)

    p_center = l.center_vertices[[ind], :] + factor*(l.center_vertices[[ind+1], :]-l.center_vertices[[ind], :])

    line = LineString([(p_center[0, 0] - diff[1]*10, p_center[0, 1] + diff[0]*10),
                       (p_center[0, 0] + diff[1]*10, p_center[0, 1] - diff[0]*10)])
    line_left = LineString([(l.left_vertices[ind, 0], l.left_vertices[ind, 1]),
                       (l.left_vertices[ind + 1, 0], l.left_vertices[ind+1, 1])])
    p_left = line.intersection(line_left)
    p_left = np.array([[p_left.x, p_left.y]])

    line_right = LineString([(l.right_vertices[ind, 0], l.right_vertices[ind, 1]),
                            (l.right_vertices[ind + 1, 0], l.right_vertices[ind + 1, 1])])
    p_right = line.intersection(line_right)
    p_right = np.array([[p_right.x, p_right.y]])

    # construct split lanelets
    id = scenario.generate_object_id()

    l1 = Lanelet(np.concatenate((l.right_vertices[:ind+1, :], p_right), axis=0),
                 np.concatenate((l.center_vertices[:ind+1, :], p_center), axis=0),
                 np.concatenate((l.left_vertices[:ind+1, :], p_left), axis=0), l.lanelet_id,
                          predecessor=l.predecessor, successor=[id], adjacent_left=None,
                          adjacent_left_same_direction=False, adjacent_right=None,
                          adjacent_right_same_direction=False,
                          line_marking_left_vertices=l.line_marking_left_vertices,
                          line_marking_right_vertices=l.line_marking_right_vertices,
                          stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                          user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                          traffic_lights=l.traffic_lights)

    l2 = Lanelet(np.concatenate((p_right, l.right_vertices[ind+1:, :]), axis=0),
                 np.concatenate((p_center, l.center_vertices[ind+1:, :]), axis=0),
                 np.concatenate((p_left, l.left_vertices[ind+1:, :]), axis=0), id,
                 predecessor=[l.lanelet_id], successor=l.successor, adjacent_left=None,
                 adjacent_left_same_direction=False, adjacent_right=None,
                 adjacent_right_same_direction=False,
                 line_marking_left_vertices=l.line_marking_left_vertices,
                 line_marking_right_vertices=l.line_marking_right_vertices,
                 stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                 user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                 traffic_lights=l.traffic_lights)

    lanelets[index] = l1
    lanelets.append(l2)

    for s in l.successor:
        for i in range(len(lanelets)):
            if lanelets[i].lanelet_id == s:
                pre = lanelets[i].predecessor
                for j in range(len(pre)):
                    if pre[j] == l.lanelet_id:
                        pre[j] = id
                lanelets[i].predecessor = pre

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def add_speed_limit(scenario, limit):
    """add a speed limit to all lanelets"""

    # create traffic sign
    elem = TrafficSignElement(TrafficSignIDZamunda.MAX_SPEED, [str(limit)])

    # loop over all lanelets
    for l in scenario.lanelet_network.lanelets:

        ts = TrafficSign(scenario.generate_object_id(), [elem], l.lanelet_id)
        scenario.lanelet_network.add_traffic_sign(ts, [l.lanelet_id])

    return scenario

def remove_line_markings(scenario):
    """remove all line markings from the lanelets"""

    lanelets = []

    for l in scenario.lanelet_network.lanelets:

        lanelets.append(Lanelet(l.left_vertices, l.center_vertices, l.right_vertices, l.lanelet_id,
                              predecessor=l.predecessor, successor=l.successor, adjacent_left=l.adj_left,
                              adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                              adjacent_right_same_direction=l.adj_right_same_direction,
                              line_marking_left_vertices=None,
                              line_marking_right_vertices=None,
                              stop_line=None, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                              user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                              traffic_lights=l.traffic_lights))

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def simplify_lanelet_polygons(scenario, tol):
    """reduce the number of points for the lanelet polygons"""

    # get list of lanelets
    lanelets = scenario.lanelet_network.lanelets

    # loop over lanelets
    for j in range(len(lanelets)):

        # loop over all vertices
        l = lanelets[j]
        left = [l.left_vertices[0, :]]
        right = [l.right_vertices[0, :]]
        center = [l.center_vertices[0, :]]

        for i in range(1, l.left_vertices.shape[0]-1):

            # compute distance of current point to lane segment without the point
            p = Point(l.left_vertices[i, 0], l.left_vertices[i, 1])
            line = LineString([(left[-1][0], left[-1][1]), (l.left_vertices[i+1, 0], l.left_vertices[i+1, 1])])
            dist1 = p.distance(line)

            p = Point(l.right_vertices[i, 0], l.right_vertices[i, 1])
            line = LineString([(right[-1][0], right[-1][1]), (l.right_vertices[i + 1, 0], l.right_vertices[i + 1, 1])])
            dist2 = p.distance(line)

            if dist1 > tol or dist2 > tol:
                left.append(l.left_vertices[i, :])
                right.append(l.right_vertices[i, :])
                center.append(l.center_vertices[i, :])

        left.append(l.left_vertices[-1, :])
        right.append(l.right_vertices[-1, :])
        center.append(l.center_vertices[-1, :])

        lanelets[j] = Lanelet(np.asarray(left), np.asarray(center), np.asarray(right), l.lanelet_id,
                              predecessor=l.predecessor, successor=l.successor, adjacent_left=l.adj_left,
                              adjacent_left_same_direction=l.adj_left_same_direction, adjacent_right=l.adj_right,
                              adjacent_right_same_direction=l.adj_right_same_direction,
                              line_marking_left_vertices=l.line_marking_left_vertices,
                              line_marking_right_vertices=l.line_marking_right_vertices,
                              stop_line=l.stop_line, lanelet_type=l.lanelet_type, user_one_way=l.user_one_way,
                              user_bidirectional=l.user_bidirectional, traffic_signs=l.traffic_signs,
                              traffic_lights=lanelets[0].traffic_lights)

    network = LaneletNetwork.create_from_lanelet_list(lanelets, cleanup_ids=True)
    scenario.replace_lanelet_network(network)

    return scenario

def remove_all_traffic_signs(scenario):
    """remove all traffic signs from the scenario"""

    ids = []
    for s in scenario.lanelet_network.traffic_signs:
        ids.append(s.traffic_sign_id)
    for id in ids:
        scenario.lanelet_network.remove_traffic_sign(id)

    return scenario

if __name__ == "__main__":
    """main entry point"""

    # load OpenDRIVE file, parse it, and convert it to a CommonRoad scenario
    #scenario = opendrive_to_commonroad("Town03.xodr")
    scenario, planning_problem = CommonRoadFileReader('Town01.xml').open()

    """scenario = split_lanelet(scenario, [2, -11.24])
    scenario = split_lanelet(scenario, [-2, -11.24])

    scenario = split_lanelet(scenario, [392, -11.24])
    scenario = split_lanelet(scenario, [396, -11.24])

    scenario = split_lanelet(scenario, [2, -317.3])
    scenario = split_lanelet(scenario, [-2, -317.3])

    scenario = split_lanelet(scenario, [392, -317.3])
    scenario = split_lanelet(scenario, [396, -317.3])"""

    test = 1

    # remove small lanelets
    """scenario = remove_small_lanelets(scenario)

    # reduce number of points used to represent the lanelet
    scenario = simplify_lanelet_polygons(scenario, 0.05)

    # remove outer lanelets
    #scenario = remove_outer_lanelets(scenario)

    # remove lanelets that are not reachable from the starting lanelet
    scenario = remove_unreachable_lanelets(scenario, 429)

    # remove lanelets that do not have a successor or predecessor
    scenario = remove_lanelets_without_successor(scenario)

    # remove short lanelets
    scenario = remove_short_lanelets(scenario, 3)

    # make sure all successors-predecessor pairs share a common point
    scenario = connect_successors(scenario)

    # remove line markings from the lanelets
    scenario = remove_line_markings(scenario)

    # add speed limit
    #scenario = add_speed_limit(scenario, 8.3)
    scenario = add_speed_limit(scenario, 6)

    # mirror y coordinate
    #scenario = mirror_lanelets(scenario, 1)

    # validate lanelet properties
    #scenario = make_lanelets_valid(scenario)

    # define planning problem
    lanelet = 196
    lanelet = 1
    lanelet = scenario.lanelet_network.find_lanelet_by_position([np.array([76, -48])])[0][0]
    #lanelet = scenario.lanelet_network.find_lanelet_by_position([np.array([185, -195])])[0][0]
    lanelet = scenario.lanelet_network.find_lanelet_by_position([np.array([0, 170])])[0][0]

    for l in scenario.lanelet_network.lanelets:
        if l.lanelet_id == lanelet:
            goal_state = CustomState(time_step=Interval(0, 0), position=l.polygon)

    #goal_state = CustomState(time_step=Interval(30, 30))
    #goal_region = GoalRegion([goal_state], lanelets_of_goal_position={0: [9]})
    goal_region = GoalRegion([goal_state], lanelets_of_goal_position=None)
    initial_state = InitialState(position=np.array([-89, 110]), velocity=0, orientation=-np.pi / 2, yaw_rate=0,
                                 slip_angle=0, time_step=0)
    initial_state = InitialState(position=np.array([140, 205]), velocity=0, orientation=-np.pi, yaw_rate=0,
                                 slip_angle=0, time_step=0)
    #initial_state = InitialState(position=np.array([396.5, -30]), velocity=0, orientation=np.pi/2, yaw_rate=0, slip_angle=0, time_step=0)
    planning_problem = PlanningProblem(1, initial_state, goal_region)"""

    # store converted file as CommonRoad scenario
    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=planning_problem,
        author="Sebastian Maierhofer",
        affiliation="Technical University of Munich",
        source="CommonRoad Scenario Designer",
        tags={Tag.URBAN},
    )
    writer.write_to_file(os.path.dirname(os.path.realpath(__file__)) + "/" + "Town01.xml",
                         OverwriteExistingFile.ALWAYS)



