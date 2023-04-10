import os
import numpy as np
from lxml import etree
from shapely.geometry import Point
from shapely.geometry import LineString

from commonroad.scenario.scenario import Tag
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import CustomState
from commonroad.scenario.state import InitialState
from commonroad.common.util import Interval
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader

from crdesigner.map_conversion.map_conversion_interface import opendrive_to_commonroad

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
            if lanelets[id].adj_left is not None:

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
                                                      traffic_lights=l.traffic_lights)

            index.append(id)

    for i in index:
        del lanelets[i]

    network = LaneletNetwork.create_from_lanelet_list(list(lanelets.values()), cleanup_ids=True)
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



if __name__ == "__main__":
    """main entry point"""

    # load OpenDRIVE file, parse it, and convert it to a CommonRoad scenario
    scenario = opendrive_to_commonroad("Town01.xodr")
    #scenario, planning_problem = CommonRoadFileReader('Town01.xml').open()

    # remove small lanelets
    scenario = remove_small_lanelets(scenario)

    # reduce number of points used to represent the lanelet
    scenario = simplify_lanelet_polygons(scenario, 0.01)

    # validate lanelet properties
    scenario = make_lanelets_valid(scenario)

    # define planning problem
    lanelet = 196

    for l in scenario.lanelet_network.lanelets:
        if l.lanelet_id == lanelet:
            goal_state = CustomState(time_step=Interval(0, 0), position=l.polygon)

    #goal_state = CustomState(time_step=Interval(30, 30))
    #goal_region = GoalRegion([goal_state], lanelets_of_goal_position={0: [9]})
    goal_region = GoalRegion([goal_state], lanelets_of_goal_position=None)
    initial_state = InitialState(position=np.array([396.5, -30]), velocity=10, orientation=np.pi/2, yaw_rate=0, slip_angle=0, time_step=0)
    planning_problem = PlanningProblem(1, initial_state, goal_region)

    # store converted file as CommonRoad scenario
    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=PlanningProblemSet([planning_problem]),
        author="Sebastian Maierhofer",
        affiliation="Technical University of Munich",
        source="CommonRoad Scenario Designer",
        tags={Tag.URBAN},
    )
    writer.write_to_file(os.path.dirname(os.path.realpath(__file__)) + "/" + "Town01.xml",
                         OverwriteExistingFile.ALWAYS)



