import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import Polygon

def polygonLaneletNetwork(scenario):
    """convert the lanelelt network to a single polygon"""

    # create dictionary that maps a lanelet ID to the corresponding lanelet
    id = [l.lanelet_id for l in scenario.lanelet_network.lanelets]
    lanelets = dict(zip(id, scenario.lanelet_network.lanelets))

    # initialization
    queue = [scenario.lanelet_network.lanelets[0].lanelet_id]
    visited = []
    pgon = lanelet2polygon(lanelets[queue[0]])

    # loop until the queue is empty
    while len(queue) > 0:

        # select first queue element
        node = queue.pop(0)
        l = lanelets[node]

        # convert current lanelet to polygon
        tmp = lanelet2polygon(l)
        pgon = pgon.union(tmp)

        # explore lanelets to the left
        if l.adj_left is not None and l.adj_left not in visited:
            tmp = union_left(l, lanelets[l.adj_left])
            pgon = pgon.union(tmp)
            queue.append(l.adj_left)

        # explore lanelets to the right
        if l.adj_right is not None and l.adj_right not in visited:
            tmp = union_left(lanelets[l.adj_right], l)
            pgon = pgon.union(tmp)
            queue.append(l.adj_right)

        # explore successor lanelets
        for s in l.successor:
            if s not in visited:
                tmp = union_successor(l, lanelets[s])
                pgon = pgon.union(tmp)
                queue.append(s)

        # explore predecessor lanelets
        for p in l.predecessor:
            if p not in visited:
                tmp = union_successor(lanelets[p], l)
                pgon = pgon.union(tmp)
                queue.append(p)

        visited.append(l.lanelet_id)

        # make sure all lanelets have been considered
        if len(queue) == 0:
            for l in lanelets.keys():
                if l not in visited:
                    queue.append(l)

    return pgon

def lanelet2polygon(lanelet):
    """convert a lanelet to a polygon"""

    # get left and right vertices of the lanelet
    left = lanelet.left_vertices.T
    right = lanelet.right_vertices.T

    # combine vertices
    V = np.concatenate((left, np.fliplr(right)), axis=1)

    # construct polygon object
    pgon = Polygon(V.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon

def union_left(lanelet, lanelet_left):
    """unite a lanelet with its left lanelet"""

    # get left and right vertices of the lanelet
    left = lanelet_left.left_vertices.T
    right = lanelet.right_vertices.T

    # combine vertices to form union
    if lanelet.adj_left_same_direction:
        V = np.concatenate((left, np.fliplr(right)), axis=1)
    else:
        V = np.concatenate((left, right), axis=1)

    # construct polygon object
    pgon = Polygon(V.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon

def union_successor(lanelet, lanelet_suc):
    """unite a lanelet with its sucessor lanelet"""

    # get left and right vertices of the lanelet
    left = lanelet.left_vertices.T
    right = lanelet.right_vertices.T

    left_suc = lanelet_suc.left_vertices.T
    right_suc = lanelet_suc.right_vertices.T

    # combine vertices to form union
    V = np.concatenate((left, left_suc, np.fliplr(right_suc), np.fliplr(right)), axis=1)

    # construct polygon object
    pgon = Polygon(V.T)

    if not pgon.is_valid:
        pgon = pgon.buffer(0)

    return pgon
