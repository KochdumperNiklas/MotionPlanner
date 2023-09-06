from commonroad.scenario.obstacle import StaticObstacle
from commonroad.geometry.shape import ShapeGroup
import matplotlib.pyplot as plt

def test_free_space(scenario, free_space):
    """test if the computed free space is valid (= does not intersect any obstacles)"""

    # loop over all obstacles
    for obs in scenario.obstacles:

        for i in range(1, len(free_space)):

            pgon = obs.occupancy_at_time(i)
            if pgon is not None:
                pgon = get_shapely_object(pgon.shape)

                if free_space[i].intersects(pgon) and free_space[i].intersection(pgon).area > 1e-4:
                    raise Exception('Test failed due to intersection with dynamic obstacle!')


def get_shapely_object(set):
    """construct the shapely polygon object for the given CommonRoad set"""

    if hasattr(set, 'shapes'):
        pgon = set.shapes[0].shapely_object
        for i in range(1, len(set.shapes)):
            pgon = pgon.union(set.shapes[i].shapely_object)
    else:
        pgon = set.shapely_object

    return pgon