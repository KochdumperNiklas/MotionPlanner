from auxiliary.polygonLaneletNetwork import polygonLaneletNetwork
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle
import numpy as np

def collisionChecker(scenario, traj, param):
    """check if the planned trajectory collides with obstancles or with the road boundary"""

    # convert lanelet network to a single polygon
    road = polygonLaneletNetwork(scenario)

    # loop over all time steps
    for i in range(traj.shape[1]):

        # get polygon representing the car
        r = Rectangle(length=param['length'], width=param['width'], center=np.array([traj[0, i], traj[1, i]]),
                      orientation=np.mod(traj[3, i], 2*np.pi))
        car = r.shapely_object

        # check for a collision with the road boundary
        if not road.contains(car):
            return False

        # loop over all obstacles
        for obs in scenario.obstacles:

            pgon = obs.occupancy_at_time(i)

            if pgon is not None:
                pgon = get_shapely_object(pgon.shape)

                if pgon.intersects(car):
                    return False

    return True

def get_shapely_object(set):
    """construct the shapely polygon object for the given CommonRoad set"""

    if hasattr(set, 'shapes'):
        pgon = set.shapes[0].shapely_object
        for i in range(1, len(set.shapes)):
            pgon = pgon.union(set.shapes[i].shapely_object)
    else:
        pgon = set.shapely_object

    return pgon



