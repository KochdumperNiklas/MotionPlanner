import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform

# dimensions of the car
L = 4.3
W = 1.7

class ManeuverAutomaton:
    """class representing a maneuver automaton"""

    def __init__(self, primitives):

        self.primitives = primitives

        # get maximum velocity and velocity difference
        v0 = [p.x[2, 0] for p in primitives]
        v_end = max(v0)
        self.v_diff = np.sort(np.diff(np.sort(np.unique(v0))))[0]

        # construct dictionary mapping velocity to motion primitives
        vel_dic = [[] for i in range(2*np.ceil(v_end/self.v_diff).astype(int))]

        for i in range(len(primitives)):
            v = primitives[i].x[2, 0]
            ind = np.round(v/self.v_diff).astype(int)
            vel_dic[ind].append(i)

        self.vel_dic = vel_dic

        # add successors for all motion primitives
        for i in range(len(primitives)):
            self.primitives[i].successors = self.velocity2primitives(self.primitives[i].x[2, -1])

    def velocity2primitives(self, v):
        """determine all motion primitives starting at the current velocity"""

        return self.vel_dic[np.round(v/self.v_diff).astype(int)]


class MotionPrimitive:
    """class representing a motion primitive"""

    def __init__(self, x, u, tFinal, successors=None, occupancy=None):

        self.x = x                          # states for the reference trajectory
        self.u = u                          # control inputs for the reference trajectory
        self.tFinal = tFinal                # final time for the motion primitive
        self.successors = successors        # list of successor motion primitives

        # compute occupancy set
        if occupancy is None:
            self.occ = construct_occupancy_set(x, tFinal)
        else:
            self.occ = occupancy


def construct_occupancy_set(x, tFinal):
    """compute the space occupied by the car"""

    occ = []
    dt = tFinal/(x.shape[1]-1)

    # shape of the car
    car = Polygon([(-L/2, -W/2), (-L/2, W/2), (L/2, W/2), (L/2, -W/2)])

    # loop over all time steps
    for i in range(x.shape[1]):
        phi = x[3, i]
        tmp = affine_transform(car, [np.cos(phi), -np.sin(phi), np.sin(phi), np.cos(phi), x[0, i], x[1, i]])
        occ.append({'space': tmp, 'time': dt*i})

    return occ
