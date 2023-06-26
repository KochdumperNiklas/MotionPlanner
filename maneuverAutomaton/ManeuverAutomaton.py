import numpy as np

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
        self.occ = occupancy                # space occupied by the car for each time step
