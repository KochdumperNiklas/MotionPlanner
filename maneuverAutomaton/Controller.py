import numpy as np
import scipy


class Controller:
    """class representing a controller consisting of a sequence of motion primitives"""

    def __init__(self, controllers):

        self.controllers = controllers
        self.cnt = 0

    def get_control_input(self, time, x):
        """returns the control inputs for the control law for the given state at the given time"""

        # check if the time is feasible
        if not (0 <= time <= self.controllers[-1].time[-1]):
            raise Exception('Time is outside the valid time for the controller!')

        # select controller
        if time >= self.controllers[self.cnt].time[-1] - 1e-10:
            self.cnt = self.cnt + 1

        # compute control input
        u = self.controllers[self.cnt].get_control_input(time, x)

        return u


class GeneratorSpaceController:
    """class representing the generator space controller"""

    def __init__(self, U, parallelo, alpha, time, x0):
        """class constructor"""

        # store object properties
        self.U = U
        self.parallelo = parallelo
        self.alpha = alpha
        self.time = time
        self.cnt = -1

        # initialize transformation x_local = T*(x - x0) to local coordinate frame
        phi = x0[3]
        self.T = scipy.linalg.block_diag(np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]), np.eye(2))
        self.x0 = np.array([[x0[0]], [x0[1]], [0], [x0[3]]])

    def get_control_input(self, time, x):
        """returns the control inputs for the control law for the given state at the given time"""

        # update control input if at a control point
        if abs(time - self.time[self.cnt + 1]) < 1e-10:
            self._update_control_input(x)

        # return current control input
        if self.time[self.cnt] - 1e-10 <= time <= self.time[self.cnt + 1] + 1e-10:
            dt = (self.time[self.cnt+1] - self.time[self.cnt])/len(self.alpha[self.cnt])
            ind = np.floor((time - self.time[self.cnt]) / dt)
            u = self.u[:, int(ind)]
        else:
            raise Exception('One of the time points required for computing the control law was skipped!')

        return u

    def _update_control_input(self, x):
        """update the control law"""

        # initialization
        P = self.parallelo[self.cnt+1]
        alpha = self.alpha[self.cnt+1]
        u = np.zeros((self.U['Z'].shape[0], len(alpha)))

        # transform state to local coordinate frame
        x_local = self.T @ (x - self.x0)

        # compute zonotope factors
        beta = np.linalg.solve(P.Z[:, 1:], x_local - P.Z[:, [0]])

        if np.any(abs(beta) > 1 + 1e-5):
            raise Exception('State is located outside of the corresponding parallelotope!')

        # compute control inputs for all time steps
        for i in range(len(alpha)):
            u[:, [i]] = self.U['Z'][:, [0]] + self.U['Z'][:, 1:] @ (alpha[i][:, [0]] + alpha[i][:, 1:] @ beta)

        # store the updated control law
        self.cnt = self.cnt + 1
        self.u = u