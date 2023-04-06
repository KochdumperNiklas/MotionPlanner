import numpy as np
import pickle
from scipy.integrate import solve_ivp

import sys
sys.path.append('./')

from vehicle.vehicleParameter import vehicleParameter
from ManeuverAutomaton import MotionPrimitive
from ManeuverAutomaton import ManeuverAutomaton

# acceleration
accelerations = [-8, -4, -2, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 2, 4, 8]

# desired final orientation
orientation = [-1, -0.8, -0.6, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]

# velocity range
v_start = 0
v_end = 30
v_diff = 0.4

# time of the motion primitives
tFinal = 1

# load parameter for the car
param = vehicleParameter()
wb = param['wheelbase']
s_max = param['s_max']

# loop over all initial velocities
primitives = []
v_init = v_start

while v_init < v_end:

    # loop over all accelerations
    for acc in accelerations:

        # check if the motion primitive can be connected to other motion primitives
        if v_start <= v_init + acc*tFinal <= v_end:

            # loop over all final orientations
            for o in orientation:

                if abs(v_init * tFinal + 0.5*acc * tFinal**2) > 0:

                    # compute the required steering angle to achieve the desired final orientation
                    steer = np.arctan(wb * o / (v_init * tFinal + 0.5*acc * tFinal**2))

                    if abs(steer) < s_max:

                        # simulate the system
                        ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]) - wb/2 * np.sin(x[3]) * x[2] * np.tan(u2) / wb,
                                                    x[2] * np.sin(x[3]) + wb/2 * np.cos(x[3]) * x[2] * np.tan(u2) / wb,
                                                    u1,
                                                    x[2] * np.tan(u2) / wb]
                        sol = solve_ivp(ode, [0, tFinal], [0, 0, v_init, 0], args=(acc, steer), dense_output=True)
                        t = np.linspace(0, tFinal, 11)
                        x = sol.sol(t)

                        # construct the motion primitive
                        primitives.append(MotionPrimitive(x, np.array([acc, steer]), tFinal))

    v_init = v_init + v_diff

# construct and save maneuver automaton
MA = ManeuverAutomaton(primitives, v_end, v_diff)

filehandler = open('maneuverAutomaton.obj', 'wb')
pickle.dump(MA, filehandler)

