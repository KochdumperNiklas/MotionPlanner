import numpy as np
import pickle
from scipy.integrate import solve_ivp
import os
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

# final time and time step size for the motion primitives
tFinal = 1
dt = 0.1

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

        # modify acceleration and final time to ensure that the motion primitive can be connected to others
        if v_start <= v_init + acc*tFinal <= v_end:
            tFinal_ = tFinal
        elif v_init + acc*tFinal < v_start:
            tFinal_ = np.round(((v_start - v_init)/acc)/dt) * dt
            acc = (v_start - v_init)/tFinal_
        else:
            tFinal_ = np.round(((v_start - v_end) / acc)/dt) * dt
            acc = (v_start - v_end) / tFinal_

        if tFinal_ > 0:

            # loop over all final orientations
            for o in orientation:

                if o == 0 or abs(v_init * tFinal_ + 0.5*acc * tFinal_**2) > 0:

                    # compute the required steering angle to achieve the desired final orientation
                    if o == 0:
                        steer = 0
                    else:
                        steer = np.arctan(wb * o / (v_init * tFinal_ + 0.5*acc * tFinal_**2))

                    if abs(steer) < s_max:

                        # simulate the system
                        ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]) - wb/2 * np.sin(x[3]) * x[2] * np.tan(u2) / wb,
                                                    x[2] * np.sin(x[3]) + wb/2 * np.cos(x[3]) * x[2] * np.tan(u2) / wb,
                                                    u1,
                                                    x[2] * np.tan(u2) / wb]
                        sol = solve_ivp(ode, [0, tFinal_], [0, 0, v_init, 0], args=(acc, steer), dense_output=True)
                        t = np.linspace(0, tFinal_, int(np.round(tFinal_/dt))+1)
                        x = sol.sol(t)

                        # construct the motion primitive
                        primitives.append(MotionPrimitive(x, np.array([acc, steer]), tFinal_))

    v_init = v_init + v_diff

# construct and save maneuver automaton
MA = ManeuverAutomaton(primitives)

filehandler = open(os.path.join('maneuverAutomaton', 'maneuverAutomaton.obj'), 'wb')
pickle.dump(MA, filehandler)

