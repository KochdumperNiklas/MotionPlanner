import numpy as np
import pickle
from scipy.integrate import solve_ivp
from ManeuverAutomaton import MotionPrimitive
from ManeuverAutomaton import ManeuverAutomaton

# control inputs
accelerations = [-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2]
steer = [-0.4, -0.2, 0, 0.2, 0.4]

# velocity range
v_start = 0
v_end = 30
v_diff = 0.4

# time of the motion primitives
tFinal = 1

# loop over all initial velocities
primitives = []
v_init = v_start

while v_init < v_end:

    # loop over all accelerations
    for acc in accelerations:

        # check if the motion primitive can be connected to other motion primitives
        if v_start <= v_init + acc*tFinal <= v_end:

            # loop over all steering angles
            for st in steer:

                # simulate the system
                ode = lambda t, x, u1, u2: [x[2] * np.cos(x[3]), x[2] * np.sin(x[3]), u1, u2]
                sol = solve_ivp(ode, [0, tFinal], [0, 0, v_init, 0], args=(acc, st), dense_output=True)
                t = np.linspace(0, tFinal, 11)
                x = sol.sol(t)

                # construct the motion primitive
                primitives.append(MotionPrimitive(x, np.array([acc, st]), tFinal))

    v_init = v_init + v_diff

# construct and save maneuver automaton
MA = ManeuverAutomaton(primitives, v_end, v_diff)

filehandler = open('maneuverAutomaton.obj', 'wb')
pickle.dump(MA, filehandler)

