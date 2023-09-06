import numpy as np
from scipy.integrate import solve_ivp

def simulate_disturbed_controller(controller, x0, tFinal, disturbance, param):
    """simulate the controlled system for random disturbances"""

    # system dynamics
    ode = lambda t, x, u, w: [x[2] * np.cos(x[3]),
                              x[2] * np.sin(x[3]),
                              u[0] + w[0],
                              x[2] * np.tan(u[1]) / param['wheelbase']] + w[1]

    # transform initial state to rear axis
    x0[0, 0] = x0[0, 0] - np.cos(x0[3, 0]) * param['b']
    x0[1, 0] = x0[1, 0] - np.sin(x0[3, 0]) * param['b']

    # initialization
    dt = 0.01
    N = int(np.ceil(tFinal/dt))
    x = np.zeros((x0.shape[0], N+1))
    u = np.zeros((2, N))

    # simulate the controlled system
    x[:, 0] = x0[:, 0]

    for i in range(N):
        u[:, i] = controller.get_control_input(i*dt, x[:, [i]])
        w = np.random.uniform(low=-disturbance, high=disturbance)
        x[:, i+1] = x[:, i] + ode(0, x[:, i], u[:, i], w) * dt
        #sol = solve_ivp(ode, [0, dt], x[:, i], args=(u[:, i], w), dense_output=True)
        #x[:, i+1] = sol.sol(dt)

    # transform trajectory back to vehicle center
    for i in range(x.shape[1]):
        x[0, i] = x[0, i] + np.cos(x[3, i]) * param['b']
        x[1, i] = x[1, i] + np.sin(x[3, i]) * param['b']

    return x, u