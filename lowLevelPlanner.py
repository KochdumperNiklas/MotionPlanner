import numpy as np
import matplotlib.pyplot as plt
from casadi import *

V_MAX = 100
A_MAX = 9
S_MAX = np.deg2rad(24.0)
dt = 0.1
L = 4.3
W = 1.7
WB = 2.4

R = np.diag([0.01, 100.0])              # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([0.01, 100.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]


def lowLevelPlanner(planning_problem, plan, space, space_xy):
    """plan a concrete trajectory for the given high-level plan via optimization"""

    # convert xv-space to polytopes
    poly_xv = []
    for p in space:
        poly_xv.append(polygon2polytope(p))

    # convert xy-space to polytopes
    poly_xy = []
    for p in space_xy:
        poly_xy.append(polygon2polytope(p))

    # assemble initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    init_state = planning_problem.initial_state

    x0 = np.array([init_state.position[0], init_state.position[1], init_state.velocity, init_state.orientation])

    # plan the trajectory via optimization
    x, u = optimal_control_problem(poly_xv, poly_xy, x0)

    return x, u

def polygon2polytope(pgon):
    """convert a polygon to a polytope"""

    # get vertices of the polygon
    vx, vy = pgon.exterior.coords.xy
    v = np.stack((vx, vy))

    # initialize matrices for the polytope
    C = np.zeros((v.shape[1] - 1, 2))
    D = np.zeros((v.shape[1] - 1, 1))

    # loop over all vertices
    for i in range(v.shape[1] - 1):

        dir = v[:, i + 1] - v[:, i]
        c = np.array([[dir[1], -dir[0]]])
        d = np.dot(c, v[:, i])

        if np.max(np.dot(c, v) - d) > 0:
            C[i, :] = -c
            D[i] = -d
        else:
            C[i, :] = c
            D[i] = d

    return C, D

def optimal_control_problem(poly_xv, poly_xy, x0):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model()

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, len(poly_xy))
    u = opti.variable(nu, len(poly_xy)-1)

    # define cost function
    cost = 0

    for i in range(len(poly_xy)-1):

        # minimize control inputs
        cost += mtimes(mtimes(u[:, i].T, R), u[:, i])

        # minimize difference between consecutive control inputs
        if i < len(poly_xy) - 2:
            cost += mtimes(mtimes((u[:, i] - u[:, i + 1]).T, RD), u[:, i] - u[:, i + 1])

    opti.minimize(cost)

    # constraint (trajectory has to satisfy the differential equation)
    for i in range(len(poly_xy)-1):
        opti.subject_to(x[:, i + 1] == f(x[:, i], u[:, i]))

    # constraint (trajectory has to be inside the safe driving corridor)
    for i in range(len(poly_xy)):

        A = poly_xy[i][0]
        b = poly_xy[i][1]
        opti.subject_to(mtimes(A, x[0:2, i]) <= b)

        """A = poly_xv[i][0]
        b = poly_xv[i][1]
        opti.subject_to(mtimes(A, x[[0, 2], i]) <= b)"""

    # constraints on the control input
    opti.subject_to(u[0, :] >= -A_MAX)
    opti.subject_to(u[0, :] <= A_MAX)
    opti.subject_to(u[1, :] >= -S_MAX)
    opti.subject_to(u[1, :] <= S_MAX)
    opti.subject_to(x[:, 0] == x0)

    # solver settings
    opti.solver('ipopt')        # opti.solver('sqpmethod', {'qpsol': 'osqp'})

    # solve optimal control problem
    sol = opti.solve()

    # get optimized values for variables
    x_ = sol.value(x)
    u_ = sol.value(u)

    return x_, u_

def vehicle_model():
    """differential equation describing the dynamic behavior of the car"""

    # states
    sx = MX.sym("sx")
    sy = MX.sym("sy")
    v = MX.sym("v")
    phi = MX.sym("phi")

    x = vertcat(sx, sy, v, phi)

    # control inputs
    acc = MX.sym("acc")
    steer = MX.sym("steer")

    u = vertcat(acc, steer)

    # dynamic function
    ode = vertcat(v * cos(phi),
                  v * sin(phi),
                  acc,
                  v * tan(steer) / WB)

    # define integrator
    options = {'tf': dt, 'simplify': True, 'number_of_finite_elements': 2}
    dae = {'x': x, 'p': u, 'ode': ode}

    intg = integrator('intg', 'rk', dae, options)

    # define a symbolic function x(k+1) = F(x(k),u(k)) representing the integration
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    return F, 4, 2
