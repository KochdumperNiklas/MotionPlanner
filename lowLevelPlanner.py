import numpy as np
import matplotlib.pyplot as plt
from casadi import *

R = np.diag([0.01, 100.0])              # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([0.01, 100.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]


def lowLevelPlanner(scenario, planning_problem, param, plan, space, vel):
    """plan a concrete trajectory for the given high-level plan via optimization"""

    # store time step size
    param['time_step'] = scenario.dt

    # convert polygons representing the drivable area to polytopes
    poly = []
    for p in space:
        poly.append(polygon2polytope(p))

    # assemble initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    init_state = planning_problem.initial_state

    x0 = np.array([init_state.position[0], init_state.position[1], init_state.velocity, init_state.orientation])

    # construct reference trajectory
    x_ref = reference_trajectory(space)

    # plan the trajectory via optimization
    x, u = optimal_control_problem(poly, vel, x0, x_ref, param)

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

def reference_trajectory(space):
    """determine parameters for reference trajectory that is defined by the condition a*x_ref + b*y_ref - c = 0"""

    x_ref = []

    # loop over all time steps
    for s in space:

        # get vertices of the polygon
        vx, vy = s.exterior.coords.xy
        v = np.stack((vx, vy))

        # split into upper and lower bound of the lanelet
        n = int(((len(vx)-1)/2))
        v_left = v[:, 0:n]
        v_right = v[:, n:2*n]

        # compute points on the reference trajectory
        p1 = 0.5*(v_left[:, 0] + v_right[:, n-1])
        p2 = 0.5*(v_left[:, n-1] + v_right[:, 0])

        # compute parameter a, b, and c for the reference trajectory a*x_ref + b*y_ref + c = 0
        d = p1 - p2
        a = d[1]
        b = -d[0]
        c = -a*p1[0] - b*p1[1]

        x_ref.append((a, b, c))

    return x_ref

def optimal_control_problem(poly, vel, x0, ref_traj, param):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model(param)

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, len(poly))
    u = opti.variable(nu, len(poly)-1)

    # define cost function
    cost = 0

    for i in range(len(poly)-1):

        # minimize control inputs
        cost += mtimes(mtimes(u[:, i].T, R), u[:, i])

        # minimize difference between consecutive control inputs
        if i < len(poly) - 2:
            cost += mtimes(mtimes((u[:, i] - u[:, i + 1]).T, RD), u[:, i] - u[:, i + 1])

        # minimize distance to reference trajectory
        cost += (ref_traj[i][0] * x[0, i] + ref_traj[i][1] * x[1, i] + ref_traj[i][2])**2

    opti.minimize(cost)

    # constraint (trajectory has to satisfy the differential equation)
    for i in range(len(poly)-1):
        opti.subject_to(x[:, i + 1] == f(x[:, i], u[:, i]))

    # constraint (trajectory has to be inside the safe driving corridor)
    for i in range(len(poly)):

        A = poly[i][0]
        b = poly[i][1]
        opti.subject_to(mtimes(A, x[0:2, i]) <= b)

        opti.subject_to(x[2, i] >= vel[i][0])
        opti.subject_to(x[2, i] <= vel[i][1])

    # constraints on the control input
    opti.subject_to(u[0, :] >= -param['a_max'])
    opti.subject_to(u[0, :] <= param['a_max'])
    opti.subject_to(u[1, :] >= -param['s_max'])
    opti.subject_to(u[1, :] <= param['s_max'])
    opti.subject_to(x[:, 0] == x0)

    # solver settings
    opti.solver('ipopt')

    # solve optimal control problem
    sol = opti.solve()

    # get optimized values for variables
    x_ = sol.value(x)
    u_ = sol.value(u)

    return x_, u_

def vehicle_model(param):
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
                  v * tan(steer) / param['wheelbase'])

    # define integrator
    options = {'tf': param['time_step'], 'simplify': True, 'number_of_finite_elements': 2}
    dae = {'x': x, 'p': u, 'ode': ode}

    intg = integrator('intg', 'rk', dae, options)

    # define a symbolic function x(k+1) = F(x(k),u(k)) representing the integration
    res = intg(x0=x, p=u)
    x_next = res['xf']

    F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    return F, 4, 2
