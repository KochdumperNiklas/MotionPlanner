import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from copy import deepcopy
from scipy.integrate import solve_ivp, odeint
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.affinity import translate
import control as ct
from maneuverAutomaton.Controller import FeedbackController

def lowLevelPlanner(scenario, planning_problem, param, plan, vel, space_all, ref_traj,
                    feedback_control=False, collision_avoidance=False, R=np.diag([0, 0]),
                    R_diff=np.diag([0, 0]), R_feedback=np.diag([0.1, 0.1]), Q_feedback=np.diag([1, 1, 3, 10])):
    """plan a concrete trajectory for the given high-level plan via optimization"""

    # construct initial guess for the trajectory
    init_guess = initial_guess(ref_traj, param)

    # plan the trajectory via optimization
    x, u = optimal_control_problem(ref_traj, init_guess, R, R_diff, param)

    # simulate the planned trajectory to increase accuracy
    x, u = simulate_trajectory(x, u, param)

    # construct feedback controller object that tracks the computed reference trajectory
    controller = construct_feedback_controller(x, u, Q_feedback, R_feedback, feedback_control, param)

    return x, u, controller

def initial_guess(ref_traj, param):
    """compute an initial guess for the state trajectory and the control inputs for the given reference trajectory"""

    # transform reference trajectory to rear axis
    ref_traj = deepcopy(ref_traj)

    for i in range(ref_traj.shape[1]):
        ref_traj[0, i] = ref_traj[0, i] - np.cos(ref_traj[3, i]) * param['b']
        ref_traj[1, i] = ref_traj[1, i] - np.sin(ref_traj[3, i]) * param['b']

    # compute steering angle
    steering_angle = []

    for i in range(0, ref_traj.shape[1]-1):
        steering_angle.append(param['wheelbase'] * ((ref_traj[3, i+1] - ref_traj[3, i]) * param['time_step']) /
                              np.maximum(0.5 * (ref_traj[2, i+1] + ref_traj[2, i]), 0.1))

    # construct initial guesses for states and inputs
    steering_angle = np.maximum(-param['s_max'], np.minimum(param['s_max'], np.expand_dims(np.asarray(steering_angle), axis=0)))
    acceleration = np.maximum(-param['a_max'], np.minimum(param['a_max'], np.diff(ref_traj[[2], :]) / param['time_step']))

    guess = {}
    guess['x'] = ref_traj
    guess['u'] = np.concatenate((acceleration, steering_angle), axis=0)

    return guess

def simulate_trajectory(x, u, param):
    """simulate the planned trajectory to increase the accuracy"""

    # system dynamics
    ode = lambda t, x, u: [x[2] * np.cos(x[3]),
                           x[2] * np.sin(x[3]),
                           u[0],
                           x[2] * np.tan(u[1]) / param['wheelbase']]

    # loop over all time steps
    for i in range(x.shape[1]-1):
        sol = solve_ivp(ode, [0, param['time_step']], x[:, i], args=(u[:, i], ), dense_output=True)
        x[:, i+1] = sol.sol(param['time_step'])

    # transform trajectory back to vehicle center
    for i in range(x.shape[1]):
        x[0, i] = x[0, i] + np.cos(x[3, i]) * param['b']
        x[1, i] = x[1, i] + np.sin(x[3, i]) * param['b']

    return x, u

def construct_feedback_controller(x, u, Q, R, feedback_control, param):
    """construct feedback controller object that tracks the computed reference trajectory"""

    # compute feedback matrices
    if not feedback_control:
        K = [np.zeros((4, 4)) for i in range(u.shape[1])]
    else:

        K = []

        for i in range(u.shape[1]):

            # linearize system dynamics
            x_lin = 0.5 * (x[:, i] + x[:, i + 1])
            u_lin = u[:, i]

            A = np.array([[0, 0, np.cos(x_lin[3]), -x_lin[2]*np.sin(x_lin[3])],
                          [0, 0, np.sin(x_lin[3]), x_lin[2]*np.cos(x_lin[3])],
                          [0, 0, 0, 0],
                          [0, 0, np.tan(u_lin[1])/param['wheelbase'], 0]])
            B = np.array([[0, 0],
                          [0, 0],
                          [1, 0],
                          [0, (x_lin[2]*(np.tan(u_lin[1])**2 + 1))/param['wheelbase']]])

            # compute feedback matrix using LQR controller
            K_ = ct.lqr(A, B, Q, R)
            K.append(K_[0])

    # construct feedback controller object
    t = param['time_step'] * np.arange(0, x.shape[1]+1)
    controller = FeedbackController(x, u, t, K)

    return controller

def optimal_control_problem(ref_traj, init_guess, R, R_diff, param):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model(param)

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, ref_traj.shape[1])
    u = opti.variable(nu, ref_traj.shape[1]-1)

    # construct initial state
    x0 = np.array([param['x0'][0], param['x0'][1], param['v_init'], param['orientation']])

    x0[0] = x0[0] - np.cos(x0[3]) * param['b']
    x0[1] = x0[1] - np.sin(x0[3]) * param['b']

    # define cost function
    cost = 0

    for i in range(ref_traj.shape[1]-1):

        # minimize control inputs
        cost += mtimes(mtimes(u[:, i].T, R), u[:, i])

        # minimize difference between consecutive control inputs
        if i < ref_traj.shape[1] - 2:
            cost += mtimes(mtimes((u[:, i] - u[:, i + 1]).T, R_diff), u[:, i] - u[:, i + 1])

    # minimize distance to reference trajectory
    for i in range(ref_traj.shape[1]):
        cost += (ref_traj[0, i] - x[0, i] - cos(x[3, i]) * param['b'])**2 + \
                (ref_traj[1, i] - x[1, i] - sin(x[3, i]) * param['b'])**2

    opti.minimize(cost)

    # constraint (trajectory has to satisfy the differential equation)
    for i in range(ref_traj.shape[1]-1):
        opti.subject_to(x[:, i + 1] == f(x[:, i], u[:, i]))

    # constraints on the control input
    opti.subject_to(u[0, :] >= -param['a_max'])
    opti.subject_to(u[0, :] <= param['a_max'])
    opti.subject_to(u[1, :] >= -param['s_max'])
    opti.subject_to(u[1, :] <= param['s_max'])
    opti.subject_to(x[2, :] >= -0.01)
    opti.subject_to(x[:, 0] == x0)

    # constraint (Kamm's circle)
    for i in range(ref_traj.shape[1] - 1):
        opti.subject_to(u[0, i]**2 + (x[2, i]**2 * tan(u[1, i]) / param['wheelbase'])**2 <= param['a_max']**2)

    # solver settings
    opti.solver('ipopt')
    opti.set_initial(x, init_guess['x'])
    opti.set_initial(u, init_guess['u'])

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
