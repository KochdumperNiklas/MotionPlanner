import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from casadi import *
from qpsolvers import solve_qp
from copy import deepcopy
from scipy.integrate import solve_ivp, odeint
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.affinity import translate
import control as ct
import pypoman
from maneuverAutomaton.Controller import FeedbackController

def lowLevelPlannerLinearization(scenario, planning_problem, param, space_all, ref_traj,
                                feedback_control=False, collision_avoidance=False, R=np.diag([0, 0]),
                                R_diff=np.diag([0, 0]), R_feedback=np.diag([0.1, 0.1]),
                                Q_feedback=np.diag([1, 1, 3, 10]), orientation_range_collision_avoidance=np.pi/40):
    """plan a concrete trajectory for the given high-level plan via linear programming"""

    # construct initial guess for the trajectory
    x, u = initial_guess(ref_traj, param)

    # plan the trajectory via quadratic programming
    x_, u_ = optimal_control_problem(x, u, param)

    # simulate the planned trajectory to increase accuracy
    x2, u2 = simulate_trajectory(x_, u_, param)

    # construct feedback controller object that tracks the computed reference trajectory
    controller = construct_feedback_controller(x2, u2, Q_feedback, R_feedback, feedback_control, param)

    return x2, u2, controller

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
        steering_angle.append(np.arctan(param['wheelbase'] * ((ref_traj[3, i+1] - ref_traj[3, i]) / param['time_step']) /
                              np.maximum(0.5 * (ref_traj[2, i+1] + ref_traj[2, i]), 0.1)))

    # construct initial guesses for states and inputs
    steering_angle = np.maximum(-param['s_max'], np.minimum(param['s_max'], np.expand_dims(np.asarray(steering_angle), axis=0)))
    acceleration = np.maximum(-param['a_max'], np.minimum(param['a_max'], np.diff(ref_traj[[2], :]) / param['time_step']))

    x = ref_traj
    u = np.concatenate((acceleration, steering_angle), axis=0)

    return x, u 

def optimal_control_problem(x, u, param):
    """plan a trajectory via quadratic programming"""

    n = x.shape[0]
    m = u.shape[0]
    N = u.shape[1]

    Atot = []
    Atot.append(np.zeros((n, n)))
    ctot = []
    ctot.append(x[:,[0]])

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
        c = np.array([[x_lin[2] * np.cos(x_lin[3])],
                      [x_lin[2] * np.sin(x_lin[3])],
                      [u_lin[0]],
                      [x_lin[2] * np.tan(u_lin[1])/param['wheelbase']]])
        
        A = A * param['time_step']
        B = B * param['time_step']
        c = c * param['time_step']
        
        # propagate system matrices  
        if i == 0:
            A_ = B
        else:
            A_ = np.concatenate(((A + np.eye(n)) @ A_, B), axis = 1)
        Atot.append(np.concatenate((A_, np.zeros((n, m*(N-i-1)))), axis=1))
        ctot.append((A + np.eye(n)) @ ctot[i] + c - A @ np.expand_dims(x_lin, axis=1) - B @ np.expand_dims(u_lin, axis=1))

    # bring optimization problem to quadratic program formulation
    P = 0
    q = 0

    for i in range(u.shape[1]):
        P = P + np.transpose(Atot[i+1]) @ Atot[i+1]
        q = q + np.transpose(ctot[i+1] - x[:, [i+1]]) @ Atot[i+1]

    ub = np.matlib.repmat(np.array([[param['a_max']], [param['s_max']]]), N, 1)
    lb = -ub

    # solve the quadratic program
    z = solve_qp(P, q, lb=lb, ub=ub, solver="cvxopt")

    u_ = np.transpose(z.reshape(N, m))

    # predicted trajectory of the linearized system
    x_ = deepcopy(x)

    for i in range(N):
        x_[:, [i+1]] = Atot[i+1] @ np.expand_dims(z, axis=1) + ctot[i+1]

    return x_, u_

def simulate_trajectory(x, u, param):
    """simulate the planned trajectory to increase the accuracy"""

    x_ = deepcopy(x)
    u_ = deepcopy(u)

    # system dynamics
    ode = lambda t, x, u: [x[2] * np.cos(x[3]),
                           x[2] * np.sin(x[3]),
                           u[0],
                           x[2] * np.tan(u[1]) / param['wheelbase']]

    # loop over all time steps
    for i in range(x.shape[1]-1):
        sol = solve_ivp(ode, [0, param['time_step']], x_[:, i], args=(u_[:, i], ), dense_output=True)
        x_[:, i+1] = sol.sol(param['time_step'])

    # transform trajectory back to vehicle center
    for i in range(x.shape[1]):
        x_[0, i] = x_[0, i] + np.cos(x_[3, i]) * param['b']
        x_[1, i] = x_[1, i] + np.sin(x_[3, i]) * param['b']

    return x_, u_

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
            K.append(-K_[0])

    # construct feedback controller object
    t = param['time_step'] * np.arange(0, x.shape[1]+1)
    controller = FeedbackController(x, u, t, K)

    return controller