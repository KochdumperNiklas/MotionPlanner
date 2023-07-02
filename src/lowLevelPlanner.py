import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pypoman
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.affinity import translate
from commonroad.common.solution import CommonRoadSolutionWriter, Solution, PlanningProblemSolution, VehicleModel, VehicleType, CostFunction
from commonroad.scenario.state import KSState, InputState
from commonroad.scenario.trajectory import Trajectory
from vehiclemodels.init_ks import init_ks
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks

R = np.diag([1, 10.0])              # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([1, 10.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]

def lowLevelPlanner(scenario, planning_problem, param, plan, vel, space_all, ref_traj):
    """plan a concrete trajectory for the given high-level plan via optimization"""

    # construct initial guess for the trajectory
    init_guess = initial_guess(ref_traj, param)

    # plan the trajectory via optimization
    x, u = optimal_control_problem(ref_traj, init_guess, vel, param)

    # simulate the planned trajectory to increase accuracy
    x, u = simulate_trajectory(x, u, param)

    # create a CommonRoad solution object
    sol = create_solution_object(scenario, planning_problem, x, u)

    # write solution to a xml file
    csw = CommonRoadSolutionWriter(sol)
    name_scenario = str(scenario.scenario_id)
    solution_name = f"solution_KS1:TR1:{name_scenario}:2020a.xml"

    csw.write_to_file(output_path="/home/niklas/Documents/Repositories/MotionPlanner/vehicle", filename=solution_name, overwrite=True)

    return x, u, sol

def initial_guess(ref_traj, param):
    """compute an initial guess for the state trajectory and the control inputs for the given reference trajectory"""

    # compute steering angle
    steering_angle = [0]

    for i in range(1, ref_traj.shape[1]-1):
        steering_angle.append(param['wheelbase'] * ((ref_traj[3, i+1] - ref_traj[3, i]) * param['time_step']) /
                              np.maximum(0.5 * (ref_traj[2, i+1] + ref_traj[2, i]), 0.1))

    steering_angle.append(steering_angle[-1])

    # construct initial guesses for states and inputs
    steering_angle = np.maximum(-param['s_max'], np.minimum(param['s_max'], np.expand_dims(np.asarray(steering_angle), axis=0)))

    steer_vel = np.maximum(-param['svel_max'], np.minimum(param['svel_max'], np.diff(steering_angle) / param['time_step']))
    acceleration = np.maximum(-param['a_max'], np.minimum(param['a_max'], np.diff(ref_traj[[2], :]) / param['time_step']))

    guess = {}
    guess['x'] = np.concatenate((ref_traj[0:2, :], steering_angle, ref_traj[2:, :]), axis=0)
    guess['u'] = np.concatenate((acceleration, steer_vel), axis=0)

    return guess

def simulate_trajectory(x, u, param):
    """simulate the planned trajectory to increase the accuracy"""

    def func_KS(t, x, u, p):
        f = vehicle_dynamics_ks(x, u, p)
        return f

    # loop over all time steps
    p = parameters_vehicle1()

    for i in range(x.shape[1]-1):
        sol = solve_ivp(func_KS, [0, param['time_step']], x[:, i], args=(u[[1, 0], i], p), dense_output=True)
        x[:, i+1] = sol.sol(param['time_step'])

    return x, u

def create_solution_object(scenario, planning_problem, x, u):
    """create a CommonRoad solution object from the computed solution trajectory"""

    # generate state list of the ego vehicle's trajectory
    state_list = []

    for i in range(x.shape[1]-1):
        """state_list.append(KSState(**{'position': x[0:2, i], 'time_step': i, 'velocity': x[3, i],
                                     'orientation': x[4, i], 'steering_angle': x[2, i]}))"""
        state_list.append(InputState(**{'steering_angle_speed': u[1, i], 'acceleration': u[0, i], 'time_step': i}))

    # create a trajectory object
    trajectory = Trajectory(initial_time_step=0, state_list=state_list)

    # create a planning problem solution object
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]

    pps = PlanningProblemSolution(planning_problem_id=planning_problem.planning_problem_id,
                                  vehicle_type=VehicleType.FORD_ESCORT,
                                  vehicle_model=VehicleModel.KS,
                                  cost_function=CostFunction.TR1,
                                  trajectory=trajectory)

    # define the object with necessary attributes.
    solution = Solution(scenario.scenario_id, [pps])

    return solution

def optimal_control_problem(ref_traj, init_guess, vel, param):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model(param)

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, len(vel))
    u = opti.variable(nu, len(vel)-1)

    # construct initial state
    x0 = np.array([param['x0'][0], param['x0'][1], 0, param['v_init'], param['orientation']])

    # define cost function
    cost = 0

    for i in range(len(vel)-1):

        # minimize control inputs
        cost += mtimes(mtimes(u[:, i].T, R), u[:, i])

        # minimize difference between consecutive control inputs
        if i < len(vel) - 2:
            cost += mtimes(mtimes((u[:, i] - u[:, i + 1]).T, RD), u[:, i] - u[:, i + 1])

        # minimize distance to reference trajectory
        cost += (ref_traj[0, i] - x[0, i])**2 + (ref_traj[1, i] - x[1, i])**2

    opti.minimize(cost)

    # constraint (trajectory has to satisfy the differential equation)
    for i in range(len(vel)-1):
        opti.subject_to(x[:, i + 1] == f(x[:, i], u[:, i]))

    # constraints on the control input
    opti.subject_to(u[0, :] >= -param['a_max'])
    opti.subject_to(u[0, :] <= param['a_max'])
    opti.subject_to(u[1, :] >= -param['svel_max'])
    opti.subject_to(u[1, :] <= param['svel_max'])
    opti.subject_to(x[2, :] >= -param['s_max'])
    opti.subject_to(x[2, :] <= param['s_max'])
    opti.subject_to(x[:, 0] == x0)

    # constraint (Kamm's circle)
    for i in range(len(vel) - 1):
        opti.subject_to(u[0, i]**2 + (x[3, i]**2 * np.tan(x[2, i]) / param['wheelbase'])**2 <= param['a_max']**2)

    # solver settings
    opti.solver('ipopt', {'verbose': False, 'ipopt.print_level': 0})
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
    steer = MX.sym("steer")
    v = MX.sym("v")
    phi = MX.sym("phi")

    x = vertcat(sx, sy, steer, v, phi)

    # control inputs
    acc = MX.sym("acc")
    steervel = MX.sym("steervel")

    u = vertcat(acc, steervel)

    # dynamic function
    ode = vertcat(v * cos(phi),
                  v * sin(phi),
                  steervel,
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

    return F, 5, 2