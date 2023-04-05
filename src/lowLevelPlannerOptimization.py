import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import pypoman
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.affinity import translate

R = np.diag([1, 1000.0])              # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([1, 1000.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]

DELTA_PHI = np.pi/20                  # orientation deviation from the reference trajectory

def lowLevelPlannerOptimization(scenario, planning_problem, param, plan, vel, space_all, ref_traj):
    """plan a concrete trajectory for the given high-level plan via optimization"""

    # compute orientation of the car along the reference trajectory
    orientation = orientation_ref_traj(ref_traj, param)

    # compute state constraints
    state_con = state_constraints(space_all, orientation, ref_traj, param)

    # plan the trajectory via optimization
    x, u = optimal_control_problem(ref_traj, state_con, vel, param)

    return x, u

def orientation_ref_traj(ref_traj, param):
    """compute the orientation of the car along the reference trajectory"""

    orientation = [param['orientation']]

    for i in range(ref_traj.shape[1]-1):
        diff = ref_traj[:, i+1] - ref_traj[:, i]
        orientation.append(np.arctan2(diff[1], diff[0]))

    return orientation

def state_constraints(space, orientation, ref_traj, param):
    """compute state constraints for the car"""

    radius = []

    # loop over all time steps
    for i in range(1, len(space)):

        # get shape of the car
        pgon = polygon_car(orientation[i], param)

        # subtract shape of the car from the free space
        pgon = minkowski_difference(space[i], pgon)

        # get radius of largest circle contained in the free space
        p = Point(ref_traj[0, i], ref_traj[1, i])

        if isinstance(pgon, Polygon):
            dist = pgon.exterior.distance(p)
        else:
            for pgon in list(pgon.geoms):
                if pgon.contains(p):
                    dist = pgon.exterior.distance(p)
                    break

        radius.append(dist)

        # debug
        """if dist < 0.1:
            plt.plot(*space[i].exterior.xy, 'r')
            plt.plot(*pgon.exterior.xy, 'b')
            phi = np.linspace(0, 2*np.pi, 1000)
            x = ref_traj[0, i] + np.cos(phi)*dist
            y = ref_traj[1, i] + np.sin(phi)*dist
            plt.plot(x, y, 'g')
            plt.axis('equal')
            plt.show()"""

    return radius

def polygon_car(orientation, param):
    """get a polygon that encloses the car for a given range of orientations"""

    # compute template directions
    dirs = [np.array([np.cos(orientation + DELTA_PHI), np.sin(orientation + DELTA_PHI)])]
    dirs.append(-dirs[-1])
    dirs.append(np.array([dirs[-1][1], -dirs[-1][0]]))
    dirs.append(-dirs[-1])

    dirs.append(np.array([np.cos(orientation - DELTA_PHI), np.sin(orientation - DELTA_PHI)]))
    dirs.append(-dirs[-1])
    dirs.append(np.array([dirs[-1][1], -dirs[-1][0]]))
    dirs.append(-dirs[-1])

    dirs.append(np.array([1, 0]))
    dirs.append(np.array([-1, 0]))
    dirs.append(np.array([0, 1]))
    dirs.append(np.array([0, -1]))

    # initialization
    A = np.zeros((2, len(dirs)))
    b = np.zeros((len(dirs), 1))

    phi = Interval(orientation - DELTA_PHI, orientation + DELTA_PHI)
    cosine = phi.cos()
    sine = phi.sin()
    l = Interval(-param['length']/2, param['length']/2)
    w = Interval(-param['width']/2, param['width']/2)

    # loop over all template directions
    for i in range(len(dirs)):
        d = dirs[i]
        tmp = (cosine * d[0] + sine * d[1]) * l + (sine * (-1) * d[0] + cosine * d[1]) * w
        b[i] = tmp.u
        A[:, i] = d

    # convert template polyhedron to polygon
    v = pypoman.compute_polytope_vertices(np.transpose(A), b)

    return Polygon([[p[0], p[1]] for p in v]).convex_hull

def minkowski_difference(pgon1, pgon2):
    """Minkowski difference of two polygons"""

    pgon = pgon1

    # get vertices of the first polygon
    x, y = pgon1.exterior.coords.xy
    V = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)

    # loop over all segments of the first polygon
    for i in range(V.shape[1] - 1):
        tmp1 = translate(pgon2, V[0, i], V[1, i])
        tmp2 = translate(pgon2, V[0, i + 1], V[1, i + 1])
        pgon = pgon.difference(tmp1.union(tmp2).convex_hull)

    return pgon

def optimal_control_problem(ref_traj, state_con, vel, param):
    """solve an optimal control problem to obtain a concrete trajectory"""

    # get vehicle model
    f, nx, nu = vehicle_model(param)

    # initialize optimizer
    opti = casadi.Opti()

    # initialize variables
    x = opti.variable(nx, len(vel))
    u = opti.variable(nu, len(vel)-1)

    # construct initial state
    x0 = np.array([param['x0'][0], param['x0'][1], param['v_init'], param['orientation']])

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

    # constraint (trajectory has to be inside the safe driving corridor)
    for i in range(len(vel)):

        if i > 1:
            opti.subject_to((ref_traj[0, i] - x[0, i])**2 + (ref_traj[1, i] - x[1, i])**2 <= state_con[i-1]**2)

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

class Interval:
    """class for interval arithmetic"""

    def __init__(self, l, u):
        """class constructor"""

        self.l = l
        self.u = u

    def __mul__(self, factor):
        """multiplication of two intervals"""

        if isinstance(factor, Interval):
            tmp = np.array([self.l * factor.l, self.l * factor.u, self.u * factor.l, self.u * factor.u])
            l = min(tmp)
            u = max(tmp)
        elif factor > 0:
            l = self.l * factor
            u = self.u * factor
        else:
            l = self.u * factor
            u = self.l * factor

        return Interval(l, u)

    def __add__(self, summand):
        """Minkowski addition of two intervals"""

        if isinstance(summand, Interval):
            l = self.l + summand.l
            u = self.u + summand.u
        else:
            l = self.l + summand
            u = self.u + summand

        return Interval(l, u)

    def cos(self):
        """cosine function"""

        if self.u - self.l > 2*np.pi:
            l = -1
            u = 1
        else:
            inf = np.mod(self.l, 2*np.pi)
            sup = np.mod(self.u, 2*np.pi)
            if inf <= np.pi:
                if sup < inf:
                    l = -1
                    u = 1
                elif sup <= np.pi:
                    l = np.cos(sup)
                    u = np.cos(inf)
                else:
                    l = -1
                    u = max(np.cos(inf), np.cos(sup))
            else:
                if sup <= pi:
                    l = min(np.cos(inf), np.cos(sup))
                    u = 1
                elif sup < inf:
                    l = -1
                    u = 1
                else:
                    l = np.cos(inf)
                    u = np.cos(sup)

        return Interval(l, u)

    def sin(self):
        """sine function"""

        tmp = self * (-1) + np.pi/2

        return tmp.cos()