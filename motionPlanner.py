from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.affinity import affine_transform
from shapely.affinity import translate
from copy import deepcopy
from casadi import *

from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams

SCENARIO = 'ZAM_Zip-1_19_T-1.xml'

V_MAX = 100
A_MAX = 9
S_MAX = np.deg2rad(24.0)
dt = 0.1
L = 4.3
W = 1.7
WB = 2.4

R = np.diag([0.01, 100.0])              # input cost matrix, penalty for inputs - [accel, steer]
RD = np.diag([0.01, 100.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]

def motionPlanner(file):
    """plan a safe motion for the given CommonRoad scenario"""

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(file).open()

    # high-level planner: decides on which lanelets to be at which points in time
    plan, space, space_xy = highLevelPlanner(scenario, planning_problem)

    # low-level planner: plans a concrete trajectory for the high-level plan
    x, u = lowLevelPlanner(planning_problem, plan, space, space_xy)

    return x, u

def highLevelPlanner(scenario, planning_problem):
    """decide on which lanelets to be at all points in time"""

    # extract goal region and initial state
    planning_problem = list(planning_problem.planning_problem_dict.values())[0]
    goal = planning_problem.goal
    goal_lanelet = list(goal.lanelets_of_goal_position.values())[0][0]
    final_time = goal.state_list[0].time_step.end

    # create dictionary that converts from lanelet ID to index in the list
    lanelets = scenario.lanelet_network.lanelets
    id = [l.lanelet_id for l in lanelets]
    id2index = dict(zip(id, [i for i in range(0, len(id))]))

    # compute costs (= number of lane changes to reach goal) for each lanelet
    cost_lane = cost_lanelets(lanelets, goal_lanelet, id2index)

    # compute free space on each lanelet at different times
    free_space = free_space_lanelet(lanelets, scenario.obstacles, final_time + 1)

    # determine the high-level plan using the A*-search algorithm
    plan, space = a_star_search(free_space, cost_lane, planning_problem, lanelets, id2index)

    # shrink space by computing reachable set backward in time starting from final set
    space = reduce_space(space, plan, lanelets, id2index)

    # transform space from lanelet coordinate system to global coordinate system
    space_xy = lanelet2global(space, plan, lanelets, id2index)

    return plan, space, space_xy


def cost_lanelets(lanelets, goal_id, id2index):
    """assign a cost to each lanelet that is based on the number of required lane changes to reach goal set"""

    # initialize costs
    cost = [np.inf for i in range(0, len(lanelets))]
    cost[id2index[goal_id]] = 0

    # initialize queue with goal lanelet
    queue = []
    queue.append(goal_id)

    # loop until costs for all lanelets have been assigned
    while len(queue) > 0:

        q = queue.pop(0)
        lanelet = lanelets[id2index[q]]

        for s in lanelet.predecessor:
            if cost[id2index[s]] == np.inf:
                cost[id2index[s]] = cost[id2index[q]]
                queue.append(s)

        if not lanelet.adj_left is None and lanelet.adj_left_same_direction:
            if cost[id2index[lanelet.adj_left]] == np.inf:
                cost[id2index[lanelet.adj_left]] = cost[id2index[q]] + 1
                queue.append(lanelet.adj_left)

        if not lanelet.adj_right is None and lanelet.adj_right_same_direction:
            if cost[id2index[lanelet.adj_right]] == np.inf:
                cost[id2index[lanelet.adj_right]] = cost[id2index[q]] + 1
                queue.append(lanelet.adj_right)

    return cost

def free_space_lanelet(lanelets, obstacles, length):
    """compute free space on each lanelet for all time steps"""

    free_space_all = []

    # loop over all lanelets
    for l in lanelets:

        occupied_space = [[] for i in range(0, length)]

        # loop over all obstacles
        for obs in obstacles:

            # loop over all time steps
            for o in obs.prediction.occupancy_set:

                # check if dynamic obstacles occupancy set intersects the lanelet
                if l.polygon.shapely_object.intersects(o.shape.shapely_object):

                    # project occupancy set onto the lanelet center line to obtain occupied longitudinal space
                    dist_min, dist_max = projection_lanelet_centerline(l, o.shape.shapely_object)
                    occupied_space[o.time_step-1].append((dist_min-0.5*L, dist_max+0.5*L))

        # unite occupied spaces that belong to the same time step to obtain free space
        free_space = []

        for o in occupied_space:

            if len(o) > 0:

                o.sort(key=lambda i: i[0])
                lower = [i[0] for i in o]
                upper = [i[1] for i in o]

                space = []

                if lower[0] > l.distance[0]:
                    pgon = interval2polygon([l.distance[0], -V_MAX], [lower[0], V_MAX])
                    space.append(pgon)

                cnt = 0

                while cnt < len(o)-1:

                    ind = cnt

                    for i in range(cnt, len(o)):
                        if lower[i] < upper[cnt]:
                            ind = i

                    pgon = interval2polygon([upper[ind], -V_MAX], [lower[ind+1], V_MAX])
                    space.append(pgon)
                    cnt = ind + 1

                if max(upper) < l.distance[len(l.distance)-1]:
                    pgon = interval2polygon([max(upper), -V_MAX], [l.distance[len(l.distance)-1], V_MAX])
                    space.append(pgon)

            else:
                pgon = interval2polygon([l.distance[0], -V_MAX], [l.distance[len(l.distance)-1], V_MAX])
                space = [pgon]

            free_space.append(space)

        free_space_all.append(free_space)

    return free_space_all

def projection_lanelet_centerline(lanelet, pgon):
    """project a polygon to the center line of the lanelet to determine the occupied space"""

    # intersect polygon with lanelet
    o_int = lanelet.polygon.shapely_object.intersection(pgon)

    vx, vy = o_int.exterior.coords.xy
    V = np.stack((vx, vy))

    # initialize minimum and maximum range of the projection
    dist_max = -np.inf
    dist_min = np.inf

    # loop over all centerline segments
    for i in range(0, len(lanelet.distance) - 1):

        # compute normalized vector representing the direction of the current segment
        diff = lanelet.center_vertices[i + 1, :] - lanelet.center_vertices[i, :]
        diff = np.expand_dims(diff / np.linalg.norm(diff), axis=0)

        # project the vertices of the polygon onto the centerline
        V_ = diff @ (V - np.transpose(lanelet.center_vertices[[i], :]))

        # update ranges for the projection
        dist_max = max(dist_max, max(V_[0]) + lanelet.distance[i])
        dist_min = min(dist_min, min(V_[0]) + lanelet.distance[i])

    return dist_min, dist_max

def is_intersecting(range1, range2):
    """check if two ranges intersect"""

    if range1[0] <= range2[0]:
        if range2[0] <= range1[1]:
            return True
    else:
        if range1[0] <= range2[1]:
            return True

    return False

def interval2polygon(lb, ub):
    """convert an interval given by lower and upper bound to a polygon"""

    return Polygon([(lb[0],lb[1]), (lb[0], ub[1]), (ub[0], ub[1]), (ub[0], lb[1])])

def expand_node(node, lanelet_id, space, lane_changes, expect_lane_changes):
    """add value for the current time step to a node for the search algorithm"""

    # add current lanelet to list of lanelets
    l = deepcopy(node.lanelets)
    l.append(lanelet_id)

    # add reachable space to the list
    s = deepcopy(node.space)
    s.append(space)

    # create resulting node
    return Node(l, s, lane_changes, expect_lane_changes)

def a_star_search(free_space, cost, planning_problem, lanelets, id2index):
    """determine optimal lanelet for each time step using A*-search"""

    # get goal time and set
    goal = planning_problem.goal
    goal_id = list(goal.lanelets_of_goal_position.values())[0][0]
    goal_state = goal.state_list[0]

    goal_space_start, goal_space_end = projection_lanelet_centerline(lanelets[id2index[goal_id]],
                                                                     goal_state.position.shapes[0].shapely_object)
    v = goal_state.velocity
    goal_space = interval2polygon([goal_space_start, v.start], [goal_space_end, v.end])

    # get lanelet and space for the initial point
    x0 = planning_problem.initial_state.position

    for l in lanelets:
        if l.polygon.shapely_object.contains(Point(x0[0], x0[1])):
            x0_id = l.lanelet_id

    pgon = interval2polygon([x0[0]-0.01, x0[1]-0.01], [x0[0]+0.01, x0[1]+0.01])
    x0_space_start, x0_space_end = projection_lanelet_centerline(lanelets[id2index[x0_id]], pgon)

    v = planning_problem.initial_state.velocity
    x0_space = interval2polygon([x0_space_start, v - 0.01], [x0_space_end, v + 0.01])

    # initialize the frontier
    frontier = []
    frontier.append(Node([x0_id], [x0_space], 0, cost[id2index[x0_id]]))

    # loop until a solution has been found
    while len(frontier) > 0:

        # sort the frontier
        frontier.sort(key=lambda i: i.cost)

        # select node with the lowest costs
        node = frontier.pop(0)

        # check if goal set has been reached
        if len(node.lanelets) >= goal_state.time_step.start and node.lanelets[len(node.lanelets)-1] == goal_id \
                and goal_space.intersects(node.space[len(node.space)-1]):
            return node.lanelets, node.space

        # create child nodes
        if len(node.lanelets) <= goal_state.time_step.end:
            children = create_child_nodes(node, free_space, cost, lanelets, id2index)
            frontier.extend(children)


def create_child_nodes(node, free_space, cost, lanelets, id2index):
    """create all possible child nodes for the current node"""

    # initialization
    children = []
    ind = len(node.lanelets)
    lanelet = lanelets[id2index[node.lanelets[ind-1]]]

    # compute reachable set using maximum acceleration and deceleration
    space = node.space[ind-1]

    space1 = affine_transform(space, [1, dt, 0, 1, 0.5 * dt ** 2 * A_MAX, dt * A_MAX])
    space2 = affine_transform(space, [1, dt, 0, 1, -0.5 * dt ** 2 * A_MAX, -dt * A_MAX])

    space = space1.union(space2)
    space = space.convex_hull

    # create children resulting from staying on the same lanelet
    space_lanelet = free_space[id2index[node.lanelets[ind-1]]][ind]

    for sp in space_lanelet:
        if sp.intersects(space):

            space_new = sp.intersection(space)
            lane_changes = node.lane_changes
            expect_lane_changes = cost[id2index[node.lanelets[ind-1]]]

            node_new = expand_node(node, node.lanelets[ind-1], space_new, lane_changes, expect_lane_changes)
            children.append(node_new)

    # create children for moving to a successor lanelet
    space_ = translate(space, -lanelet.distance[len(lanelet.distance)-1], 0)

    for suc in lanelet.successor:
        for sp in free_space[id2index[suc]][ind]:
            if space_.intersects(sp):

                space_new = sp.intersection(space_)
                lane_changes = node.lane_changes
                expect_lane_changes = cost[id2index[suc]]

                node_new = expand_node(node, suc, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the left
    if not lanelet.adj_left is None and lanelet.adj_left_same_direction:

        for space_left in free_space[id2index[lanelet.adj_left]][ind]:
            if space.intersects(space_left):

                space_new = space.intersection(space_left)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_left]]

                node_new = expand_node(node, lanelet.adj_left, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    # create children for lane change to the right
    if not lanelet.adj_right is None and lanelet.adj_right_same_direction:

        for space_right in free_space[id2index[lanelet.adj_right]][ind]:
            if space.intersects(space_right):

                space_new = space.intersection(space_right)
                lane_changes = node.lane_changes + 1
                expect_lane_changes = cost[id2index[lanelet.adj_right]]

                node_new = expand_node(node, lanelet.adj_right, space_new, lane_changes, expect_lane_changes)
                children.append(node_new)

    return children

def reduce_space(space, plan, lanelets, id2index):
    """reduce the drivable space by propagating the goal set backward in time"""

    # loop over all time steps
    for i in range(len(space)-1, 0, -1):

        # propagate reachable set backwards in time
        space1 = affine_transform(space[i], [1, -dt, 0, 1, -0.5 * dt ** 2 * A_MAX, dt * A_MAX])
        space2 = affine_transform(space[i], [1, -dt, 0, 1, 0.5 * dt ** 2 * A_MAX, -dt * A_MAX])
        space_new = space1.union(space2)
        space_new = space_new.convex_hull

        # shift reachable set by lanelet length if the lanelet is changed
        if not plan[i-1] == plan[i] and not lanelets[id2index[plan[i-1]]].adj_left == plan[i] and \
                not lanelets[id2index[plan[i-1]]].adj_right == plan[i]:
            lanelet = lanelets[id2index[plan[i-1]]]
            dist = lanelet.distance[len(lanelet.distance)-1]
            space_new = translate(space_new, dist, 0)

        # intersect with previous rechable set
        space[i - 1] = space_new.intersection(space[i - 1])

    return space

def lanelet2global(space, plan, lanelets, id2index):
    """transform free space from lanelet coordinate system to global coordinate system"""

    space_xy = []

    # loop over all time steps
    for i in range(0, len(space)):

        # initialization
        lanelet = lanelets[id2index[plan[i]]]

        lower = space[i].bounds[0]
        upper = space[i].bounds[2]

        left_vertices = []
        right_vertices = []

        # loop over the single segments of the lanelet
        for j in range(0, len(lanelet.distance)-1):

            intermediate_point = True

            if lower >= lanelet.distance[j] and lower <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d/np.linalg.norm(d) * (lower - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (lower - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

                intermediate_point = False

            if upper >= lanelet.distance[j] and upper <= lanelet.distance[j+1]:

                d = lanelet.left_vertices[j + 1] - lanelet.left_vertices[j]
                p_left = lanelet.left_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                left_vertices.append(Point(p_left[0], p_left[1]))

                d = lanelet.right_vertices[j + 1] - lanelet.right_vertices[j]
                p_right = lanelet.right_vertices[j] + d / np.linalg.norm(d) * (upper - lanelet.distance[j])
                right_vertices.append(Point(p_right[0], p_right[1]))

                break

            if len(left_vertices) > 0 and intermediate_point:

                p_left = lanelet.left_vertices[j]
                left_vertices.append(Point(p_left[0], p_left[1]))

                p_right = lanelet.right_vertices[j]
                right_vertices.append(Point(p_right[0], p_right[1]))

        # construct the resulting polygon in the global coordinate system
        right_vertices.reverse()
        left_vertices.extend(right_vertices)
        space_xy.append(Polygon(left_vertices))

    return space_xy

class Node:
    """class representing a single node for A*-search"""

    def __init__(self, lanelets, space, lane_changes, expect_lane_changes):
        """class constructor"""

        # store object properties
        self.lanelets = lanelets
        self.space = space
        self.lane_changes = lane_changes
        self.expect_lane_changes = expect_lane_changes

        # compute costs
        self.cost = self.cost_function()

    def cost_function(self):
        """cost function for A*-search"""

        return self.lane_changes + self.expect_lane_changes - len(self.lanelets)*dt


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

if __name__ == "__main__":

    x, u = motionPlanner(SCENARIO)

    scenario, planning_problem_set = CommonRoadFileReader(SCENARIO).open()

    # visualization
    plt.figure(figsize=(25, 10))

    for i in range(0, x.shape[1]):
        plt.cla()
        rnd = MPRenderer()

        rnd.draw_params.time_begin = i
        scenario.draw(rnd)
        planning_problem_set.draw(rnd)

        param = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='r')
        r = Rectangle(length=L, width=W, center=np.array([x[0, i], x[1, i]]), orientation=x[3, i])
        r.draw(rnd, param)

        rnd.render()
        plt.pause(0.01)