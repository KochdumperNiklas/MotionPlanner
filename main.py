from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.geometry.shape import Rectangle
from commonroad.visualization.draw_params import ShapeParams
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from vehicle.vehicleParameter import vehicleParameter
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton
from maneuverAutomaton.loadAROCautomaton import loadAROCautomaton
from src.highLevelPlanner import highLevelPlanner
from src.lowLevelPlannerManeuverAutomaton import lowLevelPlannerManeuverAutomaton
from src.lowLevelPlannerOptimization import lowLevelPlannerOptimization


SCENARIO = "ZAM_Zip-1_19_T-1.xml"       # CommonRoad scenario that should be solved
PLANNER = 'Optimization'                # motion planner ('Automaton' or 'Optimization')


if __name__ == "__main__":
    """main entry point"""

    # parse input arguments to script
    if len(sys.argv) > 1:
        SCENARIO = sys.argv[1]

    if len(sys.argv) > 2:
        PLANNER = bool(sys.argv[2])

    # load parameter for the car
    param = vehicleParameter()

    # load the CommonRoad scenario
    scenario, planning_problem = CommonRoadFileReader(os.path.join('scenarios', SCENARIO)).open(lanelet_assignment=True)

    # run the decision module
    lanelets, space, ref_traj = highLevelPlanner(scenario, planning_problem, param, desired_velocity='init')

    # run the low-level trajectory planner
    if PLANNER == "Optimization":
        x, u, controller = lowLevelPlannerOptimization(scenario, planning_problem, param, space, ref_traj)
    else:
        MA = loadAROCautomaton()
        x, u, controller = lowLevelPlannerManeuverAutomaton(scenario, planning_problem, param, space, ref_traj, MA)

    if x is None:
        raise Exception('Low-level trajectory planner failed!')

    # visualization
    plt.figure(figsize=(25, 10))

    for i in range(0, x.shape[1]):
        plt.cla()
        rnd = MPRenderer()

        rnd.draw_params.time_begin = i
        scenario.draw(rnd)
        planning_problem.draw(rnd)

        settings = ShapeParams(opacity=1, edgecolor="k", linewidth=0.0, zorder=17, facecolor='#d95558')
        r = Rectangle(length=param['length'], width=param['width'], center=np.array([x[0, i], x[1, i]]),
                      orientation=np.mod(x[3, i], 2*np.pi))
        r.draw(rnd, settings)

        rnd.render()
        plt.xlim([min(x[0, :]) - 20, max(x[0, :]) + 20])
        plt.ylim([min(x[1, :]) - 20, max(x[1, :]) + 20])
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.pause(0.1)
