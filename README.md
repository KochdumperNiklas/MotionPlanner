# Description

This repository contains the Python implementation of the reachability-based decision module for autonomous road vehicles presented in the paper **Efficient Reachability-Based Decision Making for Autonomous Driving**.

# Installation
We recommend to create a Anaconda environment for running the code:

  `conda create --name motionPlanner python=3.8`
  
  `conda activate motionPlanner`

The required python packages can be installed with pip:

  `pip install -r requirements.txt`
  
Finally, the motion planner can be executed by running the main file:

  `python3 main.py "ZAM_Zip-1_19_T-1.xml" "Optimization"`

where the input arguments are the name of the CommonRoad scenario that should be solved and the name of the motion planner that is used (**"Automaton"** or **"Optimization"**). Vehicle parameters like for example length, width, and maximum acceleration can be specified in the file */vehicle/vehicleParameter.py*.

# Interface to CommonRoad

Our motion planners require the traffic scenarios that should be solved to be represented in the [CommonRoad format](https://mediatum.ub.tum.de/doc/1379638/1379638.pdf). Please note that many other common data formats used for autonomous driving such as OpenStreetMap, OpenDRIVE, and SUMO can be automatically converted to CommonRoad format using the [CommonRoad Scenario Designer](https://mediatum.ub.tum.de/doc/1624607/document.pdf). In this repository we provide 100 exemplary CommonRoad traffic scenarios in the directory */scenarios*. Many additional scenarios are available on the [CommonRoad website](https://commonroad.in.tum.de/). Please note that our implementation currently does not support interactive scenarios. The 2000 scenarios we used for the numerical evaluation in our paper are provided in our [CodeOcean repeatability package](https://codeocean.com/capsule/1869710/tree) under */data/CommonRoadScenarios.zip*.  

# Interface to CARLA

The repository also provides an interface for running our motion planners in the [CARLA simulator](https://carla.org/). For this, first install and start the CARLA simulator (we tested our code with CARLA version 0.9.13) on your computer. After the simulator is running, simply execute the file *mainCARLA.py* from our repository:

  `python3 mainCARLA.py`

The script will plan a route from a randomly selected start point to a randomly selected destination on the map. The task of the motion planner is then to follow this route. Currently, we support CARLA maps **Town01** and **Town03**. 

# Interface to AROC

The **"Automaton"** motion planner we provide in this repository uses a maneuver automaton that we constructed using the MATLAB toolbox [AROC](https://aroc.in.tum.de). To create a maneuver automaton for a different vehicle type (with different length, width, wheelbase, etc.) install the AROC toolbox and execute the script *maneuverAutomaton/AROC/createManeuverAutomaton.m* from our repository in MATLAB. Since the maneuver automaton is quite large and MATLAB does not clear the memory in-between, your computer may run out of memory during the computation. We therefore recommend to use the script *maneuverAutomaton/AROC/runScript.sh* instead, which periodically restarts MATLAB to avoid this issue. Our code also supports loading custom maneuver automata created with AROC. For this, simply replace the file *maneuverAutomaton/AROC/automaton.zip* in the repository with your exported custom maneuver automaton you created with AROC. 

