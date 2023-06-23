import numpy as np
import scipy.io
import os
from mat4py import loadmat
from copy import deepcopy
import sys
sys.path.append('./')

from maneuverAutomaton.Controller import *

def loadAROCcontroller(MA, primitives, x0):
    """load the controller constructed with AROC for the given sequence of motion primitives"""

    # get path to directory that stores the controller files
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'AROC', 'automaton', 'controllers')

    # initialization
    controllers = []
    x = x0
    t = 0

    # loop over all motion primitives
    for p in primitives:

        # load controller data from file
        file = os.path.join(path, 'controller' + str(p + 1) + '.mat')
        mat = loadmat(file)

        if len(mat['parallelo']) == 2:
            mat['alpha'] = [mat['alpha']]

        # create controller object
        time = t + np.linspace(0, MA.primitives[p].tFinal, len(mat['parallelo']))
        controllers.append(GeneratorSpaceController(mat['U'], mat['parallelo'], mat['alpha'], time, x))

        # update state and time
        phi = x[3]
        T = scipy.linalg.block_diag(np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]), np.eye(2))

        x = T @ MA.primitives[p].x[:, [-1]] + np.array([[x[0]], [x[1]], [0], [x[3]]])
        x = x[:, 0]
        t = t + MA.primitives[p].tFinal

    # construct overall controller object
    controller = Controller(controllers)

    return controller

def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)