import numpy as np

def vehicleParameter():
    """parameter for the car"""

    param = {}
    param['length'] = 4.3                   # length of the car
    param['width'] = 1.7                    # width of the car
    param['wheelbase'] = 2.3                # length of the wheelbase
    param['a_max'] = 9                      # maximum acceleration
    param['s_max'] = 1                      # maximum steering angle

    return param