import xml.etree.ElementTree as ET
import os
import numpy as np
from shapely.geometry import Polygon
import shutil
import time as timer
import pickle
import sys
sys.path.append('./')

from maneuverAutomaton.ManeuverAutomaton import MotionPrimitive
from maneuverAutomaton.ManeuverAutomaton import ManeuverAutomaton

def loadAROCautomaton():
    """load a maneuver automaton create with the AROC toolbox"""

    # check if the maneuver automaton has been changed
    path = os.path.dirname(os.path.abspath(__file__))

    create = False

    if not os.path.isdir(os.path.join(path, 'AROC', 'automaton')):
        create = True
    else:
        if not os.path.isfile(os.path.join(path, 'AROC', 'automaton', 'lastVersion.obj')):
            create = True
        else:
            filehandler = open(os.path.join(path, 'AROC', 'automaton', 'lastVersion.obj'), 'rb')
            data = pickle.load(filehandler)
            time_modified = timer.ctime(os.path.getmtime(os.path.join(path, 'AROC', 'automaton.zip')))
            if not time_modified == data['time']:
                create = True

    # extract the files if the automaton has been changed
    if create == True:
        if os.path.isdir(os.path.join(path, 'AROC', 'automaton')):
            shutil.rmtree(os.path.join(path, 'AROC', 'automaton'))
        shutil.unpack_archive(os.path.join(path, 'AROC', 'automaton.zip'), os.path.join(path, 'AROC', 'automaton'))
        time_modified = timer.ctime(os.path.getmtime(os.path.join(path, 'AROC', 'automaton.zip')))
        filehandler = open(os.path.join(path, 'AROC', 'automaton', 'lastVersion.obj'), 'wb')
        pickle.dump({'time': time_modified}, filehandler)

    # load the XML file
    tree = ET.parse(os.path.join(path, 'AROC', 'automaton', 'automaton.xml'))

    # get the root element of the XML tree
    root = tree.getroot()
    primitives = []

    # iterate over all the child elements of the root (= single motion primitives)
    for child in root:

        # open .xml file storing the motion primitive
        tree_prim = ET.parse(os.path.join(path, 'AROC', 'automaton', 'primitives', child.text))
        primitive = tree_prim.getroot()

        for subchild in primitive:

            # get successor motion primitives
            if subchild.tag == 'successors':

                successors = []

                for id in subchild:
                    successors.append(int(id.text) - 1)

            # get reference trajectory
            elif subchild.tag == 'trajectory':

                for element in subchild:

                    if element.tag == 'states':

                        x = np.zeros((4, len(element)))
                        t = np.zeros((len(element)))
                        cnt = 0

                        for state in element:

                            for entry in state:
                                if entry.tag == 'time':
                                    t[cnt] = float(entry.text)
                                elif entry.tag == 'x':
                                    x[0, cnt] = float(entry.text)
                                elif entry.tag == 'y':
                                    x[1, cnt] = float(entry.text)
                                elif entry.tag == 'velocity':
                                    x[2, cnt] = float(entry.text)
                                elif entry.tag == 'orientation':
                                    x[3, cnt] = float(entry.text)

                            cnt = cnt + 1

                    elif element.tag == 'inputs':

                        u = np.zeros((2, len(element)))
                        cnt = 0

                        for input in element:

                            for entry in input:
                                if entry.tag == 'acceleration':
                                    u[0, cnt] = float(entry.text)
                                elif entry.tag == 'steer':
                                    u[1, cnt] = float(entry.text)

                            cnt = cnt + 1

            # get occupancy set
            elif subchild.tag == 'occupancySet':

                occSet = []

                for occupancy in subchild:
                    for entry in occupancy:

                        if entry.tag == 'time':

                            for time in entry:
                                if time.tag == 'intervalStart':
                                    timeStart = float(time.text)
                                elif time.tag == 'intervalEnd':
                                    timeEnd = float(time.text)

                        elif entry.tag == 'set':

                            V = np.zeros((2, len(entry)))
                            cnt = 0

                            for point in entry:
                                for dim in point:
                                    ind = int(dim.attrib['dimension'])
                                    V[ind-1, cnt] = float(dim.text)
                                cnt = cnt + 1

                    pgon = Polygon(np.transpose(V))
                    occSet.append({'space': pgon, 'starttime': timeStart, 'endtime': timeEnd})

        # construct motion primitive object
        primitives.append(MotionPrimitive(x, u, t[-1], successors, occSet))

    # create maneuver automaton
    MA = ManeuverAutomaton(primitives)

    return MA