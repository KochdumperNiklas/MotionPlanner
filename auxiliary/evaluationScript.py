import csv
import os

PLANNER = 'HighLevel'   # planner ('HighLevel', 'Automaton', 'AutomatonNaive' or 'Optimization')

def evaluation_matrics(data):
    """compute the performance metrics for the given data"""

    success = 0
    comp_time = 0
    collisions = 0

    for d in data:
        if d[1].replace('.', '', 1).isdigit():
            comp_time = comp_time + float(d[1])
            success = success + 1
            if d[2] == 'collision':
                collisions = collisions + 1

    return success/len(data) * 100, comp_time/success, collisions/success * 100

def print_results(success, comp_time, collisions):
    """print the performance metrics to the console"""

    print('success rate:           ' + "{:10.2f}".format(success) + '%')
    print('computatoin time:       ' + "{:10.2f}".format(comp_time) + 's')
    print('collisions:             ' + "{:10.2f}".format(collisions) + '%')

if __name__ == "__main__":
    """main entry point"""

    # construct path to the file with the data
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    file_path = os.path.join(dir_path, 'computation_time_' + PLANNER + '.csv')

    # read data from .csv file
    data = []

    with open(file_path, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            data.append(lines)

    # performance for all scenarios
    success, comp_time,  collisions = evaluation_matrics(data)

    print(' ')
    print('All scenarios')
    print(' ')
    print_results(success, comp_time, collisions)

    # performance for planning horizon < 4
    data_ = [data[i] for i in range(len(data)) if float(data[i][4]) < 4]

    success, comp_time, collisions = evaluation_matrics(data_)

    print(' ')
    print('Planning horizon < 4 seconds')
    print(' ')
    print_results(success, comp_time, collisions)

    # performance for planning horizon >= 4
    data_ = [data[i] for i in range(len(data)) if float(data[i][4]) >= 4]

    success, comp_time, collisions = evaluation_matrics(data_)

    print(' ')
    print('Planning horizon >= 4 seconds')
    print(' ')
    print_results(success, comp_time, collisions)

    # performance for urban scenarios
    data_ = [data[i] for i in range(len(data)) if 'URBAN' in data[i][3]]

    success, comp_time, collisions = evaluation_matrics(data_)

    print(' ')
    print('Urban scenarios')
    print(' ')
    print_results(success, comp_time, collisions)

    # performance for highway scenarios
    data_ = [data[i] for i in range(len(data)) if 'HIGHWAY' in data[i][3]]

    success, comp_time, collisions = evaluation_matrics(data_)

    print(' ')
    print('Highway scenarios')
    print(' ')
    print_results(success, comp_time, collisions)

    # performance for highway scenarios
    data_ = [data[i] for i in range(len(data)) if 'INTERSECTION' in data[i][3]]

    success, comp_time, collisions = evaluation_matrics(data_)

    print(' ')
    print('Scenarios with intersections')
    print(' ')
    print_results(success, comp_time, collisions)