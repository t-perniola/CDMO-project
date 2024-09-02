'''
main.py <approach> <instance_number> if you want to execute a specific instance
main.py <approach> if you want to execute all the instances
main.py if you want to execute all the instances over all the approaches
'''

import sys
import os
from MIP_PulP import MIP 
from SMT import SMT
from CP import CP
from SAT import SAT

def run_approach(approach, number):
    match approach:
        case 'MIP':
            MIP(number)
        case 'SMT':
            SMT(number)
        case 'CP':
            CP(number)
        case 'SAT':
            SAT(number)
        case _:
            print('Invalid parameters')

def run_model(argv):
    approach = ''

    if len(argv) > 1:
        approach = argv[1]

    run_all_instances = False
    if len(argv) == 2:
        run_all_instances = True
    else:
        instance_number = argv[2]

    instance_numbers = []
    for filename in os.listdir('Instances'):
        inst_num = filename.split('.')[0][-2:]
        instance_numbers.append(inst_num)
    
    if not run_all_instances and len(argv) > 1:
        match approach:
            case 'MIP':
                MIP(instance_number)
            case 'SMT':
                SMT(instance_number)
            case 'CP':
                CP(instance_number)
            case 'SAT':
                SAT(instance_number)
            case _:
                print('Invalid parameters')

    elif run_all_instances:
        match approach:
            case 'MIP':
                for n in instance_numbers:
                    MIP(n)
            case 'SMT':
                for n in instance_numbers:
                    SMT(n)
            case 'CP':
                for n in instance_numbers:
                    CP(n)
            case 'SAT':
                for n in instance_numbers:
                    SAT(n)
            case _:
                print('Invalid parameters')

    else:
        for approach in ['CP', 'SMT', 'MIP']:
            for n in instance_numbers:
                run_approach(approach, n)

if __name__ == "__main__":
    run_model(sys.argv)

