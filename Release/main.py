'''
main.py <approach> <instance_number> if you want to execute a specific instance
main.py <approach> if you want to execute all the instances
main.py if you want to execute all the instances over all the approaches
'''

import sys
import os
from models.MIP.MIP_Gurobi import MIP as MIP_Gurobi
from models.MIP.MIP_PulP import MIP as MIP_PulP
from models.SMT.SMT_model import SMT
from models.CP.CP_model import CP
from models.SAT.SAT_model import SAT

# Main function
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of main.py
    DATA_PATH = os.path.join(BASE_DIR, "instances", "dat_instances")  # Safe path
    for filename in os.listdir(DATA_PATH):
        inst_num = filename.split('.')[0][-2:]
        instance_numbers.append(inst_num)
    
    # execute the specified instance with the specified approach
    if not run_all_instances and len(argv) > 1:
        match approach:
            case 'MIP':
                gurobi_bool = input("Use Gurobi as solver? (y/n): ").strip().lower() == 'y'
                MIP_Gurobi(instance_number) if gurobi_bool else MIP_PulP(instance_number)
            case 'SMT':
                sb_bool = input("Use Symmetry Breaking constraints? (y/n): ").strip().lower() == 'y'
                bin_search_bool = input("Use Binary Search? (y/n) [if 'n', Branch and Bound will be used]: ").strip().lower() == 'y'
                SMT(instance_number, bin_search_bool, sb_bool)
            case 'CP':
                chuffed_bool = input("Use Chuffed as solver? (y/n) [if 'n', Gecode will be used]: ").strip().lower() == 'y'
                sb_bool = chuffed_bool or input("Use Symmetry Breaking constraints? (y/n): ").strip().lower() == 'y'
                CP(instance_number, sb_bool, chuffed_bool)
            case 'SAT':
                sb_bool = input("Use Symmetry Breaking constraints? (y/n): ").strip().lower() == 'y'
                search_type = input("Use Binary Search? (y/n) [if 'n', Branch and Bound will be used]: ").strip().lower == 'y'
                SAT(instance_number, sb_bool, search_method="binary" if search_type else "branch_and_bound")
            case _:
                print('Invalid parameters')

    # execute all the instances for the specified approach
    elif run_all_instances:
        match approach:
            case 'MIP':
                for n in instance_numbers:
                    MIP_PulP(n)
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

    else: # execute all approaches for all instances
        for approach in ['CP', 'SMT', 'MIP']:
            for n in instance_numbers:
                run_approach(approach, n)

def run_approach(approach, number):
    match approach:
        case 'MIP':
            MIP_PulP(number)
        case 'SMT':
            SMT(number)
        case 'CP':
            CP(number)
        case 'SAT':
            SAT(number)
        case _:
            print('Invalid parameters')

if __name__ == "__main__":
    run_model(sys.argv)
