import sys
import os
from models.MIP.MIP_Gurobi import MIP as MIP_Gurobi
from models.MIP.MIP_PulP import MIP as MIP_PulP
from models.SMT.SMT_model import SMT
from models.CP.CP_model import CP
from models.SAT.SAT_model import SAT
import multiprocessing as mp

# Main function
def run_model(argv):
    # Se non vengono passati argomenti, esegue tutte le istanze per tutti gli approcci.
    if len(argv) == 1:
        instance_numbers = get_instance_numbers()
        for approach in ['SAT', 'CP', 'SMT', 'MIP']:
            # Se l'approccio è SAT, imposta il metodo spawn una volta
            if approach == "SAT":
                try:
                    mp.set_start_method("spawn")
                except RuntimeError:
                    pass
            for n in instance_numbers:
                run_approach(approach, n)
        return

    # Se viene passato un solo argomento oltre il nome dello script:
    # main.py <approach>   => esegue tutte le istanze per l'approccio specificato
    if len(argv) == 2:
        approach = argv[1]
        instance_numbers = get_instance_numbers()
        if approach == 'SAT':
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                pass
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
        return

    # Se vengono passati due argomenti oltre il nome dello script:
    # main.py <approach> <instance_number>  => esegue l'istanza specificata per l'approccio indicato
    if len(argv) >= 3:
        approach = argv[1]
        instance_number = argv[2]
        if approach == 'SAT':
            try:
                mp.set_start_method("spawn")
            except RuntimeError:
                pass
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
                sb_bool = chuffed_bool or (input("Use Symmetry Breaking constraints? (y/n): ").strip().lower() == 'y')
                CP(instance_number, sb_bool, chuffed_bool)
            case 'SAT':
                sb_bool = input("Use Symmetry Breaking constraints? (y/n): ").strip().lower() == 'y'
                search_type = input("Use Binary Search? (y/n) [if 'n', Branch and Bound will be used]: ").strip().lower() == 'y'
                SAT(instance_number, sb_bool, search_method="binary" if search_type else "branch_and_bound")
            case _:
                print('Invalid parameters')

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

def get_instance_numbers():
    instance_numbers = []
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of main.py
    DATA_PATH = os.path.join(BASE_DIR, "instances", "dat_instances")  # Safe path
    for filename in os.listdir(DATA_PATH):
        inst_num = filename.split('.')[0][-2:]
        instance_numbers.append(inst_num)
    instance_numbers.sort()
    return instance_numbers

if __name__ == "__main__":
    run_model(sys.argv)
