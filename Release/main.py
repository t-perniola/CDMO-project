'''
main.py <approach> <instance_number> if you want to execute a specific instance
main.py <approach> if you want to execute all the instances
'''

import sys
#from MIP import MIP
from SMT import SMT
from SMT_old import SMT_old

def run_model(argv):
    approach = argv[1]

    run_all_instances = False
    if len(argv) == 2:
        run_all_instances = True
    else:
        instance_number = argv[2]
    
    if not run_all_instances:
        match approach:
            case 'MIP':
                print("MIP")
                #MIP(instance_number)
            case 'SMT':
                sb_bool = check_sb()
                SMT(instance_number, sb_bool=sb_bool)
            case "SMT_old":
                SMT_old(instance_number)
            case _:
                print('Invalid parameters')

# ask the user if he want a symm breaking constraint
def check_sb():
    user_input = input("Add a symmetry breaking constraint? [y/n]: ").strip().lower()
    while user_input not in ["y", "n"]:
        user_input = input("Invalid input. Please enter 'y' for yes or 'n' for no: ").strip().lower()
    sb_bool = user_input == "y"
    return sb_bool

if __name__ == "__main__":
    run_model(sys.argv)

#Example
#run_model('MIP', '01')
