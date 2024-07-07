'''
main.py <approach> <instance_number> if you want to execute a specific instance
main.py <approach> if you want to execute all the instances
'''

import sys
from MIP import MIP 

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
                MIP(instance_number)
            case _:
                print('Invalid parameters')


if __name__ == "__main__":
    run_model(sys.argv)

#Example
run_model('MIP', '01')
