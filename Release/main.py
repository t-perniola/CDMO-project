from MIP import MIP 

def run_model(approach, instance_number, run_all_instances=False):
    if not run_all_instances:
        match approach:
            case 'MIP':
                MIP(instance_number)
            case _:
                print('Invalid parameters')

#Example
run_model('MIP', '09')
