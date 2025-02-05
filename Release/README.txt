TO RUN THE MODELS ON THE INSTANCES
----------------------------------

0. The current directory should be the one where the "Release" folder is.

1. Build the Docker image using the command "docker build -t <image_name> ."

2. Start the Docker container using the command "docker run -it -d --name <container-name> <image-name> <approach> <instance_number>".
   (This just starts the container; the approach and instance number will not be executed yet.)

3. Run the python script such that the results will be stored always in the same container. There are three different ways to run the script:
    3.1 By using the command "docker exec -it <container_name> /app/run_model.sh <approach> <instance_number>"
    3.2 By using the command "docker exec -it <container_name> /app/run_model.sh <approach>"
    3.3 By using the command "docker exec -it <container_name> /app/run_model.sh"

   Using (3.1) you run the program only on <instance_number> using the approach <approach>;
   Using (3.2) you run the program over all the instances using the approach <approach>;
   Using (3.3) you run the program over all the instances using all the approaches;

   <approach> takes the values 'CP', 'SAT', 'SMT' or 'MIP'
   <instance_number> must be in the format '0x' for the instances having number lower than 10, and 'x' for the instances having number up to 21.

   An example of correct command is: "docker exec -it <container_name> /app/run_model.sh CP 05" or "docker exec -it <container_name> /app/run_model.sh SMT"

By default, the output JSON files are stored within the container. If you want to copy these files to your local machine, you must first create a Docker volume: "docker volume create <volume_name>" and then you have to run the file using the following command: "docker run -it --rm -v $(pwd)/res:/app/res <image-name> <approach> <instance_number>".
This command will run the model, and the output will be stored in the res/ directory on your local machine.

Note that, since the MIP model is very huge, you may need an active academic licence in order to use the library.
At this purpose we put in the res/MIP folder the json we generated with our licence, just for check.

In order to simply check the output in the volume, you may use Docker desktop.

.......................................................................................................................................................................................

TO CHECK JSON SOL
-----------------

0. The current directory should be the one where the "Release" folder is.

1. In order to check the solutions into the json files, the folder path where the instances are is the following: "instances/dat_instances"
   - e.g. on Windows a correct command is "python check_solution.py instances/dat_instances res"