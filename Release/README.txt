The instructions should concisely explain how to reproduce the result of a single model on a specific instance, 
without modifying the source code of your project; for instance, you could provide a script that takes as input 
arguments the instance number and/or the approach chosen. Moreover, you should also provide a way to run all the 
models on all the possible instances automatically, aiming to generate all the feasible results with a single 
operation. The choices for the instructions format and the implementation of the Docker environment are completely 
up to you.

1. Build the Docker image using the command "docker build -t <image_name> ."
2. Run the python script. There are two different ways to run the script:
    2.1 By using the command "docker run --rm <image_name> <approach> <instance_number>"
    2.2 By using the command "docker run --rm <image_name> <approach>"
    2.3 By using the command "docker run --rm <image_name>"

   Using (2.1) you run the program only on <instance_number> using the approach <approach>;
   Using (2.2) you run the program over all the instances using the approach <approach>;
   Using (2.3) you run the program over all the instances using all the approaches;

   <approach> takes the values 'CP', 'SAT', 'SMT' or 'MIP'
   <instance_number> must be in the format '0x' for the instances having number lower than 10, and 'x' for 
   the instances having number up to 21.

   An example of correct command is: "docker run --rm <image_name> MIP 02" or "docker run --rm <image_name> CP 19"

Since the output json files are stored in the image, if you want bring them in the local machine you have to first 
mount a volume using the command "docker volume create <volume_name>" and then you have to run the file using the
following command "docker run --rm -v <volume_name>:/res <image_name> <args>"

Note that, since the MIP model is very huge, you may need an active academic licence in order to use the library.
At this purpose we put in the res/MIP folder the json I generated with my licence, just for check.

In order to simply check the output in the volume, you may use Docker desktop.
