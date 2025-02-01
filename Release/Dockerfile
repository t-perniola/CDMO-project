# Use an official Python runtime as a parent image
FROM python:3.11

# Install bash and coreutils
RUN apt-get update && apt-get install -y bash coreutils

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies for all models
RUN pip install gurobipy z3-solver minizinc pulp numpy

# Make run_model.sh executable
RUN chmod +x /app/run_model.sh

# Define the command to run when the container starts
ENTRYPOINT ["bash", "/app/run_model.sh"]
