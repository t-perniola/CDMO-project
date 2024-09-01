# Use an official Python runtime as a parent image
FROM python:3.11

ADD main.py .

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies for all models
RUN pip install gurobipy z3-solver minizinc

# Make main.sh executable
RUN chmod +x /app/run_model.sh

# Define the command to run when the container starts
#CMD ["./run_model.sh"]
ENTRYPOINT ["/app/run_model.sh"]
