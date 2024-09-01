#!/bin/bash

# Check if parameters are passed
if [ "$#" -eq 0 ]; then
    echo "Running all models on all instances..."
    python main.py
elif [ "$#" -eq 1 ]; then
    echo "Running model $1 on all instances..."
    python main.py "$1"
else
    echo "Running model $1 on instance $2..."
    python main.py "$1" "$2"
fi
