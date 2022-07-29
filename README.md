# Project Software Engineering Lab 3: Cloth Folding using RL in Unity

This project mainly consists of two parts:

- `Unity_Simulation`: This directory contains the Unity project implementing our simulation environment.
- `RL`: The code used to train a reinforcement learning agent using Stable Baselines 3 and Unity ML Agents on our Unity environment.

## Makefile

This project also contains a Makefile for quickly executing some more complicated tasks, the following make targets are present:

- `make executable`: Uses Unity in batch mode to build an executable of the project.
- `make clean`: Cleanup the executable built by Unity
- `make rl-bootstrap`: Bootstraps the RL environment so that it is ready to execute the project, i.e. install all necessary dependencies using pip. Be sure to create a virtual environment first in `RL/` as specified in `RL/README.md`.

## Documentation

All source code of this project contains documentation where relevant for functions, modules and classes. The `RL` and `Unity_Simulation` projects further detail how to generate a documentation website using the provided functionality.
