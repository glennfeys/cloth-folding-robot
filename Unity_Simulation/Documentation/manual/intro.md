# Software Engineering Lab 3: Unity Simulation

Welcome to our Unity project used in simulating the RL environment our agent will be trained on in order to learn to fold a piece of fabric. The project itself consists of a number of components, separated using different namespaces. For the interaction between Unity and our Python reinforcement learning environment, we're using the [Unity ML Agents framework (Release 12)](https://github.com/Unity-Technologies/ml-agents).

## Global project structure

The project is separated globally in two main namespaces:

- [SoftBody](/api/SoftBody.html) which implements the physics of our piece of textile, either using a CPU based integrator or using a GPU based implementation (which was spun-out into [SoftBody.Gpu](/api/SoftBody.Gpu.html)).
- [RigidBody](/api/RigidBody.html) which implements the ML Agents logic driving our Baxter robot's behaviour in Unity and some additional features used in its interaction with the surroundings (e.g. [Magnetic Grippers](/api/RigidBody.BaxterHandGrab.html)).

Some smaller namespaces contain tests and configuration functionality.

## Explore the API reference

For full details we refer you to the [API reference](/api/Configuration.html) included in this documentation site.
