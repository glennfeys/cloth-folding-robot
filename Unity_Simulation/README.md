# Unity Simulation

Welcome to our Unity project used in simulating the RL environment our agent will be trained on in order to learn to fold a piece of fabric. The project itself consists of a number of components, separated using different namespaces. For the interaction between Unity and our Python reinforcement learning environment, we're using the Unity ML Agents framework (Release 12).

## How to generate the project documentation?

This project contains XML Docstrings on methods and classes where relevant and necessary. We use this information to generate a documentation site with [the DocFx tool by Microsoft](https://dotnet.github.io/docfx/).

### Install DocFx

Before generating the documentation, you'll need to install the DocFx tool. Instructions are provided on [the github site](https://dotnet.github.io/docfx/tutorial/docfx_getting_started.html#2-use-docfx-as-a-command-line-tool).

### Generate the documentation.

Once you've installed the DocFx tool using Homebrew, NuGet or directly from Github, you can run the following command to generate & serve the documentation.

```bash
docfx Documentation/docfx.json --serve
```

You can now visit the documentation at [localhost:8080/manual/intro.html](http://localhost:8080/manual/intro.html).

## How to change observation size

When changing the observations, make sure to check that the observation space size is correct.
This can be set using the Behavior Parameters in the inspector window (when baxter is selected):

![Observation Size](https://i.imgur.com/AkxtnFw.jpg)

**Vector Observation > Space Size** should be set to the amount of observations added to the VectorSensor in CollectObservations.

Also see RL/README.md concerning the observation space configuration on the python side, as it must match the configuration in unity.

## Running tests

The spatial hasher implementation contains some tests. You can use Unity's test runner in play mode to run these tests.
You can open the test runner by clicking **Window > General > Test Runner**.
