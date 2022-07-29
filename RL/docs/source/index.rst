.. Software Engineering Lab 3: Reinforcement Learning documentation master file, created by
   sphinx-quickstart on Wed May  5 16:27:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reinforcement Learning
======================
Welcome to the documentation site of our Reinforcement Learning project for the Software Engineering Lab 3 assignment.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Main components of the project
==============================
This project has 3 important modules:

* The :mod:`main` module which is implemented as a command line interface using ``argparser`` for running training and evaluating a trained model.
* The :mod:`baselines` module wich contains the implementation of a number of custom ``DQNPolicies`` for Stable Baselines 3.
* The :mod:`utilities` module which encapsulates logic related to the multi-step training, i.e. a wrapper for the ``GymEnvironment`` and ``SideChannel``.

The :mod:`q_learning` module wich contains code for our custom implementation of a DQN, that was used during the first weeks of the project.
  

Indices and tables
==================
Below are some indices and search pages that can be used for an additional overview of the project.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
