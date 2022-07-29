"""
This module contains code for running
general training on a specified environment
with a chosen QNetwork implementation.
"""

import random
from typing import Tuple, Optional

import gin
from gym_unity.envs import UnityToGymWrapper
from q_learning.models import Buffer, Trajectory, Experience, Observation, Action
import numpy as np
import q_learning.q_network as q_network


class Trainer:
    """
    This class contains methods used for running
    training on a given QNetwork and Environment.
    """

    @staticmethod
    @gin.configurable
    def generate_trajectories(env: UnityToGymWrapper, q_net: q_network.QNetwork,
                              buffer_size: int,
                              epsilon: float) -> Tuple[Buffer, float]:
        """Given a Unity Gym Environment and a Q-Network, this method will generate a
        buffer of Experiences obtained by running the Environment with the Policy
        derived from the Q-Network.

        :param env: The Unity Gym Environment to perform our training against
        :type env: UnityToGymWrapper
        :param q_net: The QNetwork determining the agent's policy
        :type q_net: QNetwork
        :param buffer_size: The size of the experience Buffer (replay memory) we want to attain
        :type buffer_size: int
        :param epsilon: A parameter used to perturb the actions using a numpy randn()
        :type epsilon: float
        :return: The obtained experiences
        :rtype: Buffer
        """
        # Create an empty Buffer
        buffer: Buffer = []

        # Reset the environment
        env.reset()

        # Trajectory for the current agent
        trajectory: Trajectory = []

        last_observation: Optional[Observation] = None
        cumulative_rewards = 0.0

        # The last action of the agent (using a one hot encoding of the possible actions)
        last_action: Action = Action(np.zeros(14))
        while len(buffer) < buffer_size:  # While not enough data in the buffer
            observation, reward, done, _ = env.step(
                last_action.decision_output.tolist())
            reward += 70

            cumulative_rewards += reward
            if done:
                # Create its last experience (is last because the Agent terminated)
                last_experience = Experience(
                    obs=Observation(last_observation) if last_observation
                    is not None else Observation(observation.copy()),
                    reward=reward,
                    done=done,
                    action=last_action,
                    next_obs=Observation(observation.copy()),
                )
                # Add the Trajectory and the last experience to the buffer
                buffer.extend(trajectory)
                buffer.append(last_experience)
                trajectory = []

                # Clear its last observation and action (Since the trajectory is over)
                last_observation = None

                last_action = Action(np.zeros(14))
                env.reset()
                continue
            else:
                # If the Agent requesting a decision has a "last observation"
                if last_observation is not None and last_action.decision_output.any(
                ):
                    # Create an Experience from the last observation and the Decision Step
                    exp = Experience(
                        obs=Observation(last_observation),
                        reward=reward,
                        done=False,
                        action=last_action,
                        next_obs=Observation(observation.copy()),
                    )
                    # Update the Trajectory of the Agent
                    trajectory.append(exp)
                # Store the observation as the new "last observation"
                last_observation = observation.copy()

            # Generate an action for all the Agents that requested a decision
            # Compute the values for each action given the observation
            if random.uniform(0, 1) >= epsilon:
                actions_values = q_net.inference(Observation(observation))
            else:
                # Add some noise with epsilon to the values
                pos = random.randint(0, 13)
                actions_values = np.zeros(14)
                actions_values[pos] = 1.0
            # Store the action that was picked, it will be put in the trajectory later
            last_action = Action(actions_values)

        return buffer, cumulative_rewards

    @staticmethod
    def update_q_net(q_net: q_network.QNetwork, buffer: Buffer) -> None:
        """Update the QNetwork with the latest episode's results

        :param q_net: the QNetwork to train
        :type q_net: QNetwork
        :param buffer: latest experiences
        :type buffer: Buffer
        """
        q_net.train(buffer)
