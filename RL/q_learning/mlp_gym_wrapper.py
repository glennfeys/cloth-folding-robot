"""
This module provides a wrapper for the Unity Environment
that was used in our custom MLP QNetwork.
"""

from typing import List

import numpy as np
from gym_unity.envs import UnityToGymWrapper, GymStepResult
from mlagents_envs.base_env import BaseEnv

from q_learning.models import ModelsConfig


class UnityToMLPGymWrapper(UnityToGymWrapper):
    """
    This class provides a wrapper for the Unity Gym Environment
    that is tailored towards our own custom MLP QNetwork and converts
    its outputs to the correct Unity input.
    """

    def __init__(self, unity_env: BaseEnv):
        super().__init__(unity_env)
        # pylint doesn't pick up that this model is configured using gin, and thus doesn't need arguments.
        # pylint: disable=no-value-for-parameter
        self.models_config = ModelsConfig()

    def step(self, action: List[float]) -> GymStepResult:
        """Perform one timestep in the environment, taking our own
        MLPQNetwork's action as input and transforming it a single action for Unity

        :param action: The output of the DQN; in the MLP QNetwork configured to be
         of length 7 (one for each joint on a single arm)
        :type action: List[float]
        :return: The output of a GymStep by the Unity Gym Environment
        :rtype: GymStepResult
        """
        action = np.array(action)

        # Determine which joint moves
        position = np.argmax(action)
        joint = 1 + (position // 2)

        # Convert this to the input position expected by unity
        unity_action = (position % 2)

        # Creating a list for the input of the Unity environment
        result = np.zeros(self.models_config.baxter_end + 1)
        if action.any():
            # 0 in our unity_action means backwards; 1 means forward; we convert this for Unity
            if unity_action == 0:
                unity_action = -1
            result[joint] = unity_action

            # For stretching joints we won't mirror the action and
            # just perform the same action on the joint on the opposite arm,
            # for rotating joints we will however mirror the action itself
            mirror = 1
            if (joint - 1) % 2 == 0:
                mirror = -1
            result[-7:] = result[1:8] * mirror

        # Perform the action on the Unity Environment
        return super().step(list(result))
