"""
This module defines an abstract class used
for conforming the interface of our QNetwork
implementation (i.e. DDQN).
"""

import numpy as np
from q_learning.models import Buffer, Observation


class QNetwork:
    """
    Informal class defining what our Q Network should adhere to.
    """

    def inference(self, obs: Observation) -> np.ndarray:
        """For a given observation run inference and
        return the actions which should be performed.

        :param obs: The observations made by the agent
        :type obs: Observation
        :return: The actions to be performed in the next step()
        :rtype: np.ndarray
        """

    def train(self, buffer: Buffer) -> None:
        """Perform a new training episode using
        the experiences in the Buffer.

        :param buffer: Experiences buffer made during the latest episodes
        :type buffer: Buffer
        """
