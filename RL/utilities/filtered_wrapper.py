"""Module for custom wrapper used to change the observation range
"""
from typing import List, Tuple

from gym import Wrapper, spaces
import numpy as np


class FilteredWrapper(Wrapper):
    """This is a subclass for the gym wrapper class allowing us to
    change the observations by using a custom observation range.
    """

    def __init__(self, env, observation_range: List[Tuple[int, int]]):
        super().__init__(env)

        self.__observation_mask = np.concatenate(
            [range(start, stop) for (start, stop) in observation_range])
        self.observation_space = spaces.Box(
            low=env.observation_space.low[self.__observation_mask],
            high=env.observation_space.high[self.__observation_mask],
            shape=(len(self.__observation_mask),))
