"""The volatile space gym wrapper allows us to change observation and action space between tasks.
"""
from typing import List, Any, Tuple, Union

import numpy as np
from gym_unity.envs import UnityToGymWrapper, GymStepResult
from mlagents_envs.base_env import BaseEnv


class VolatileSpaceUnityGymWrapper(UnityToGymWrapper):
    """This is a subclass of the UnityToGymWrapper
    which allows us to change the observation and action space
    in between tasks.
    """

    def __init__(self, unity_env: BaseEnv, state_dto: 'StateManager' = None):
        """ Creates a Gym wrapper around a given gym wrapper, where the observation and
        action space can be changed in between tasks
        :param unity_env: The environment
        :param state_dto: if this parameter is passed, it will use the embedded
        model to pre-evaluate the model until
        it reaches the desired state to start training
        """

        super().__init__(unity_env)

        self.__observation_mask = list(range(self._observation_space.shape[0]))
        self.state_dto = state_dto

    def set_observation_range(self, ranges: List[Tuple[int, int]]) -> None:
        """ Hides the non-relevant observations from the Gym class using this wrapper by passing a
        list of ranges of relevant observations
        @param ranges: list of ranges [(start, stop), (start, stop)] in the observation space
        """
        selected_indexes = np.concatenate(
            [range(start, stop) for (start, stop) in ranges])
        self.__observation_mask = selected_indexes

    def step(self, action: List[Any], use_train_mask=True) -> GymStepResult:
        """ Perform one timestep in the environment, taking our own
        MLPQNetwork's action as input and transforming it a single action for Unity

        :param action: The output (decision) made by the Deep Q Network
        :param use_train_mask: Uses the mask of the model that is being trained, if available
        :return:
            observation (object/list): agent's observation of the current environment
            reward (float/list) : amount of reward returned after previous action
            done (boolean/list): whether the episode has ended.
            info (dict): contains auxiliary diagnostic information.
        """
        use_train_mask = use_train_mask and self.state_dto is not None and \
         self.state_dto.train_model is not None

        observation, reward, done, info = super().step(action)
        mask = self.__observation_mask if not use_train_mask else \
         self.state_dto.train_observation_mask
        return observation[mask], reward, done, info

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        """ Resets the state of the environment and returns an initial observation.
        Returns: observation (object/list): the initial observation of the
        space.
        """

        obs = super().reset()[self.__observation_mask]

        while self.state_dto is not None and \
         self.state_dto.train_state != self.state_dto.curr_state:
            action, _state = self.state_dto.eval_model.predict(obs)
            obs, _, done, _ = self.step(action, False)
            self.render()
            if done:
                print(
                    "evaluation of previous models did not reach"\
                     "a state where it could start training {}"
                    .format(self.state_dto.train_state))
                obs = super().reset()[self.__observation_mask]

        return obs
