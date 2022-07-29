"""
Module used to get the models for each step and for setting up the side channel with Unity.
"""
from typing import Callable, Tuple, List
import uuid
from enum import Enum

import gin
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from stable_baselines3 import DQN

from baselines.custom_dqn_policies import AddaptedAdamDQNPolicy
from utilities.filtered_wrapper import FilteredWrapper
from utilities.volatile_space_gym_wrapper import VolatileSpaceUnityGymWrapper
import numpy as np


class BaxterState(Enum):
    """
    Enum class for the different steps of the reinforcement learning.
    """
    GRAB_CLOTH_1 = 1
    FOLD_1 = 2
    GRAB_CLOTH_2 = 3
    FOLD_2 = 4

    @staticmethod
    def from_str(label: str):
        """Returns the correct Enum based on the given string

        :param label: Name of state
        :type label: str
        :raises ValueError: Thrown when unknown state is given as label
        :return: Enum of the state
        :rtype: BaxterState
        """
        if label == "GrabCloth1":
            return BaxterState.GRAB_CLOTH_1
        elif label == "Fold1":
            return BaxterState.FOLD_1
        elif label == "GrabCloth2":
            return BaxterState.GRAB_CLOTH_2
        elif label == "Fold2":
            return BaxterState.FOLD_2
        else:
            raise ValueError

    @staticmethod
    def to_csharp(label) -> str:
        """Returns the correct string based on the given enum value

        :param label: enum value
        :type label: BaxterState
        :raises NotImplementedError: Thrown when unknown enum value is given
        :return: String representing the enum value
        :rtype: str
        """
        if label == BaxterState.GRAB_CLOTH_1:
            return "GrabCloth1"
        elif label == BaxterState.FOLD_1:
            return "Fold1"
        elif label == BaxterState.GRAB_CLOTH_2:
            return "GrabCloth2"
        elif label == BaxterState.FOLD_2:
            return "Fold2"
        else:
            raise NotImplementedError


class StateChannel(SideChannel):
    """
    This Unity ML Agents Side Channel
    is used for communicating the switch to
    another step in the folding process from
    our Python environment to Unity.

    For example, when the first fold is completed this channel
    will be used to reconfigure the Unity environment for the next step.
    """

    def __init__(self, state_manager) -> None:
        """Initialize the Side Channel with a proper UUID.
        """
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.state_manager = state_manager

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        When a message is received, Unity is letting us know which state to move on to
        """
        message = msg.read_string()
        self.state_manager.set_state(BaxterState.from_str(message))

    def send_string(self, data: str) -> None:
        """This message is used to pass a
        string  message to Unity

        :param data: The message to be passed to Unity.
        :type data: str
        """
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(BaxterState.to_csharp(data))
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)


class StateManager:
    """
    This class manages the env based on the state of the training.
    """

    def __init__(self, train_state=None):
        self.env = None
        self.eval_model = None
        self.curr_state = None
        self.train_state = train_state  # the model to be trained

        self.__is_env_loaded = False
        self.__await_state = None

        # The following attributes are initialised to None to comply to the linter
        self.train_model = None
        self.train_observation_mask = None
        self.train_observation_range = []

        self.evaluation_model_creator = {
            BaxterState.GRAB_CLOTH_1: eval_grabcloth1,
            BaxterState.FOLD_1: eval_fold1,
            BaxterState.GRAB_CLOTH_2: eval_grabcloth2,
            BaxterState.FOLD_2: eval_fold2
        }
        self.evaluation_models = {}
        self.eval_observation_ranges = {}

        self.training_model_creator = {
            BaxterState.GRAB_CLOTH_1: train_grabcloth1,
            BaxterState.FOLD_1: train_fold1,
            BaxterState.GRAB_CLOTH_2: train_grabcloth2,
            BaxterState.FOLD_2: train_fold2
        }

    def set_state(self, state: BaxterState) -> None:
        """Sets the current state

        :param state: State in which the training should happen
        :type state: BaxterState
        """
        if not self.__is_env_loaded:  # if environment not available yet
            self.__await_state = state
            return

        if self.curr_state == state:
            print(self.curr_state)
            print("already in state")
            return

        print("New state: ", state)
        self.curr_state = state
        print(self.curr_state)

        if state != self.train_state:
            if state not in self.evaluation_models:
                (self.evaluation_models[state], self.eval_observation_ranges[state]) \
                    = self.evaluation_model_creator[state](self.env)
            self.eval_model = self.evaluation_models[state]
            self.env.set_observation_range(self.eval_observation_ranges[state])
        else:
            self.eval_model = None
            self.env.set_observation_range(self.train_observation_range)

    def initialize_env(self, env) -> None:
        """Initialize the environment in the correct state
        :param env: The current environment
        :type env: UnityToGymWrapper
        """
        self.env = env

        self.train_model, self.train_observation_range = \
            (None, None) if self.train_state is None \
            else self.training_model_creator[self.train_state](env)
        self.train_observation_mask = None if self.train_model is None else \
            np.concatenate([range(start, stop) for (start, stop) in self.train_observation_range])

        self.__is_env_loaded = True
        self.set_state(self.__await_state)
        self.__await_state = None


@gin.configurable
def eval_grabcloth1(env, observation_range: List[Tuple[int, int]],
                    load_name: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for evaluating GrabCloth1

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            evaluate this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param load_name: The location of the file containing the model to evaluate
    :type load_name: str
    :return:
        model : the training model for GrabCloth1
        observation_range : specifies which observations are used in this model
    """
    model = DQN.load(load_name, env=FilteredWrapper(env, observation_range))
    return model, observation_range


@gin.configurable
def eval_fold1(env, observation_range: List[Tuple[int, int]],
               load_name: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for evaluating Fold1

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            evaluate this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param load_name: The location of the file containing the model to evaluate
    :type load_name: str
    :return:
        model : the training model for Fold1
        observation_range : specifies which observations are used in this model
    """
    model = DQN.load(load_name, env=FilteredWrapper(env, observation_range))
    return model, observation_range


@gin.configurable
def eval_grabcloth2(env, observation_range: List[Tuple[int, int]],
                    load_name: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for evaluating GrabCloth2

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            evaluate this model based on the observations
                            received from the Unity side
    :type observation_range: list
    :param load_name: The location of the file containing the model to evaluate
    :type load_name: str
    :return:
        model : the training model for GrabCloth2
        observation_range : specifies which observations are used in this model
    """
    model = DQN.load(load_name, env=FilteredWrapper(env, observation_range))
    return model, observation_range


@gin.configurable
def eval_fold2(env, observation_range: List[Tuple[int, int]],
               load_name: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for evaluating Fold2

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            evaluate this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param load_name: The location of the file containing the model to evaluate
    :type load_name: str
    :return:
        model : the training model for Fold2
        observation_range : specifies which observations are used in this model
    """
    model = DQN.load(load_name, env=FilteredWrapper(env, observation_range))
    return model, observation_range


@gin.configurable
def train_grabcloth1(env: VolatileSpaceUnityGymWrapper,
                     observation_range: List[Tuple[int, int]], verbose: int,
                     gamma: float, batch_size: int, buffer_size: int,
                     learning_starts: int, learning_rate: float,
                     exploration_fraction: float,
                     exploration_initial_eps: float,
                     exploration_final_eps: float, target_update_interval: int,
                     tensorboard_log: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for training GrabCloth1

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            train this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param verbose: Specifies the verbosity of the Baselines DQN model
    :type verbose: int
    :param gamma: The gamma parameter of the DQN algorithm used
    :type gamma: float
    :param batch_size: The batch size used in training of the DQN algorithm
    :type batch_size: int
    :param buffer_size: size of the replay buffer
    :type buffer_size: int
    :param learning_starts: Number of simulation steps after which training starts
    :type learning_starts: int
    :param learning_rate: Learning rated used in training the DQN
    :type learning_rate: float
    :param exploration_fraction: Fraction by which the exploration rate epsilon is
                                 reduced after each training
    :type exploration_fraction: float
    :param exploration_initial_eps: Initial exploration fraction epsilon.
    :type exploration_initial_eps: float
    :param exploration_final_eps: The minimum value epsilon can reach.
    :type exploration_final_eps: float
    :param target_update_interval: Interval size to update the target network
    :type target_update_interval: int
    :param tensorboard_log: Path determining where to store the TensorBoard logs
    :type tensorboard_log: str

    :return:
        model : the training model for GrabCloth1
        observation_range : specifies which observations are used in this model
    """

    model = DQN(AddaptedAdamDQNPolicy,
                env=FilteredWrapper(env, observation_range),
                verbose=verbose,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                learning_rate=learning_rate,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                target_update_interval=target_update_interval,
                tensorboard_log=tensorboard_log)
    return model, observation_range


@gin.configurable
def train_fold1(env: VolatileSpaceUnityGymWrapper,
                observation_range: List[Tuple[int, int]], verbose: int,
                gamma: float, batch_size: int, buffer_size: int,
                learning_starts: int, learning_rate: float,
                exploration_fraction: float, exploration_initial_eps: float,
                exploration_final_eps: float, target_update_interval: int,
                tensorboard_log: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for training Fold1

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            train this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param verbose: Specifies the verbosity of the Baselines DQN model
    :type verbose: int
    :param gamma: The gamma parameter of the DQN algorithm used
    :type gamma: float
    :param batch_size: The batch size used in training of the DQN algorithm
    :type batch_size: int
    :param buffer_size: size of the replay buffer
    :type buffer_size: int
    :param learning_starts: Number of simulation steps after which training starts
    :type learning_starts: int
    :param learning_rate: Learning rated used in training the DQN
    :type learning_rate: float
    :param exploration_fraction: Fraction by which the exploration rate epsilon is
                                 reduced after each training
    :type exploration_fraction: float
    :param exploration_initial_eps: Initial exploration fraction epsilon.
    :type exploration_initial_eps: float
    :param exploration_final_eps: The minimum value epsilon can reach.
    :type exploration_final_eps: float
    :param target_update_interval: Interval size to update the target network
    :type target_update_interval: int
    :param tensorboard_log: Path determining where to store the TensorBoard logs
    :type tensorboard_log: str

    :return:
        model : the training model for Fold1
        observation_range : specifies which observations are used in this model
    """
    model = DQN(AddaptedAdamDQNPolicy,
                env=FilteredWrapper(env, observation_range),
                verbose=verbose,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                learning_rate=learning_rate,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                target_update_interval=target_update_interval,
                tensorboard_log=tensorboard_log)
    return model, observation_range


@gin.configurable
def train_grabcloth2(env: VolatileSpaceUnityGymWrapper,
                     observation_range: List[Tuple[int, int]], verbose: int,
                     gamma: float, batch_size: int, buffer_size: int,
                     learning_starts: int, learning_rate: float,
                     exploration_fraction: float,
                     exploration_initial_eps: float,
                     exploration_final_eps: float, target_update_interval: int,
                     tensorboard_log: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for training GrabCloth2

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            train this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param verbose: Specifies the verbosity of the Baselines DQN model
    :type verbose: int
    :param gamma: The gamma parameter of the DQN algorithm used
    :type gamma: float
    :param batch_size: The batch size used in training of the DQN algorithm
    :type batch_size: int
    :param buffer_size: size of the replay buffer
    :type buffer_size: int
    :param learning_starts: Number of simulation steps after which training starts
    :type learning_starts: int
    :param learning_rate: Learning rated used in training the DQN
    :type learning_rate: float
    :param exploration_fraction: Fraction by which the exploration rate epsilon is
                                 reduced after each training
    :type exploration_fraction: float
    :param exploration_initial_eps: Initial exploration fraction epsilon.
    :type exploration_initial_eps: float
    :param exploration_final_eps: The minimum value epsilon can reach.
    :type exploration_final_eps: float
    :param target_update_interval: Interval size to update the target network
    :type target_update_interval: int
    :param tensorboard_log: Path determining where to store the TensorBoard logs
    :type tensorboard_log: str

    :return:
        model : the training model for GrabCloth2
        observation_range : specifies which observations are used in this model
    """
    model = DQN(AddaptedAdamDQNPolicy,
                env=FilteredWrapper(env, observation_range),
                verbose=verbose,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                learning_rate=learning_rate,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                target_update_interval=target_update_interval,
                tensorboard_log=tensorboard_log)
    return model, observation_range


@gin.configurable
def train_fold2(env: VolatileSpaceUnityGymWrapper,
                observation_range: List[Tuple[int, int]], verbose: int,
                gamma: float, batch_size: int, buffer_size: int,
                learning_starts: int, learning_rate: float,
                exploration_fraction: float, exploration_initial_eps: float,
                exploration_final_eps: float, target_update_interval: int,
                tensorboard_log: str) -> Tuple[DQN, List[Tuple[int, int]]]:
    """Gets the model and observation range for training Fold2

    :param env: Unity environment to evaluate on
    :type env: UnityToGymWrapper
    :param observation_range: The range of observations that is used to
                            train this model based on the observations
                            received from the Unity side
    :type observation_range: List[Tuple[int, int]]
    :param verbose: Specifies the verbosity of the Baselines DQN model
    :type verbose: int
    :param gamma: The gamma parameter of the DQN algorithm used
    :type gamma: float
    :param batch_size: The batch size used in training of the DQN algorithm
    :type batch_size: int
    :param buffer_size: size of the replay buffer
    :type buffer_size: int
    :param learning_starts: Number of simulation steps after which training starts
    :type learning_starts: int
    :param learning_rate: Learning rated used in training the DQN
    :type learning_rate: float
    :param exploration_fraction: Fraction by which the exploration rate epsilon is
                                 reduced after each training
    :type exploration_fraction: float
    :param exploration_initial_eps: Initial exploration fraction epsilon.
    :type exploration_initial_eps: float
    :param exploration_final_eps: The minimum value epsilon can reach.
    :type exploration_final_eps: float
    :param target_update_interval: Interval size to update the target network
    :type target_update_interval: int
    :param tensorboard_log: Path determining where to store the TensorBoard logs
    :type tensorboard_log: str

    :return:
        model : the training model for Fold2
        observation_range : specifies which observations are used in this model
    """

    model = DQN(AddaptedAdamDQNPolicy,
                env=FilteredWrapper(env, observation_range),
                verbose=verbose,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                learning_rate=learning_rate,
                exploration_fraction=exploration_fraction,
                exploration_initial_eps=exploration_initial_eps,
                exploration_final_eps=exploration_final_eps,
                target_update_interval=target_update_interval,
                tensorboard_log=tensorboard_log)
    return model, observation_range


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate.
    :param initial_value: Initial learning rate.
    :return: schedule function to decay the parameter linearly.
    """

    def func(progress_remaining: float) -> float:
        """
        Determine the new learning rate according to a linear schedule.
        :param progress_remaining: Decays from the initial value 1.0 to 0.0.
        :return: new learning rate
        """
        if progress_remaining > 0.25:
            return initial_value
        else:
            return (progress_remaining + 0.75) * initial_value

    return func
