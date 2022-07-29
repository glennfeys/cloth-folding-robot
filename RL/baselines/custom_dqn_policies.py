"""
This module contains the custom DQN Policy configurations
we made while training our ML Agent using StableBaselines3.
"""
from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule


class RMSPropDQNPolicy(DQNPolicy):
    """A Custom DQN Policy using
    the RMSProp loss.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = [
            256, 128, 64
        ],  # This list will not be modified so this is safe
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[
            BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.RMSprop,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(RMSPropDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


class AdaptedAdamDQNPolicy(DQNPolicy):
    """A Custom DQN Policy using
    the Adam loss.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = [256, 128, 64],
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[
            BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(AdaptedAdamDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


# Unfortunately due to a mistake in the past, we need to keep this class with a typo to keep compatibility with
# the old models.
class AddaptedAdamDQNPolicy(AdaptedAdamDQNPolicy):
    """duplicate for legacy models
    """
    pass


class SGDDQNPolicy(DQNPolicy):
    """A Custom DQN Policy using
    the stochastic gradient descent loss.
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = [256, 128, 64],
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[
            BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(SGDDQNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("RMSPropDQNPolicy", RMSPropDQNPolicy)
register_policy("AdaptedAdamDQNPolicy", AdaptedAdamDQNPolicy)
register_policy("AddaptedAdamDQNPolicy", AddaptedAdamDQNPolicy)
register_policy("SGDDQNPolicy", SGDDQNPolicy)
