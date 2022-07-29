"""
This module contains an implementation of an
MLP backed Double DQN based on TensorFlow.
"""
import random

import gin
import q_learning.q_network as q_network
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from q_learning.models import Observation, Buffer


@gin.configurable
class MLPQNetwork(q_network.QNetwork):
    """
    This class implements an MLP-Based
    Double Deep Q Network using TensorFlow
    """

    def __init__(self, input_size: int, output_size: int, batch_size: int,
                 epochs: int, gamma: float):

        self.mlp = MLPQNetwork._build_model(input_size, output_size)
        self.mlp_target = MLPQNetwork._build_model(input_size, output_size)
        self.mlp_target.set_weights(self.mlp.get_weights())
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma

    def inference(self, obs: Observation) -> np.ndarray:
        observation = obs.observation_space
        observation = observation.reshape((1, observation.shape[0]))
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        action = self.mlp_target.predict(state)
        action = action[0]
        return action

    def train(self, buffer: Buffer) -> None:
        batch_size = min(len(buffer), self.batch_size)
        random.shuffle(buffer)
        # Split the buffer into batches
        batches = [
            buffer[batch_size * start:batch_size * (start + 1)]
            for start in range(len(buffer) // batch_size)
        ]
        for batch in batches:
            for _ in range(self.epochs):
                X = []
                Y = []
                targets = self.mlp_target.predict(
                    tf.convert_to_tensor(np.stack(
                        [ex.obs.observation_space for ex in batch]),
                                         dtype=tf.float32))
                future_targets = self.mlp_target.predict(
                    tf.convert_to_tensor(np.stack(
                        [ex.next_obs.observation_space for ex in batch]),
                                         dtype=tf.float32))
                for idx, _ in enumerate(batch):
                    sample = batch[idx]
                    state = sample.obs.observation_space.reshape(1, -1)
                    reward = sample.reward
                    done = sample.done
                    action = np.argmax(sample.action.decision_output)
                    target = targets[idx]
                    if done:
                        target[action] = reward
                    else:
                        # Compute the Bellman Equation
                        q_future = max(future_targets[idx])
                        target[action] = reward + q_future * self.gamma
                    X.append(state)
                    Y.append(target.reshape(1, -1))
                X = tf.concat(X, 0)
                Y = tf.concat(Y, 0)
                self.mlp.fit(x=X, y=Y, epochs=1, batch_size=batch_size)
        self.mlp_target.set_weights(self.mlp.get_weights())

    @staticmethod
    def _build_model(input_size: int, output_size: int):
        """This static method generates
        a sequential model used in our DQN

        :param input_size: [description]
        :type input_size: int
        :param output_size: [description]
        :type output_size: int
        :return: [description]
        :rtype: [type]
        """
        q_net = Sequential()
        # Input layer
        q_net.add(
            Dense(input_size,
                  input_dim=input_size,
                  activation='relu',
                  kernel_initializer='he_uniform'))

        q_net.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))

        # Output layer
        q_net.add(
            Dense(output_size,
                  activation='linear',
                  kernel_initializer='he_uniform'))

        # Configure the optimizer
        q_net.compile(optimizer=tf.optimizers.Adam(1E-3), loss='mse')
        return q_net
