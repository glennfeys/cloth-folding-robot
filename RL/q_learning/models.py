"""
Contains the data models used in the Deep Q Learning implementation.
"""
from typing import NamedTuple, List, Tuple
import gin
import numpy as np


@gin.configurable
class ModelsConfig:
    """
    This class is used for configuring the structure
    of the training observations coming from unity.
    These consist of two parts: the state of Baxter and the state of the smart fabric.
    """

    def __init__(self, baxter_start: int, baxter_end: int, cloth_start: int,
                 cloth_end: int):
        """Configure the structure of the training observations

        :param baxter_start: Start index in the observation space of Baxter's joint states
        :type baxter_start: int
        :param baxter_end: End index in the observation space of Baxter's joint states
        :type baxter_end: int
        :param cloth_start: Start index in the observation space of the cloth's state
        :type cloth_start: int
        :param cloth_end: End index in the observation space of the cloth's state
        :type cloth_end: int
        """
        self.baxter_start = baxter_start
        self.baxter_end = baxter_end
        self.cloth_start = cloth_start
        self.cloth_end = cloth_end


class Observation:
    """
    Wrapper class used for converting
    the observation space yielded by
    a .step() to a more structured format.
    """

    class Baxter:
        """
        Wrapper class for the state of Baxter's joints
        """

        def __init__(self, points: np.ndarray):
            """Create a wrapper around the joint state as observed in Unity

            :param points: [description]
            :type points: [type]
            """
            self.points = points

        def __getitem__(self, key):
            return self.points[key]

        def __str__(self):
            return str(self.points)

        def __repr__(self):
            return "Baxter(" + str(self) + ")"

    class Cloth:
        """
        Wrapper class for the state of the Cloth
        """

        def __init__(self, amount_of_points_x: int, amount_of_points_z: int,
                     points: np.ndarray):
            self.amount_of_points_x = amount_of_points_x
            self.amount_of_points_z = amount_of_points_z
            self.points = points

        def __getitem__(self, key: Tuple[int, int]):
            """Get the value of the point within
            our cloth's state at position key.

            :param key: Position in the cloth
            :type key: Tuple[int, int]
            :return: Value of our cloth state at the position
            :rtype: float
            """
            row, col = key
            index = self.amount_of_points_x * row * 3 + col * 3
            return self.points[index:index + 3]

        def __str__(self):
            i = 2
            vectors = []
            while i <= self.points.size - 1:
                vectors.append(str(self.points[i - 2:i + 1]))
                i += 3

            return "[" + "\n".join(vectors) + "]"

        def __repr__(self):
            return "Cloth(" + str(self) + ")"

    def __init__(self, observation_space: np.ndarray):
        """Wrapper class used for converting
          the observation space yielded by
          a .step() to a more structured format.

        :param observation_space: State from GymEnvironment
        :type observation_space: np.ndarray
        """
        # pylint doesn't pick up that this model is configured using gin, and thus doesn't need arguments.
        # pylint: disable=no-value-for-parameter
        models_config = ModelsConfig()

        self.observation_space = observation_space
        self.baxter = Observation.Baxter(
            observation_space[models_config.
                              baxter_start:models_config.baxter_end + 1])
        # Note: The number of X and Z points of the cloth were specified in the observations
        #       in our custom Q Network. Due to the fact our models are also trained
        #       with these two additional inputs, we were unable to remove this within
        #       the time constraints
        self.cloth = Observation.Cloth(
            int(observation_space[models_config.cloth_start]),
            int(observation_space[models_config.cloth_start + 1]),
            observation_space[models_config.cloth_start +
                              2:models_config.cloth_end + 1])

    def get_baxter(self):
        """Get the state of Baxter.

        :return: A structured object yielding Baxter's state
        :rtype: Observation.Baxter
        """
        return self.baxter

    def get_cloth(self):
        """Get the state of the Cloth.

        :return: A structured object yielding the Cloth's state
        :rtype: Observation.Cloth
        """
        return self.cloth

    def combine(self) -> np.ndarray:
        """Get the combined state
        of Baxter and the cloth.

        :return: The combined state as returned by the environment
        :rtype: np.ndarray
        """
        return self.observation_space


class Action:
    """
    Wrapper class for the decision output
    as given to our gym environment
    """

    def __init__(self, decision_output: np.ndarray):
        """This method takes the output of our DQN and allows easy conversion
        to the action format expected by our Unity gym environment.

        :param decision_output: The output (decision) made by the Deep Q Network
        :type decision_output: np.ndarray
        """
        self.decision_output = decision_output


class Experience(NamedTuple):
    """
    An experience contains the data of one Agent transition.
    - Observation
    - Action
    - Reward
    - Done flag
    - Next Observation
    """

    obs: Observation
    action: Action
    reward: float
    done: bool
    next_obs: Observation


# A Trajectory is an ordered sequence of Experiences
Trajectory = List[Experience]

# A Buffer is an unordered list of Experiences from multiple Trajectories
Buffer = List[Experience]
