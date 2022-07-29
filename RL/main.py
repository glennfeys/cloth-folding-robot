"""
Main module used for connecting to the Unity Environment,
initializing Stable Baselines DQN models and performing
the training.
"""
import argparse
import sys
import gin

from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from utilities.state_manager import StateManager, StateChannel, BaxterState
from utilities.volatile_space_gym_wrapper import VolatileSpaceUnityGymWrapper
from utilities.mode_channel import ModeChannel

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='Specify the config profile',
                    default='config')
parser.add_argument('--name',
                    help='The name of the training',
                    default='TrainingNoName')
parser.add_argument('--train', help='The model to train', default=None)
parser.add_argument('--eval',
                    action='store_true',
                    help='Evaluation mode',
                    default=False)
parser.add_argument(
    '--single',
    action='store_true',
    help=
    'Single stage mode (train a single stage), else the model will be trained in a chained manner.',
    default=False)
args, known = parser.parse_known_args()

# Check arguments for mutual exclusivity
if args.single and args.eval:
    print('--single and --eval are mutually exclusive', file=sys.stderr)
    sys.exit(1)
if args.train and args.eval:
    print('--train and --eval are mutually exclusive', file=sys.stderr)
    sys.exit(1)

training_name = args.name
train_state = None if args.train is None else BaxterState.from_str(args.train)
config_file = args.config
evaluate_mode = args.eval or train_state is None
single_stage_mode = args.single


@gin.configurable
def train_loop(unity_file: str, unity_log_file: str, total_timesteps: int,
               model_folder: str, save_name: str):
    """This method will start a training loop of the Reinforcement Learning
    using the specified parameters.

    :param unity_file: Path to the Unity executable or None to use the Unity editor to run training
                       on.
    :type unity_file: str
    :param unity_log_file: Path to store the log file of the Unity executable.
    :type unity_log_file: str
    :param total_timesteps: The total amount of timesteps to run the training.
    :type total_timesteps: int
    :param model_folder: Folder where to store the training weights
    :type model_folder: str
    :param save_name: Name of the file in which to store the model
    :type save_name: str
    """
    print(f'training {train_state}')

    state_manager = StateManager(train_state)
    state_channel = StateChannel(state_manager)

    env = UnityEnvironment(file_name=unity_file,
                           seed=1,
                           side_channels=[state_channel],
                           log_folder=unity_log_file + training_name)
    env = VolatileSpaceUnityGymWrapper(env, state_manager)
    state_manager.initialize_env(env)

    # let unity know which model we're training, so it can end the episode after this state
    state_channel.send_string(train_state)

    eval_callback = EvalCallback(Monitor(state_manager.train_model.env.envs[0]),
                                 best_model_save_path=model_folder + 'best/',
                                 log_path=model_folder + 'logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False)
    state_manager.train_model.learn(total_timesteps=total_timesteps,
                                    tb_log_name=save_name,
                                    callback=eval_callback)
    state_manager.train_model.save(model_folder + save_name)
    env.close()


def eval_loop():
    """This method will go through the different models to evaluate the model.
    """
    print("evaluation mode")
    state_manager = StateManager()
    state_channel = StateChannel(state_manager)

    env = UnityEnvironment(file_name=None,
                           seed=1,
                           side_channels=[state_channel])
    env = VolatileSpaceUnityGymWrapper(env)
    state_manager.initialize_env(env)

    steps = 10000
    obs = env.reset()
    for _ in range(steps):
        action, _state = state_manager.eval_model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


@gin.configurable
def single_stage_training(unity_file: str, unity_log_file: str,
                          total_timesteps: int, model_folder: str,
                          save_name: str):
    """This method will start the training of a single stage with the specified
    training stage

    :param unity_file: Path to the Unity executable or None to use the Unity editor to run training
                       on.
    :type unity_file: str
    :param unity_log_file: Path to store the log file of the Unity executable.
    :type unity_log_file: str
    :param total_timesteps: The total amount of timesteps to run the training.
    :type total_timesteps: int
    :param model_folder: Folder where to store the training weights
    :type model_folder: str
    :param save_name: Name of the file in which to store the model
    :type save_name: str
    """
    print("training single stage {}".format(train_state))

    mode_channel = ModeChannel()

    state_manager = StateManager(train_state)
    state_channel = StateChannel(state_manager)

    env = UnityEnvironment(file_name=unity_file,
                           seed=1,
                           side_channels=[state_channel, mode_channel],
                           log_folder=unity_log_file + training_name)
    env = VolatileSpaceUnityGymWrapper(env, state_manager)
    state_manager.initialize_env(env)

    mode_channel.send_string('Single')
    # let unity know which model we're training, so it can end the episode after this state
    state_channel.send_string(train_state)

    eval_callback = EvalCallback(Monitor(env),
                                 best_model_save_path=model_folder + 'best/',
                                 log_path=model_folder + 'logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False)
    state_manager.train_model.learn(total_timesteps=total_timesteps,
                                    tb_log_name=save_name,
                                    callback=eval_callback)
    state_manager.train_model.save(model_folder + save_name)
    env.close()


if __name__ == "__main__":
    gin.parse_config_file('configs/{}.gin'.format(config_file))
    # pylint doesn't pick up that this model is configured using gin, and thus doesn't need arguments.
    # pylint: disable=no-value-for-parameter
    if evaluate_mode:
        eval_loop()
    elif single_stage_mode:
        single_stage_training()
    else:
        train_loop()
