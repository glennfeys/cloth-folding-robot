# SEL3: Reinforcement Learning

This project contains the source code of the reinforcement learning facet of our project for
Software Engineering Lab 3, in our final version we are using StableBaselines3 for our DQN
implementation alongside some custom policies (see `baselines/custom_dqn_policies.py`). However,
our own custom DQN implementation used in the first weeks of the project can still be found in
`q_learning/`.

## Generating documentation for the project

This project uses Sphinx-style docstrings in order to document classes, methods and modules. In the folder `docs/` you can find a Sphinx project configured properly for generating a documentation site from the source code.

### Install Sphinx

First, follow the instructions on the website of the [Sphinx project](https://www.sphinx-doc.org/en/master/usage/installation.html) for your OS.

### Required dependencies

Be sure that you have installed the required dependencies in `requirements.txt` via pip (see the next section of the README for more details).

### Generate documentation

To run the generator and build documentation for our project run the following commands.

```bash
cd docs/
make html
```

Your documentation site will be built in `docs/build/html`. You can use `python -m http.server` or any other web server to open the documentation site in a browser now.

## How to get started with this project

The project is set up using pyenv/virtualenv, [Gin configuration](https://github.com/google/gin-config) and Sphinx documentation.

### Install pyenv

We'll be using pyenv to ensure everyone is running the same Python version, including the Gorilla workstation.

To install pyenv follow the instructions on [Github](https://github.com/pyenv/pyenv#installation) (a version compatible with WSL is also available).

If you aren't using pyenv, beware of potential issues and check your version is compatible with the Python version in `.python-version`.
We recommend using Python 3.8.x, Python 3.9.x is known to cause issues with the chosen version of ML-Agents.

### Use virtualenv

Dependency management in Python is hard, as to ensure consistency we'll be using virtualenv. The `/env` folder is excluded from version control.

If you don't have virtualenv on your system yet, install it as follows:

```bash
pip install virtualenv
```

On macOS and Linux:

```bash
python3 -m venv env
```

On Windows:

```bash
py -m venv env
```

Activate the virtual-environment.

On macOS and Linux:

```bash
source env/bin/activate
```

On Windows:

```bash
.\env\Scripts\activate
```

Whenever you push your code, be sure to freeze your environment

```bash
python3 -m pip freeze --exclude-editable > requirements.txt
```

### Bootstrap Environment

The `bootstrap_env.sh` script will allow you to install a consistent release version of the ML Agents. Run it to install all ML Agents and other dependencies for this project using `pip`.

### Coding Style

As to guarantee consistent styling of our code, we are using the [Google Python Styleguide](https://google.github.io/styleguide/pyguide.html). However, to simplify following these style rules we'll be adding the rules for pylint in `pylintrc`. As to ensure the behaviour of the VSCode formatter is consistent it might be handy to update your `.vscode/settings.json` to contain the following.

**Note:** Be sure your interpreter path is correctly set in VSCode to the virtual environment.

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintPath": "pylint",
  "editor.formatOnSave": true,
  "python.formatting.provider": "yapf",
  "python.linting.pylintEnabled": true,
  "python.formatting.yapfArgs": [
    "--style={based_on_style: google, indent_width: 4}"
  ],
  "autoDocstring.docstringFormat": "sphinx"
}
```

To simplify generating documentation, install the `autoDocstring` vscode extension. It will be set to automatically generate the correct format for Sphinx docstrings.

## Gin Configuration Framework

We use the [Gin configuration](https://github.com/google/gin-config) framework to set configuration variables in our project. More specifically, the default value of any function parameter (or class constructor parameter) can be configured inside the config.gin file.

To make a function's arguments configurable, add the annotation @gin.configurable on top and add the configuration parameters to the config.gin file like this;
`<function name>.<argument name> = <default value>`

The config files are to be found in `configs/`, the default config file is `config.gin` but any other config file can be added and specified with the commandline argument `--config <filename>` to `main.py` where `<filename>` doesn't contain the file extension. For example `--config dev` will use the `configs/dev.gin` file.

## Tensorboard

We use tensorboard for logging information about the training progress, such as episode length, reward and exploration rate.

To monitor/view these logs during or after training, use following command:

`tensorboard --logdir ./tensorboard_logs`

you can then visit http://localhost:6006 to access the tensorboard frontend (you can choose to use another port with `--port=1234`)

## Configuring the Unity Simulation

The Unity Simulation also has a number of configurable parameters relating to the cloth specifically as well as the simulation timescale, these can be found in `defaultConfiguration.xml`. 
## Training the models locally

The `main.py` script accepts a number of parameters and is the main entrypoint for training and evaluating the RL side of this project. Firstly, have a look at the files in `configs/`, here you'll find a number of `*.gin` files. The `gorilla.gin` file is utilised in the training on the gorilla workstation. The `config.gin` file is the default configuration.

To train locally you'll have to open the Unity editor on the `TestingScene` for the chained training, for training a single stage you'll have to ensure the respective scene was selected. This is done by the `train.sh` script on Gorilla.

To start a training of a stage of the folding process, you can specify its name as defined in `utilities/state_manager.py` using the BaxterState-enum.

| Stage                                                 | parameter value |
| ----------------------------------------------------- | --------------- |
| Grab the cloth from a neutral position                | GrabCloth1      |
| Perform the first fold                                | Fold1           |
| Grab the already folded cloth from a neutral position | GrabCloth2      |
| Perform the second fold                               | Fold2           |

Start a training of a stage of the folding process in a chained manner:

```bash
python main.py --config <profile_name> --train <stage_name> --name <name_for_results>
```

Note that the default configuration should be fine to evaluate and train the chained model locally, so you can simply run without the `--config` parameter.

To run the trained models in a chained manner, run the following command: _Used for creating the screencasts and evaluating the model._

```bash
python main.py --config <profile_name> --eval
```

You can also train a single stage, here we will not be evaluating any prior states (i.e. trained models). However here you'll have to ensure the relevant scene in Unity is running.

```bash
python main.py --config <profile_name> --train <stage_name> --name <name_for_results> --single
```

## Training on Gorilla Workstation

The training on the Gorilla workstation consists of several steps as described below.

### Building the Unity project

If there are any changes to the Unity project, the first step should be rebuilding the executable. This can be done with the command `docker exec build_container sh build.sh`. The executable can be found in `~/Project/Unity_Simulation/Build/`.

### Screen

To be able to quit the ssh session while running a training, we use `screen`. At this moment, every training should be run on the `training_screen`. You can see if the screen is already open with `screen -ls`. If the screen isn't open, you should run the command `screen -S training_screen`, otherwise you can switch to the correct screen with `screen -r`. You can leave the screen without stopping the training with `ctrl + a d`.

### Training script

Once you are in the correct screen, you can finally run the training script. Switch to RL directory and run the command `./train.sh $TRAINING_RUN_NAME $TRAINING_SCENE_NAME $TRAINING_TYPE [--single]`.

- `$TRAINING_RUN_NAME`: Directory name under which the logs for your training run are stored.
- `$TRAINING_SCENE_NAME`: The name of the scene in the Unity build to start.
- `$TRAINING_TYPE`: The part of the program you want to train (GrabCloth1, Fold1, GrabCloth2, Fold2).
- `--single`: (Optional) Pass in --single to train the model only on the given type.

The Unity logs, Python logs and checkpoints can all be found in `/media/data/sel/sel01` in the appropriate directory.


## Observation Config

_Note: This no longer applies to the Stable Baselines 3 implementation, but is relevant for evaluating our custom DQN._

In the Gin config files, some values depend on values from the unity environment. Make sure these are correct, i.e. matching with the unity config \(also see Unity_Simulation/README.md\)

the parameters concerning the observations are as follows:
| Config parameter name | Example value | Description |
| ------------------------ | ------------- | ----------- |
| ModelsConfig.baxter_end | 14 | end index of joint observations
| ModelsConfig.cloth_start | 15 | start index of cloth observations
| ModelsConfig.cloth_end | 2044 | end index of cloth observations
| MLPQNetwork.input_size | 2045 | observation vector size
| MLPQNetwork.output_size | 14 | action vector size