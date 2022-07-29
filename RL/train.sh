#! /bin/bash

if [ "$#" -eq 3 -o "$#" -eq 4]; then
  source env/bin/activate
  export TRAIN_SCENE="$2"
  if ["$#" -eq 3]; then
    taskset --cpu-list 0-6 xvfb-run --auto-servernum --error-file error.txt --server-args='-screen 0 640x480x24:32' python -u main.py --config gorilla --train "$3" --name "$1" | tee /media/data/sel/sel01/TrainingLog/"$1".log
  else
    taskset --cpu-list 0-6 xvfb-run --auto-servernum --error-file error.txt --server-args='-screen 0 640x480x24:32' python -u main.py --config gorilla --train "$3" --name "$1" --single | tee /media/data/sel/sel01/TrainingLog/"$1".log
  fi
  deactivate
else
  echo "train.sh: Utility script for running training on Gorilla" >&2
  echo "Usage: $0 <training_run_name> <training_scene_name> <training_type>" >&2
  echo "<training_run_name>: Directory name under which the logs for your training run are stored." >&2
  echo "<training_scene_name>: The name of the scene in the Unity build to start." >&2
  echo "<training_type>: The part of the program you want to train (GrabCloth1, Fold1, GrabCloth2, Fold2)." >&2
  echo "<single>: (Optional) Train the model only on the given type."
  exit 1
fi
