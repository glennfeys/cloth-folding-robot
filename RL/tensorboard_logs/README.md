# Tensorboard logs from training

| Log Name              | Description                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| dqn_grab_cloth_1_7    | Model trained to grab the unfolded cloth from a neutral position.                                                                |
| dqn_fold_1_200K_v1_4  | Statically trained model for the first fold, i.e. with grippers already on the cloth.                                            |
| dqn_fold_1_200K_v1_13 | First fold model trained on cloths already gripped using the grab cloth 1 model.                                                 |
| grab_cloth_2_3        | Second grab model trained on cloths folded using `dqn_fold_1_200K_v1_13_3` and rotated 90°.                                      |
| dqn_fold_2_200K_v1.0  | Statically trained model for the second fold, i.e. with grippers on the cloth and the cloth already neatly folded and rotated.   |
| fold_2_200k_v2_1      | Second fold models (all training results included for completeness) trained on cloths folded and grabbed using the other models. |
