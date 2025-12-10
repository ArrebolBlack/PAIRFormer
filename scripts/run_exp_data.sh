for n in {1..9}; do
  python -m src.launch.train \
    experiment=miRAW_TargetNet_baseline \
    run.train_reduction=none \
    train.loss_type=bce \
    data.path.train=data/miRAW_Test_for_Train_Validation/miRAW_Test1-${n}_split-ratio-0.9_Train_Validation.txt \
    data.path.val=data/miRAW_Test_for_Train_Validation/miRAW_Test1-${n}_split-ratio-0.9_Train_Validation.txt \
    logging.wandb.group="data_recipe_1_Test_data_num"
done
