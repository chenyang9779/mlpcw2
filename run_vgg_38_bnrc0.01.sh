export PYTHONPATH=$PYTHONPATH:/home/s2693586/mlpractical

python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 5 --experiment_name VGG_38_bnrc0.001 --use_gpu True --num_classes 100 --block_type 'bnrc_block' --continue_from_epoch -1 --learning_rate 1e-2
