# End-2-End learning (randomly shuffle path)
python3 cmpnet/src/mpnet_train_dubins.py \
        --model_path data/mpnet/dubins_car \
        --no_env 30000  \
        --no_motion_paths 1 \
        --grad_step 1 \
        --learning_rate 0.01 \
        --num_epochs 100 \
        --memory_strength 0.5 \
        --n_memories 10000 \
        --n_tasks 1 \
        --device 0 \
        --freq_rehersal 100 \
        --batch_rehersal 100 \
        --start_epoch 0 \
        --data_path data/ \
        --env_type dubins_car \
        --memory_type res \
        --total_input_size 2806 \
        --AE_input_size 2800 \
        --mlp_input_size 34 \
        --output_size 3