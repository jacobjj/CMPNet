python3 cmpnet/src/cmpnet_test_dubins.py \
        --model_path data/mpnet/dubins_car \
        --grad_step 1 \
        --learning_rate 0.001 \
        --memory_strength 0.5 \
        --n_memories 1 \
        --n_tasks 1 \
        --device 0 \
        --data_path data/ \
        --start_epoch 100 \
        --memory_type res \
        --env_type dubins_car \
        --world_size 2.75 \
        --total_input_size 2806 \
        --AE_input_size 2800 \
        --mlp_input_size 34 \
        --output_size 3 \
        --seen_N 4000 \
        --seen_NP 1 \
        --seen_s 0 \
        --seen_sp 4000 \
        --unseen_N 50 \
        --unseen_NP 1 \
        --unseen_s 100 \
        --unseen_sp 0
# seen: 100, 200, 0, 4000
# unseen: 10, 2000, 100, 0
