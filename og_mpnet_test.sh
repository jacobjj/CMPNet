python og_mpnet_test.py --model_path CMPnet_res/baxter_test_small_3_m10000_ms50_b100_f100_lr01/ \
--grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --data_path ../data/test_100/ \
--start_epoch 1 --memory_type res --env_type baxter --world_size 20 \
--total_input_size 16067 --AE_input_size 16053 --mlp_input_size 74 --output_size 7 \
--seen_N 10 --seen_NP 200 --seen_s 0 --seen_sp 4000 \
--unseen_N 10 --unseen_NP 200 --unseen_s 100 --unseen_sp 0  \
--dl1 0 --docker 0

