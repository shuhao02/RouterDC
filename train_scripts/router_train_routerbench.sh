export CUDA_VISIBLE_DEVICES="0"

# using cluster_loss 
top_ks=(4)
last_ks=(4)
training_steps=1000
learning_rate="5e-5"
tempreture=1
similarity_function="cos"
sample_loss_weight=0
cluster_loss_weight=1
seeds=(7)
cost_rates=(0 1)

for cost_rate in "${cost_rates[@]}"; do
    for top_k in "${top_ks[@]}"; do
        for last_k in "${last_ks[@]}"; do
            for seed in "${seeds[@]}"; do
                python -u train_router_mdeberta_routerbench.py --training_steps ${training_steps} --top_k ${top_k} --last_k ${last_k} --learning_rate ${learning_rate} --eval_steps 100 --tempreture ${tempreture} --similarity_function ${similarity_function} --sample_loss_weight ${sample_loss_weight} --cluster_loss_weight ${cluster_loss_weight} --seed ${seed} --final_eval --cost_rate ${cost_rate} \
                --data_paths './datasets/routerbench_cluster/gsm8k_train.csv' './datasets/routerbench_cluster/hellaswag_train.csv' './datasets/routerbench_cluster/mbpp_train.csv' './datasets/routerbench_cluster/mmlu_train.csv' './datasets/routerbench_cluster/winograde_train.csv' './datasets/routerbench_cluster/arc_challenge_train.csv' \
                --test_data_paths './datasets/routerbench_zs/gsm8k_test.csv' './datasets/routerbench_zs/hellaswag_test.csv' './datasets/routerbench_zs/mbpp_test.csv' './datasets/routerbench_zs/mmlu_test.csv' './datasets/routerbench_zs/winograde_test.csv' './datasets/routerbench_zs/arc_challenge_test.csv' \
                --save_path "/data/home/chensh/projects/LLM_router/logs/paper_result/routerbench/cr_${cost_rate}_slw_${sample_loss_weight}_tk_${top_k}_lk_${last_k}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}" \
                > "./logs/paper_result/routerbench/cr_${cost_rate}_slw_${sample_loss_weight}_tk_${top_k}_lk_${last_k}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}.log" 2>&1
            done
        done
    done
done
