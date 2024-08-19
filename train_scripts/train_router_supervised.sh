export CUDA_VISIBLE_DEVICES="3"

# using cluster_loss
cluster_loss_weight=0
training_steps=1000
learning_rate="5e-5"
tempreture=0
similarity_function="dot"
seeds=(6)

for seed in "${seeds[@]}"; do
    python -u train_supervised.py --training_steps ${training_steps} --learning_rate ${learning_rate} --eval_steps 20 --tempreture ${tempreture} --similarity_function ${similarity_function} --final_eval --seed ${seed} --cluster_loss_weight ${cluster_loss_weight} \
    --data_paths "./datasets/split2_model7_cluster/gsm8k-train.json" "./datasets/split2_model7_cluster/humaneval_train.json"  "./datasets/split2_model7_cluster/arc_challenge_train.json" "./datasets/split2_model7_cluster/cmmlu_train.json" "./datasets/split2_model7_cluster/mmlu_train.json" \
    --test_data_paths "./datasets/split2_model7/gsm8k-test.json" "./datasets/split2_model7/humaneval_test.json" "./datasets/split2_model7/arc_challenge_test.json" "./datasets/split2_model7/cmmlu_test.json" "./datasets/split2_model7/mmlu_test.json" \
    --save_path "/data/home/chensh/projects/LLM_router/logs/paper_result/supervised/${similarity_function}_clw_${cluster_loss_weight}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}" \
    --test_data_type "multi_attempt" "multi_attempt" "probability" "probability" "probability" \
    > "./logs/paper_result/supervised/${similarity_function}_clw_${cluster_loss_weight}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}.log" 2>&1
done