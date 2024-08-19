export CUDA_VISIBLE_DEVICES="6"

# using cluster_loss 
training_steps=1000
learning_rate="5e-5"
tempreture=1
similarity_function="cos"
seeds=(0)

for seed in "${seeds[@]}"; do
    python -u train_lora_retriever.py --training_steps ${training_steps} --learning_rate ${learning_rate} --eval_steps 50 --tempreture ${tempreture} --similarity_function ${similarity_function} --seed ${seed} \
    --data_paths "./datasets/split2_model7_cluster/gsm8k-train.json" "./datasets/split2_model7_cluster/humaneval_train.json"  "./datasets/split2_model7_cluster/arc_challenge_train.json" "./datasets/split2_model7_cluster/cmmlu_train.json" "./datasets/split2_model7_cluster/mmlu_train.json" \
    --ref_data_paths "./datasets/lora_retriever/cluster_0.json" "./datasets/lora_retriever/cluster_1.json" "./datasets/lora_retriever/cluster_2.json" "./datasets/lora_retriever/cluster_3.json" "./datasets/lora_retriever/cluster_4.json" \
    --test_data_paths "./datasets/split2_model7/gsm8k-test.json" "./datasets/split2_model7/humaneval_test.json" "./datasets/split2_model7/arc_challenge_test.json" "./datasets/split2_model7/cmmlu_test.json" "./datasets/split2_model7/mmlu_test.json" \
    --save_path "/data/home/chensh/projects/LLM_router/logs/lora_retriever/lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}" \
    --test_data_type "multi_attempt" "multi_attempt" "probability" "probability" "probability" \
    > "./logs/lora_retriever/lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}.log" 2>&1
done