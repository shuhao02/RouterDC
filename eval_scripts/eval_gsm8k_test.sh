model_name_list=("Mistral-7B-v0.1" "MetaMath-Mistral-7B" "OpenHermes-2.5-Mistral-7B" "Code-Mistral-7B" "WizardMath-7B-V1.1" "Arithmo2-Mistral-7B")
model_name_pre=("mistralai" "meta-math" "teknium" "ajibawa-2023" "WizardLM" "upaya07")

model_index=0
while [ $model_index -lt 7 ]; do
  model_name="/data/home/chensh/data/huggingface_model/${model_name_pre[model_index]}/${model_name_list[model_index]}"

  lm_eval --model hf \
      --model_args pretrained=$model_name \
      --tasks gsm8k-repeat10 \
      --device cuda:4 \
      --batch_size 64 \
      --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_test-t0.2/${model_name_list[model_index]}" \
      --log_samples \
      --gen_kwargs "do_sample=True,temperature=0.2"
#      --limit 1000
  model_index=$((model_index + 1))
done