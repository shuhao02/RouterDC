model_name_list=("MetaMath-Mistral-7B" "OpenHermes-2.5-Mistral-7B" "WizardMath-7B-V1.1" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "Hermes-2-Pro-Mistral-7B" "dolphin-2.2.1-mistral-7b" "mistral-7B-forest-dpo" "Mistral-7B-v0.1")
model_name_pre=("meta-math" "teknium" "WizardLM" "itpossible" "HuggingFaceH4" "cognitivecomputations" "NousResearch" "cognitivecomputations" "abhishekchohan" "mistralai")


model_index=0
while [ $model_index -lt 1 ]; do
  model_path="/data/home/chensh/data/huggingface_model/${model_name_pre[model_index]}/${model_name_list[model_index]}"
  bash eval_math.sh "${model_path}" "${model_name_list[model_index]}" "2"
  model_index=$((model_index + 1))
done