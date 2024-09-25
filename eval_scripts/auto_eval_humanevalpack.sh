model_name_list=("MetaMath-Mistral-7B" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "dolphin-2.2.1-mistral-7b" "Mistral-7B-v0.1" "Meta-Llama-3-8B" "dolphin-2.9-llama3-8b")
model_name_pre=("meta-math" "itpossible" "HuggingFaceH4" "cognitivecomputations" "NousResearch" "mistralai" "meta-llama" "cognitivecomputations")

model_index=2
language="js"
while [ $model_index -lt 3 ]; do
  model_path="${model_name_pre[model_index]}/${model_name_list[model_index]}"
  bash eval_humanevalpack_${language}.sh "${model_path}" "${model_name_list[model_index]}" "7"
  model_index=$((model_index + 1))
done