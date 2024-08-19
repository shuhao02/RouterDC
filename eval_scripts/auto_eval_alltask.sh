# model_name_list=("MetaMath-Mistral-7B" "OpenHermes-2.5-Mistral-7B" "WizardMath-7B-V1.1" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "Hermes-2-Pro-Mistral-7B" "dolphin-2.2.1-mistral-7b" "mistral-7B-forest-dpo" )
# model_name_pre=("meta-math" "teknium" "WizardLM" "itpossible" "HuggingFaceH4" "cognitivecomputations" "NousResearch" "cognitivecomputations" "abhishekchohan")

# model_name_list=("MetaMath-Mistral-7B" "OpenHermes-2.5-Mistral-7B" "WizardMath-7B-V1.1" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "Hermes-2-Pro-Mistral-7B" "dolphin-2.2.1-mistral-7b" "mistral-7B-forest-dpo" "Mistral-7B-v0.1")
# model_name_pre=("meta-math" "teknium" "WizardLM" "itpossible" "HuggingFaceH4" "cognitivecomputations" "NousResearch" "cognitivecomputations" "abhishekchohan" "mistralai")

# model_name_list=("WizardMath-7B-V1.1" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "Mistral-7B-v0.1")
# model_name_pre=("WizardLM" "itpossible" "HuggingFaceH4" "cognitivecomputations" "mistralai")


#model_name_list=("Meta-Llama-3-8B" "Phi-3-mini-4k-instruct" "Llama3-8B-Chinese-Chat" "dolphin-2.9-llama3-8b" "XwinCoder-7B" "CodeLlama-13b-hf" "Llama-2-13b-hf" "Qwen1.5-7B" "gemma-7b" "falcon-7b" "CodeLlama-7b-hf" "Nous-Hermes-Llama2-13b" "Llama2-Chinese-13b-Chat" "LLaMA2-13B-Tiefighter" "openbuddy-llama2-13b-v11.1-bf16")
#model_name_pre=("meta-llama" "microsoft" "shenzhi-wang" "cognitivecomputations" "Xwin-LM" "meta-llama" "meta-llama" "Qwen" "google" "tiiuae" "meta-llama" "NousResearch" "FlagAlpha" "KoboldAI" "OpenBuddy")

model_name_list=("MetaMath-Mistral-7B" "Chinese-Mistral-7B-v0.1" "zephyr-7b-beta"  "dolphin-2.6-mistral-7b" "dolphin-2.2.1-mistral-7b" "Mistral-7B-v0.1" "Meta-Llama-3-8B" "dolphin-2.9-llama3-8b")
model_name_pre=("meta-math" "itpossible" "HuggingFaceH4" "cognitivecomputations" "NousResearch" "mistralai" "meta-llama" "cognitivecomputations")


model_index=0
while [ $model_index -lt 10 ]; do
  model_path="/data/home/chensh/data/huggingface_model/${model_name_pre[model_index]}/${model_name_list[model_index]}"
  bash eval_alltask.sh "${model_path}" "${model_name_list[model_index]}" "5"
  # bash eval_math.sh "${model_path}" "${model_name_list[model_index]}" "2"
  # bash eval_mmlu.sh "${model_path}" "${model_name_list[model_index]}" "1"
  # bash eval_humaneval.sh "${model_path}" "${model_name_list[model_index]}" "6"
#  bash eval_hellaswag.sh "${model_path}" "${model_name_list[model_index]}" "7"
  # bash eval_alltask_pruned.sh "${model_path}" "${model_name_list[model_index]}" "3" 
  # bash eval_gsm8k_pruned.sh "${model_path}" "${model_name_list[model_index]}" "3" 
  # bash eval_humaneval_pruned.sh "${model_path}" "${model_name_list[model_index]}" "2" "0.5"
  # bash eval_mbpp_pruned.sh "${model_path}" "${model_name_list[model_index]}" "7"
  # bash eval_mbpp.sh "${model_path}" "${model_name_list[model_index]}" "0"
  model_index=$((model_index + 1))
done