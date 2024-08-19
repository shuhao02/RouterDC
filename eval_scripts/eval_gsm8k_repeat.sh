model_name="/data/home/chensh/data/huggingface_model/mistralai/Mistral-7B-v0.1"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/Mistral-7B-v0.1" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000


model_name="/data/home/chensh/data/huggingface_model/meta-math/MetaMath-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/MetaMath-Mistral-7B" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000

model_name="/data/home/chensh/data/huggingface_model/HuggingFaceH4/zephyr-7b-beta"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/zephyr-7b-beta" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000

model_name="/data/home/chensh/data/huggingface_model/teknium/OpenHermes-2.5-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/OpenHermes-2.5-Mistral-7B" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000


model_name="/data/home/chensh/data/huggingface_model/ajibawa-2023/Code-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/Code-Mistral-7B" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000

model_name="/data/home/chensh/data/huggingface_model/WizardLM/WizardMath-7B-V1.1"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/WizardMath-7B-V1.1" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000


model_name="/data/home/chensh/data/huggingface_model/upaya07/Arithmo2-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/Arithmo2-Mistral-7B" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000


model_name="/data/home/chensh/data/huggingface_model/Nondzu/Mistral-7B-code-16k-qlora"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks gsm8k_train \
    --device cuda:6 \
    --batch_size 64 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/Mistral-7B-code-16k-qlora" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000





#model_name="/data/home/chensh/data/huggingface_model/BioMistral/BioMistral-7B"
#lm_eval --model hf \
#    --model_args pretrained=$model_name \
#    --tasks gsm8k_train \
#    --device cuda:6 \
#    --batch_size 64\
#    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/BioMistral-7B" \
#    --log_samples \
#    --gen_kwargs "do_sample=True,temperature=0.2" \
#    --limit 1000



#CUDA_VISIBLE_DEVICES=7 nohup python prune_model.py --finetuned_model "/data/home/chensh/data/huggingface_model/meta-math/MetaMath-Mistral-7B" --prune_rate 0.01 --log_path meta_math --task gsm8k &
#
#CUDA_VISIBLE_DEVICES=6 nohup python prune_model.py --finetuned_model "/data/home/chensh/data/huggingface_model/meta-math/MetaMath-Mistral-7B" --prune_rate 0.001 --log_path meta_math --task gsm8k &