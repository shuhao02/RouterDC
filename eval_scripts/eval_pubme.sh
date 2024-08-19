model_name="/data/home/chensh/data/huggingface_model/mistralai/Mistral-7B-v0.1"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks pubmedqa \
    --device cuda:5 \
    --batch_size 8 \
    --output_path "/data/home/chensh/projects/LLM_router/output/pubmedqa/Mistral-7B-v0.1" \
    --log_samples

model_name="/data/home/chensh/data/huggingface_model/meta-math/MetaMath-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks pubmedqa \
    --device cuda:5 \
    --batch_size 8 \
    --output_path "/data/home/chensh/projects/LLM_router/output/pubmedqa/MetaMath-Mistral-7B" \
    --log_samples

model_name="/data/home/chensh/data/huggingface_model/HuggingFaceH4/zephyr-7b-beta"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks pubmedqa \
    --device cuda:5 \
    --batch_size 8 \
    --output_path "/data/home/chensh/projects/LLM_router/output/pubmedqa/zephyr-7b-beta" \
    --log_samples


model_name="/data/home/chensh/data/huggingface_model/teknium/OpenHermes-2.5-Mistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks pubmedqa \
    --device cuda:5 \
    --batch_size 8 \
    --output_path "/data/home/chensh/projects/LLM_router/output/pubmedqa/OpenHermes-2.5-Mistral-7B" \
    --log_samples


model_name="/data/home/chensh/data/huggingface_model/BioMistral/BioMistral-7B"

lm_eval --model hf \
    --model_args pretrained=$model_name \
    --tasks pubmedqa \
    --device cuda:5 \
    --batch_size 8 \
    --output_path "/data/home/chensh/projects/LLM_router/output/pubmedqa/BioMistral/BioMistral-7B" \
    --log_samples