model_path=/data/home/chensh/data/huggingface_model/Qwen/Qwen1.5-7B
model_name=Qwen1.5-7B
device=4
export CUDA_VISIBLE_DEVICES="${device}"

lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks gsm8k_train \
    --device "cuda:${device}" \
    --batch_size auto \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000