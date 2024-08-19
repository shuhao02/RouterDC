model_path=$1
model_name=$2
device=$3
export CUDA_VISIBLE_DEVICES="${device}"

# math
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks minerva_math_* \
    --device "cuda:${device}" \
    --batch_size auto \
    --output_path "/data/home/chensh/projects/LLM_router/output/MATH/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \