model_path=$1
model_name=$2
device=$3

export CUDA_VISIBLE_DEVICES="${device}"
# mmlu
lm_eval --model hf \
    --model_args pretrained=$model_path \
    --tasks mmlu \
    --device "cuda:${device}" \
    --batch_size 16 \
    --output_path "/data/home/chensh/projects/LLM_router/output/mmlu_5shot/${model_name}" \
    --log_samples \
    --num_fewshot 5