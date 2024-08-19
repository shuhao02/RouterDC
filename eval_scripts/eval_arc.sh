#model_path="/data/home/chensh/data/huggingface_model/itpossible/Chinese-Mistral-7B-v0.1"
#model_name="Chinese-Mistral-7B-v0.1"
#device="6"

model_path=$1
model_name=$2
device=$3
export CUDA_VISIBLE_DEVICES="${device}"

# arc
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks arc_easy \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/arc_easy/${model_name}" \
    --log_samples \
    --trust_remote_code

lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks arc_challenge \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/arc_challenge/${model_name}" \
    --log_samples \
    --trust_remote_code 