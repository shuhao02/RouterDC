#model_path="/data/home/chensh/data/huggingface_model/itpossible/Chinese-Mistral-7B-v0.1"
#model_name="Chinese-Mistral-7B-v0.1"
#device="6"

model_path=$1
model_name=$2
device=$3
export CUDA_VISIBLE_DEVICES="${device}"

# # gsm8k
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks gsm8k_train \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_train-t0.2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \
    --limit 1000

lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks gsm8k-repeat10 \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/gsm8k_test-t0.2/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2"

# MATH
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks minerva_math_* \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/MATH/${model_name}" \
    --log_samples \
    --gen_kwargs "do_sample=True,temperature=0.2" \

# hellaswag
lm_eval --model vllm \
      --model_args pretrained=$model_path \
      --tasks hellaswag \
      --device "cuda:${device}" \
      --batch_size 32 \
      --output_path "/data/home/chensh/projects/LLM_router/output/hellaswag_validation/${model_name}" \
      -f 10 \
      --log_samples

# mmlu
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks mmlu \
    --device "cuda:${device}" \
    --batch_size auto \
    --output_path "/data/home/chensh/projects/LLM_router/output/mmlu_5shot/${model_name}" \
    --log_samples \
    --num_fewshot 5

# arc
lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks arc_easy \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/arc_easy/${model_name}" \
    --log_samples

lm_eval --model vllm \
    --model_args pretrained=$model_path \
    --tasks arc_challenge \
    --device "cuda:${device}" \
    --batch_size 32 \
    --output_path "/data/home/chensh/projects/LLM_router/output/arc_challenge/${model_name}" \
    --log_samples

# humaneval
save_path="/data/home/chensh/projects/LLM_router/output/humaneval/${model_name}/"

accelerate launch  /data/home/chensh/projects/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 512 \
  --tasks humaneval \
  --load_data_path "/data/home/chensh/data/LLM_datasets/openai_humaneval" \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json" \
  --log_sample_path "${save_path}log_samples.json"

# MBPP
save_path="/data/home/chensh/projects/LLM_router/output/mbpp/${model_name}/"

accelerate launch  /data/home/chensh/projects/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 2048 \
  --tasks mbpp \
  --load_data_path "/data/home/chensh/data/LLM_datasets/openai_humaneval" \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 10 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json" \
  --log_sample_path "${save_path}log_samples.json"



# ceval
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks ceval-valid \
    --device "cuda:${device}" \
    --batch_size auto \
    --output_path "/data/home/chensh/projects/LLM_router/output/ceval-validation/${model_name}" \
    --log_samples \
    --num_fewshot 5

# cmmlu
lm_eval --model vllm \
    --model_args "pretrained=$model_path,max_model_len=4096" \
    --tasks cmmlu \
    --device "cuda:${device}" \
    --batch_size auto \
    --output_path "/data/home/chensh/projects/LLM_router/output/cmmlu/${model_name}" \
    --log_samples \
    --num_fewshot 5

# humaneval_js

save_path="/data/home/chensh/projects/LLM_router/output/humanevalpack_java/${model_name}/"

accelerate launch  /data/home/chensh/projects/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 512 \
  --tasks humanevalsynthesize-js \
  --load_data_path "/data/home/chensh/data/LLM_datasets/bigcode/humanevalpack" \
  --prompt continue \
  --temperature 0.2 \
  --n_samples 10 \
  --batch_size 30 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json" \
  --log_sample_path "${save_path}log_samples.json"