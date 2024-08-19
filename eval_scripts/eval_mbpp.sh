model_path=$1
model_name=$2
device=$3

# humaneval
save_path="/data/home/chensh/projects/LLM_router/output/mbpp/${model_name}/"
export CUDA_VISIBLE_DEVICES="${device}"

accelerate launch  /data/home/chensh/projects/bigcode-evaluation-harness/main.py \
  --model $model_path \
  --max_length_generation 2048 \
  --tasks mbpp \
  --load_data_path "/data/home/chensh/data/LLM_datasets/openai_humaneval" \
  --temperature 0.2 \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path ${save_path} \
  --metric_output_path "${save_path}metrics.json" \
  --log_sample_path "${save_path}log_samples.json"
