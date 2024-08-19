model_path=$1
model_name=$2
device=$3

lm_eval --model hf \
      --model_args pretrained=$model_path \
      --tasks hellaswag \
      --device "cuda:${device}" \
      --batch_size 32 \
      --output_path "/data/home/chensh/projects/LLM_router/output/hellaswag_train/${model_name}" \
      -f 10 \
      --log_samples