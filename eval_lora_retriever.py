# the code is prepare for cluser_id.


import argparse
import json
import os
import random

import torch.nn as nn
import torch.optim
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter

from train_lora_retriever import RouterDataset, RouterModule, evaluation


if __name__ == '__main__': 
    device = "cuda"
    parser = argparse.ArgumentParser(description="the training code for router")
    parser.add_argument('--test_data_paths',nargs='+', default=["./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/MATH_prealgebra.json", "./datasets/split2_model7/mbpp.json", "./datasets/split2_model7/ceval.json" ,"./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json",  "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--ref_data_paths', nargs='+', default=["./datasets/lora_retriever/cluster_0.json","./datasets/lora_retriever/cluster_1.json", "./datasets/lora_retriever/cluster_2.json", "./datasets/lora_retriever/cluster_3.json","./datasets/lora_retriever/cluster_4.json",])
    parser.add_argument('--test_data_type', nargs='+', default=["probability", "probability", "multi_attempt","probability", "multi_attempt", "multi_attempt", "probability",  "probability"])
    parser.add_argument('--trained_router_path', default="/data/home/chensh/projects/LLM_router/logs/router_debug/ablation_H/H_5_slw_0_clw_1_clw_2_0_cos_tk_3_lk_3_lr_5e-5_step_1000_t_1_seed_5/best_training_model.pth")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    # get router model (flan-t5-encoder)
    tokenizer = AutoTokenizer.from_pretrained("/data/home/chensh/data/huggingface_model/microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained("/data/home/chensh/data/huggingface_model/microsoft/mdeberta-v3-base")
    
    router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=7, similarity_function="cos").to(device)

    state_dict = torch.load(args.trained_router_path)
    router_model.load_state_dict(state_dict)

    cluster_model_map = [3, 6, 5, 5, 4]

    print("test start")
    test_result = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, batch_size=32, device="cuda", ref_data_path=args.ref_data_paths, cluster_model_map=cluster_model_map)

    output_order = ['mmlu', 'gsm8k', 'cmmlu', 'arc', 'humaneval', 'MATH', 'mbpp', 'ceval']
    key_list = list(test_result.keys())
    key_order = []
    for key_candidate in output_order:
        for key in key_list:
            if key_candidate in key:
                key_order.append(key)
                break
    for key in key_order:
        print(f"{test_result[key][1] * 100}", end=' ')

    