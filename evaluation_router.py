import argparse
import random

import torch.optim
from transformers import AutoTokenizer, DebertaV2Model

from train_router_mdeberta import RouterModule, evaluation


if __name__ == '__main__': 
    device = "cuda"
    parser = argparse.ArgumentParser(description="the training code for router")
    parser.add_argument('--test_data_paths',nargs='+', default=["./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/MATH_prealgebra.json", "./datasets/split2_model7/mbpp.json", "./datasets/split2_model7/ceval.json" ,"./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json",  "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--test_data_type', nargs='+', default=["probability", "probability", "multi_attempt","probability", "multi_attempt", "multi_attempt", "probability",  "probability"])
    parser.add_argument('--trained_router_path', default="./logs/paper_result/ablation_zero_score/H_3_slw_0_clw_1_clw_2_0_cos_tk_3_lk_3_lr_5e-5_step_1000_t_1_seed_5/best_training_model.pth")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    # get router model (flan-t5-encoder)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base")
    
    router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=7, similarity_function="cos").to(device)

    state_dict = torch.load(args.trained_router_path)
    router_model.load_state_dict(state_dict)

    print("test start")
    test_result = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, batch_size=32, device="cuda")
    print(test_result)

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

    