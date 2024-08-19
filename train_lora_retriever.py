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
from collections import Counter
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, DebertaV2Model
from utils.meters import AverageMeter
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子


class RouterDataset(Dataset):
    def __init__(self,
                 data_path,
                 source_max_token_len: int = 512,
                 target_max_token_len: int = 512,
                 size: int = None,
                 data_type: str = "multi_attempt",
                 dataset_id = 0,
                 ):
        with open(data_path, 'r') as f:
            if data_path.endswith('.json'):
                self.data = json.load(f)
        if size:
            while(len(self.data) < size):
                self.data.extend(self.data)
            self.data = self.data[:size]
        self.router_node = list(self.data[0]['scores'].keys())
        self.tokenizer = None
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.data_type = data_type
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        data_point = self.data[index]
        # if 'ajibawa-2023/Code-Mistral-7B' in data_point['scores']:
        #     data_point['scores'].pop('ajibawa-2023/Code-Mistral-7B')
        scores = torch.tensor(list(data_point['scores'].values()))
        # Instruction in LoRaretriever
        question = "Represent the sentence for similar task retrieval" + data_point['question']
        question_id = self.tokenizer(
            question,
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        question_id['input_ids'] = question_id.input_ids.flatten()
        question_id['attention_mask'] = question_id.attention_mask.flatten()
        cluster_id = data_point['cluster_id'] if "cluster_id" in data_point else 0
        return question_id, scores, self.dataset_id, cluster_id

    def __len__(self):
        return len(self.data)

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer



# using inner product first
class RouterModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function = "cos"):
        super(RouterModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function


    def set_initial_embeddings(self, data_loader):
        all_embeddings = torch.zeros_like(self.embeddings.weight)
        with torch.no_grad():
            for step, batch in tqdm(enumerate(data_loader)):
                inputs, scores, dataset_ids, _ = batch
                inputs = inputs.to(device)
                scores = scores.to(device)
                dataset_ids = dataset_ids.to(device)
                _, hidden_states = self.forward(**inputs)
                temp_embeddings = []
                for i in range(self.node_size):
                    mask_node_i = torch.where(scores[:, i] > 0.5, 1, 0)
                    embeddings_i = torch.sum(hidden_states * mask_node_i.unsqueeze(1), dim=0) / torch.sum(mask_node_i)
                    temp_embeddings.append(embeddings_i)
                temp_embeddings = torch.stack(temp_embeddings)
                all_embeddings = all_embeddings * (step/(step+1)) + temp_embeddings * (1/(step+1))
                if step > 50:
                    break
            self.embeddings.weight = torch.nn.Parameter(all_embeddings)
            

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    '''The forward function pass the input to t5 and compute the similarity between model output and trainable embedding'''
    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            # If positive model does not well, skip this.
            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # make the last_x ignore the true items
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)
            
            # if the last_x all false, compute the loss.
            # mask2 = torch.where(torch.sum(last_index_true, dim=1).view(-1,1) < 0.3 * last_k, 1, 0)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            # * mask2
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. dataset_id:", dataset_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        similar_score = similar_score / t
        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_orthogonal_regular_loss(self):
        embedding_similarity = self.compute_similarity(self.embeddings.weight, self.embeddings.weight)
        loss = torch.norm(embedding_similarity - torch.eye(self.node_size).type_as(embedding_similarity))
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        similar_score = similar_score / t
        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
# evaluation the router with dataset. 
def evaluation(router_model, dataset_paths, dataset_types, tokenizer, batch_size, device, ref_data_path, cluster_model_map):   
    # get the embedding of each cluster
    
    result = {}
    with torch.no_grad():
        # get the embeddings for each cluster
        cluster_embeddings = []
        for index, data_path in enumerate(ref_data_path):
            test_dataset = RouterDataset(data_path=data_path)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            temp_embeddings = []
            for i, batch in enumerate(data_loader):
                if i > 5:
                    break
                inputs, scores, _, _ = batch
                inputs = inputs.to(device)
                scores = scores.to(device)
                _, hidden_state = router_model.forward(**inputs)
                temp_embeddings.append(hidden_state)
            temp_embeddings = torch.concat(temp_embeddings, dim=0)
            cluster_embedding = torch.mean(temp_embeddings, dim=0)
            cluster_embeddings.append(cluster_embedding)
        cluster_embeddings = torch.stack(cluster_embeddings)
        

        assert len(dataset_paths) == len(dataset_types)
        for index, data_path in enumerate(dataset_paths):
            test_dataset = RouterDataset(data_path=data_path)
            test_dataset.register_tokenizer(tokenizer)
            data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            correct_predict = 0
            correct = 0
            for batch in data_loader:
                inputs, scores, _, _ = batch
                inputs = inputs.to(device)
                scores = scores.to(device)
                _, hidden_state = router_model.forward(**inputs)
                x = router_model.compute_similarity(hidden_state, cluster_embeddings)
                softmax_x = nn.Softmax(dim=1)(x)
                _, max_index = torch.max(softmax_x, dim=1)
                
                cluster_model_map = torch.tensor(cluster_model_map).type_as(max_index)
                maped_max_index = torch.gather(cluster_model_map, dim=0, index=max_index)

                _, target_max_index = torch.max(scores, dim=1)
                equals = maped_max_index.eq(target_max_index)
                correct += equals.sum().item()

                if dataset_types[index] == "probability":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, maped_max_index.unsqueeze(1), 1)
                    scores[scores > 0] = 1
                    correct_predict += (scores * mask).sum().item()
                elif dataset_types[index] == "multi_attempt":
                    mask = torch.zeros_like(scores)
                    mask = mask.scatter_(1, maped_max_index.unsqueeze(1), 1)
                    correct_predict += (scores * mask).sum().item()

            acc_predict = correct_predict/len(test_dataset)
            acc = correct/len(test_dataset)
            print(f"acc_{data_path}:", acc_predict)
            print("acc", acc)
            result[data_path] = [acc, acc_predict]
    return result


if __name__ == '__main__': 
    device = "cuda"
    parser = argparse.ArgumentParser(description="the training code for router")
    parser.add_argument('--data_paths', nargs='+', default=["./datasets/split2_model7_cluster/gsm8k-train.json","./datasets/split2_model7_cluster/humaneval_train.json", "./datasets/split2_model7_cluster/arc_challenge_train.json", "./datasets/split2_model7_cluster/mmlu_train.json","./datasets/split2_model7_cluster/cmmlu_train.json",])
    parser.add_argument('--ref_data_paths', nargs='+', default=["./datasets/lora_retriever/cluster_0.json","./datasets/lora_retriever/cluster_1.json", "./datasets/lora_retriever/cluster_2.json", "./datasets/lora_retriever/cluster_3.json","./datasets/lora_retriever/cluster_4.json",])
    parser.add_argument('--test_data_paths',nargs='+', default=["./datasets/split2_model7/gsm8k-test.json", "./datasets/split2_model7/humaneval_test.json", "./datasets/split2_model7/arc_challenge_test.json", "./datasets/split2_model7/mmlu_test.json", "./datasets/split2_model7/cmmlu_test.json"])
    parser.add_argument('--test_data_type', nargs='+', default=["multi_attempt", "multi_attempt", "probability", "probability", "probability"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--training_steps', type=int, default=100)
    parser.add_argument('--eval_steps',type=int,default=10)
    parser.add_argument('--learning_rate', type=float, default=0.00005)
    parser.add_argument('--save_path', type=str, default='./logs/router_debug/')
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--last_k',type=int, default=3)
    parser.add_argument('--tempreture', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--similarity_function', type=str, default='cos')
    parser.add_argument('--sample_loss_weight', type=float, default=0)
    parser.add_argument('--regular_loss_weight', type=float, default=0)
    parser.add_argument('--cluster_loss_weight', type=float, default=0)
    parser.add_argument('--H', type=int, default=3)
    parser.add_argument('--set_initial', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    setup_seed(args.seed)

    # get router model (flan-t5-encoder)
    tokenizer = AutoTokenizer.from_pretrained("/data/home/chensh/data/huggingface_model/microsoft/mdeberta-v3-base", truncation_side='left', padding=True)
    encoder_model = DebertaV2Model.from_pretrained("/data/home/chensh/data/huggingface_model/microsoft/mdeberta-v3-base")


    # get the training data (x, y)

    # Compute the best performed LLM for each cluster, then select the corresponding LLM for each cluster.
    cluster_counters = {}
    for data_path in args.data_paths:
        with open(data_path, "r") as f:
            dataset_file = json.load(f)
        for data_item in dataset_file:
            if data_item['cluster_id'] in cluster_counters:
                for key, value in data_item['scores'].items():
                    if value > 0:
                        cluster_counters[data_item['cluster_id']][key] += 1
            else:
                cluster_counters[ data_item['cluster_id']] = Counter()

    model_list = ['mistralai/Mistral-7B-v0.1',
                'meta-math/MetaMath-Mistral-7B',
                'itpossible/Chinese-Mistral-7B-v0.1',
                "HuggingFaceH4/zephyr-7b-beta",
                "cognitivecomputations/dolphin-2.6-mistral-7b",
                "meta-llama/Meta-Llama-3-8B",
                "cognitivecomputations/dolphin-2.9-llama3-8b"]
    
    cluster_model_map = []
    for i in range(5):
        suitable_model_name, _ = cluster_counters[i].most_common()[0]
        suitable_model_index = model_list.index(suitable_model_name)
        cluster_model_map.append(suitable_model_index)
    
    router_datasets = [RouterDataset(data_path, size=1000, data_type=args.test_data_type[i], dataset_id = i) for i, data_path in enumerate(args.data_paths)]
    for router_dataset in router_datasets:
        router_dataset.register_tokenizer(tokenizer)
    router_dataset = ConcatDataset(router_datasets)

    print(f"init_model, router_node size: {router_datasets[0].router_node}")
    router_model = RouterModule(encoder_model, hidden_state_dim=768, node_size=len(router_datasets[0].router_node), similarity_function=args.similarity_function).to(device)
    
    if args.set_initial:
        router_dataloader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=True)
        router_model.set_initial_embeddings(router_dataloader)

    # get the optimizer (AdamW)
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=args.learning_rate)

    # start training
    print("Training start!!!")
    pbar = tqdm(range(args.training_steps))
    step = 0
    training_log = []
    max_average = 0
    max_training_average = 0

    while(True):
        losses = AverageMeter('Loss', ':3.2f')
        data_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=True)
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, scores, dataset_ids, cluster_ids = batch
            inputs = inputs.to(device)
            scores = scores.to(device)
            dataset_ids = dataset_ids.to(device)
            cluster_ids = cluster_ids.to(device)
            x, hidden_state = router_model.forward(t=args.tempreture, **inputs)
            loss = router_model.compute_cluster_loss(hidden_state=hidden_state, cluster_ids=cluster_ids, t=args.tempreture, H=args.H)

            losses.update(loss.item(), scores.size(0))
            loss.backward()
            if step % args.gradient_accumulation == 0:   
                optimizer.step()

            pbar.set_postfix({"step": f"{step}","loss": loss.item()})
            pbar.write(f"step:{step}, loss:{loss.item()}")
            # print(f"step:{step}, loss:{loss.item()}")
            pbar.update(1)
            step += 1
            if step >= args.training_steps:
                break
            if (step + 1) % args.eval_steps == 0:
                print("validation start")
                # data_paths =  ["./datasets/gsm8k-train.json", "./datasets/mmlu_validation.json", "./datasets/humaneval_train.json", "./datasets/arc_easy.json", "./datasets/cmmlu_train.json"]
                val_result = evaluation(router_model, args.data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device, ref_data_path=args.ref_data_paths, cluster_model_map=cluster_model_map)
                print("test start")
                # test_data_paths = ["./datasets/split2/gsm8k-test.json", "./datasets/split2/humaneval_test.json", "./datasets/split2/arc_challenge_test.json", "./datasets/split2/cmmlu_test.json"]
                # test_data_type = ["multi_attempt", "multi_attempt", "probability", "probability"]
                test_result = evaluation(router_model, args.test_data_paths, args.test_data_type, tokenizer, batch_size = args.batch_size, device=device, ref_data_path=args.ref_data_paths, cluster_model_map=cluster_model_map)
                result = {**val_result, **test_result}
                average = sum([ value[1] for value in test_result.values()]) / len(test_result)
                print("average testing", average)
                if average > max_average:
                    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "best_model.pth"))
                    max_average = average
                training_log.append(result)
                training_average = sum([ value[1] for value in val_result.values()]) / len(test_result)
                print("average training", training_average)
                if training_average > max_training_average:
                    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "best_training_model.pth"))
                    max_training_average = training_average
                
        # pbar.write(f"step:{step}, avg_loss_per_epoch:{losses.avg}")
        print(f"step:{step}, avg_loss_per_epoch:{losses.avg}")
        if step >= args.training_steps:
            break

    # save the model
    with open(os.path.join(args.save_path, "training_log.json"), 'w') as f:
        json.dump(training_log, f)

    with open(os.path.join(args.save_path, "config.txt"), 'w') as f:
        f.write(str(args))

    torch.save(router_model.state_dict(),  os.path.join(args.save_path, "model.pth"))