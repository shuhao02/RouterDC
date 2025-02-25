# (NeurIPS 2024) RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models

Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang

---
Official Implementation of NeurIPS 2024 paper "[RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models](https://arxiv.org/abs/2409.19886)".

# Abstract
Recent works show that assembling multiple off-the-shelf large language models
(LLMs) can harness their complementary abilities. To achieve this, routing is a
promising method, which learns a router to select the most suitable LLM for each
query. However, existing routing models are ineffective when multiple LLMs
perform well for a query. To address this problem, in this paper, we propose a
method called query-based Router by Dual Contrastive learning (**RouterDC**). The
RouterDC model consists of an encoder and LLM embeddings, and we propose
two contrastive learning losses to train the RouterDC model. Experimental results
show that RouterDC is effective in assembling LLMs and largely outperforms
individual top-performing LLMs as well as existing routing methods on both
in-distribution (+2.76%) and out-of-distribution (+1.90%) tasks. 

# Quick Start

## Datasets
We have provided the necessary training datasets in the [datasets](./datasets) folder.

To create your own training datasets from scratch, follow these steps:

- **Evaluate LLM Outputs:** Use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-evaluation-harness
](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#features) to evaluate each language model (LLM). To log the output of each samples, we slightly modify the [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#features) as mention in [issue](https://github.com/bigcode-project/bigcode-evaluation-harness/issues/215#issuecomment-2044445209). The commands to generate the answers for each dataset subset can be found in the [eval_scripts](./eval_scripts) folder.
- **Prepare the Dataset:** Allocate the scores for each LLM, then merge the scores with the queries to create the training and testing datasets. Detailed instructions can be found in [convert_dataset_7_model.ipynb](convert_dataset_7_model.ipynb).
- **Assign Cluster IDs:** Allocate cluster IDs for the training dataset by following the process outlined in [cluster_generate.ipynb](src/cluster_generate.ipynb).

## Training
Refer to the [train_scripts](train_scripts) folder for detailed training instructions.

## Testing
During training, the model automatically evaluates at predefined evaluation steps. 
You can also manually evaluate a specific checkpoint using [evaluation_router.py](evaluation_router.py).

## Citation
If you find RouterDC is useful for your research and applications, please cite using this BibTeX:

```
@inproceedings{chen2024RouterDC,
  title={{RouterDC}: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models},
  author={Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang},
  booktitle={Neural Information Processing Systems},
  year={2024}
}
```
