# (NeurIPS 2024) RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models

Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang

---

Official Implementation of NeurIPS 2024 paper "[RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models](https://arxiv.org/abs/2409.19886)".

# Quick Start

## Datasets
We have provided the necessary training datasets in the [datasets](./datasets) folder.

To create your own training datasets from scratch, follow these steps:

- **Evaluate LLM Outputs:** Use EleutherAI's [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-evaluation-harness
](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#features) to evaluate each language model (LLM). This work use  and  for evaluating language model. So, the first thing to do is to configure the environment of these two libraries. First, set up the environment for these libraries. You can find the commands to generate the answers for each dataset subset in the [eval_scripts](./eval_scripts) folder.
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
