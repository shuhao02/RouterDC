# RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models


# Quick Start

## Datasets
We have provided the training datasets in [datasets](./datasets).

If you want to create the training datasets from scratch, please follow the listed steps.

- Evaluate LLM for each dataset with logging their output. This work use [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-evaluation-harness
](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#features) for evaluating language model. So, the first thing to do is to configure the environment of these two libraries. After that, the command for generating the answers of each subset can be found in [eval_scripts](./eval_scripts).
- Calculate the score of each LLM and merge them with the query together as the training & testing dataset. See [convert_dataset_7_model.ipynb](convert_dataset_7_model.ipynb) for further details.
- Allocate the cluster id for the training dataset. Please see [cluster_generate.ipynb](src/cluster_generate.ipynb) for further details.

## Training
See [train_scripts](train_scripts) for further details.

## Testing
During training, the model will evaluate at the evaluate_step. You can also use the [evaluation_router.py](evaluation_router.py) to evaluate a given checkpoint.