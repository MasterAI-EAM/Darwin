This is the directory for paper "LARGE LANGUAGE MODELS AS MASTER KEY: UNLOCKING THE SECRETS OF MATERIALS SCIENCE", which presents a new natural language processing (NLP) task called structured information inference (SII) to address the complexities of information extraction at the device level in material science. This project is part of the whole DARWIN plan. The original data of this project is from https://www.nature.com/articles/s41560-021-00941-3.

The instructions of code:

- data
  - regression360.json: train dataset of material & device prediction (MDP) regression task
  - sii360.json: train dataset of SII task
  - regression40.json: test dataset of material & device prediction (MDP) regression task
  - sii40.json: test dataset of SII task
  - original_text.json: original text of 40 papers in SII test dataset
- [train](https://github.com/MasterAI-EAM/Darwin/blob/main/train.py): code for training LLaMA-7B (outside in main directory)
- sii_test.py: code for running test of SII task
- regression_test.py: code for runing test of MDP regression task.
- sii_evaluate.ipynb: code for evaluating SII results.
- regression_evaluate.py: code for evaluating MDP regression results.


## Data Format
sii360.json/regression360.json/sii40.json/regression40.json is JSON file containing a list of dictionaries, and each dictionary contains the following fields:
- `instruction`: `str`, describes the task the model should perform. For SII, we use "Summarize stack and method information from given paragraph about solar cell". For MDP, we use "What's the PCE of the perovskite solar cell with the parameters below".
- `input`: `str`, input for the task. For SII, input is original text of paper. For MDP, input is schema with corresponding values.
- `output`: `str`, the answer to the instruction. For SII, answer is schema with corresponding values. For MDP, answer is PCE value (and Voc, Jsc, FF)

## Getting Started

First install the requirements in the main directory:

```bash
pip install -r requirements.txt
```

Then download the checkpoints of the open-source LLaMA-7B weights from huggingface. 

## Fine-tuning
To fine-tune LLaMA-7b with SII/MDP datasets, below is a command that works on a machine with 4 A100 80G GPUs in FSDP `full_shard` mode.
Replace `<your_random_port>` with a port of your own, `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` with the
path to your converted checkpoint and tokenizer, and `<your_output_dir>` with where you want to store your outputs.
```bash
torchrun  --nproc_per_node=8 --master_port=<your_random_port> train.py \
    --model_name_or_path <your path to LLaMA-7b> \
    --data_path <your path to dataset> \
    --bf16 True \
    --output_dir <your output dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 False
```

## Fine-tuning

We fine-tune our models using standard Hugging Face training code.
We fine-tune LLaMA-7B and LLaMA-13B with the following hyperparameters:

| Hyperparameter | LLaMA-7B | LLaMA-13B |
|----------------|----------|-----------|
| Batch size     | 128      | 128       |
| Learning rate  | 2e-5     | 1e-5      |
| Epochs         | 3        | 5         |
| Max length     | 512      | 512       |
| Weight decay   | 0        | 0         |

To reproduce our fine-tuning runs for LLaMA, first install the requirements

```bash
pip install -r requirements.txt
```

Below is a command that fine-tunes LLaMA-7B with our dataset on a machine with 4 A100 80G GPUs in FSDP `full_shard` mode.
We were able to reproduce a model of similar quality as the one we hosted in our demo with the following command using **Python 3.10**.
Replace `<your_random_port>` with a port of your own, `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` with the
path to your converted checkpoint and tokenizer (following instructions in the PR), and `<your_output_dir>` with where you want to store your outputs.

```bash
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

The same script also works for OPT fine-tuning. Here's an example for fine-tuning OPT-6.7B

```bash
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path "facebook/opt-6.7b" \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    --tf32 True
```

Note the given training script is meant to be simple and easy to use, and is not particularly optimized.
To run on more gpus, you may prefer to turn down `gradient_accumulation_steps` to keep a global batch size of 128. Global batch size has not been tested for optimality.

### Addressing OOM

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. Commands given above enable parameter sharding, so no redundant model copy is stored on any GPU.
If you'd like to further reduce the memory footprint, here are some options:

- Turn on CPU offload for FSDP with `--fsdp "full_shard auto_wrap offload"`. This saves VRAM at the cost of longer runtime.
- In our experience, DeepSpeed stage-3 (with offload) can at times be more memory efficient than FSDP with offload. Here's an example to use DeepSpeed stage-3 with 4 GPUs with both parameter and optimizer offload:
    ```bash
    pip install deepspeed
    torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
        --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --output_dir <your_output_dir> \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --deepspeed "./configs/default_offload_opt_param.json" \
        --tf32 True
    ```
  - The DeepSpeed library also provides some [helpful functions](https://deepspeed.readthedocs.io/en/latest/memory.html) to estimate memory usage. 
- [LoRA](https://arxiv.org/abs/2106.09685) fine-tunes low-rank slices of the query, key, and value embedding heads. This can reduce the total memory footprint from 112GB to about 7x4=28GB. We may release our re-implemention of this in the future, but for now the [peft](https://github.com/huggingface/peft) codebase can be a useful resource.

## Recovering Alpaca Weights

The weight diff between Alpaca-7B and LLaMA-7B is located [here](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff/tree/main).
To recover the original Alpaca-7B weights, follow these steps:
```text
1. Convert Meta's released weights into huggingface format. Follow this guide:
    https://huggingface.co/docs/transformers/main/model_doc/llama
2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
    https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
3. Run this function with the correct paths. E.g.,
    python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir> --path_tuned <path_to_store_recovered_weights>
```

Once step 3 completes, you should have a directory with the recovered weights, from which you can load the model like the following

```python
import transformers
alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("<path_to_store_recovered_weights>")
alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("<path_to_store_recovered_weights>")
```

### Authors

All grad students below contributed equally and the order is determined by random draw.

- [Rohan Taori](https://www.rohantaori.com/)
- [Ishaan Gulrajani](https://ishaan.io/)
- [Tianyi Zhang](https://tiiiger.github.io/)
- [Yann Dubois](https://yanndubs.github.io/)
- [Xuechen Li](https://www.lxuechen.com/)

All advised by [Tatsunori B. Hashimoto](https://thashim.github.io/). Yann is also advised by [Percy Liang](https://cs.stanford.edu/~pliang/) and Xuechen is also advised by [Carlos Guestrin](https://guestrin.su.domains/).

### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```

Naturally, you should also cite the original LLaMA paper [1] and the Self-Instruct paper [2].

### Acknowledgements

We thank Yizhong Wang for his help in explaining the data generation pipeline in Self-Instruct and providing the code for the parse analysis plot.
We thank Yifan Mai for helpful support, and members of the Stanford NLP Group as well as the Center for Research on Foundation Models (CRFM) for their helpful feedback.
