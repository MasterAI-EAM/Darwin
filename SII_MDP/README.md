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

To run on more gpus, you may prefer to turn down `gradient_accumulation_steps` to keep a global batch size of 128. Global batch size has not been tested for optimality.


## **Authors**

This project is a collaborative effort by the following:

UNSW: Tong Xie, Shaozhou Wang, Qingyuan Linghu, Wei Huang, Wenjie Zhang, Bram Hoex

CityU HK: Yuwei Wan, Yufei Zhou, Chunyu Kit

University of Sydney: Clara Grazian

GreenDynamics: Yixuan Liu

All advised by Bram Hoex from UNSW Engineering

## **Citation**

If you use the data or code from this repository in your work, please cite it accordingly.

## **Acknowledgements**

This project has referred to the following open-source projects:

- Meta LLaMA: **[LLaMA](https://github.com/facebookresearch/llama)**
- Stanford Alpaca: **[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)**
