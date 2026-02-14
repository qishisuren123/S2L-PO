# S2L-PO: Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO

![Framework](https://github.com/qishisuren123/S2L-PO/blob/main/method.png)

## Overview

S2L-PO (Small-to-Large Policy Optimization) is a novel framework that leverages smaller models as natural explorers to enhance rollout diversity in Group Relative Policy Optimization (GRPO) for large language models. Unlike traditional token-level perturbations (e.g., temperature scaling), S2L-PO introduces **policy-level diversity** through parameter-level compression, providing temporally consistent and structured exploration signals.

### Key Features

- **Policy-Level Diversity**: Utilizes smaller models' inherent diversity stemming from parameter compression rather than token-level randomness
- **Progressive Annealing**: Smoothly transitions from small-model exploration to on-policy learning, balancing exploration and exploitation
- **Computational Efficiency**: Reduces rollout compute by offloading generation to smaller models while achieving superior performance
- **Easy Integration**: Seamlessly compatible with existing GRPO implementations

### Main Results

- **+8.8% improvement** on AIME 2024 (using 1.7B explorer to guide 8B model)
- **Faster convergence** with fewer effective training steps
- **Better sample efficiency** across mathematical reasoning benchmarks
- **Maintained generalization** on out-of-domain tasks (CommonsenseQA)

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Training](#3-training)
  - [4. Evaluation](#4-evaluation)
- [Experimental Settings](#experimental-settings)
- [Reproducing Results](#reproducing-results)
- [Citation](#citation)
- [License](#license)

## Installation

S2L-PO is built on top of the [verl](https://github.com/volcengine/verl) framework. Please follow verl's installation instructions:


## Project Structure

```
S2L-PO/
├── examples/
│   └── s2l-po/
│       └── off2on_4b_8b_16_8.sh       # Main training script
├── verl/
│   └── examples/
│       └── run_small_model_sampling.sh # Data preparation script
├── eval/                              # Evaluation framework
│   ├── data/                         # Benchmark datasets
│   │   ├── aime24.jsonl
│   │   ├── aime25.jsonl
│   │   └── math500.jsonl
│   ├── output/                       # Model outputs
│   ├── result/                       # Evaluation results
│   ├── inference.py                  # Inference script
│   ├── extract_answers.py            # Answer extraction
│   ├── scoring.py                    # Scoring script
│   └── run_evaluation.py             # Main evaluation script
└── README.md                         # This file
```

## Quick Start

### 1. Environment Setup

Ensure you have the required dependencies installed as described in the [Installation](#installation) section.

### 2. Data Preparation

Generate rollouts from the smaller model for offline exploration:

```bash
cd S2L-PO/verl/examples
bash run_small_model_sampling.sh
```

This script will:
- Load the smaller model (e.g., Qwen3-1.7B or Qwen3-4B)
- Generate diverse rollouts on the training dataset
- Save rollouts for use in the progressive training phase

**Configuration**: Edit the script to specify:
- `--model_path`: Path to the small model
- `--dataset`: Training dataset (default: DAPO17k)
- `--output_dir`: Directory to save generated rollouts

### 3. Training

Run the main S2L-PO training with progressive annealing:

```bash
cd S2L-PO/examples/s2l-po
bash off2on_4b_8b_16_8.sh
```

This script implements the core S2L-PO algorithm:
- **Phase 1 (Steps 1-N)**: Progressive transition from small-model rollouts to large-model rollouts
- **Phase 2 (Steps N+1-)**: Full on-policy GRPO training


**Customization**: Modify the script for different model pairs:
```bash
# Example: Using 1.7B to guide 8B
--small_model_path Qwen/Qwen3-1.7B \
--large_model_path Qwen/Qwen3-8B

# Example: Using 4B to guide 14B
--small_model_path Qwen/Qwen3-4B \
--large_model_path Qwen/Qwen3-14B
```

### 4. Evaluation

After training, evaluate the model on mathematical reasoning benchmarks:

```bash
cd S2L-PO/eval

# Full evaluation on all benchmarks (AIME24, AIME25, MATH500)
python run_evaluation.py \
    --model_path /path/to/trained/model \
    --mode nothink \
    --k 16 \
    --tensor_parallel_size 1

# Evaluate on specific benchmarks
python run_evaluation.py \
    --model_path /path/to/trained/model \
    --mode nothink \
    --k 16 \
    --benchmarks aime24 aime25
```

**Evaluation Parameters**:
- `--model_path`: Path to the trained model checkpoint
- `--mode`: Evaluation mode (`think` or `nothink`)
  - `think`: Temperature=0.6, top-p=0.95, top-k=20
  - `nothink`: Temperature=0.7, top-p=0.8, top-k=20, presence_penalty=1.5
- `--k`: Number of rollouts per question for Pass@k computation (default: 16)
- `--benchmarks`: Specific benchmarks to evaluate (default: all)
- `--tensor_parallel_size`: Number of GPUs for parallel inference

**Output Files**:
```
eval/
├── output/
│   └── ModelName_TIMESTAMP/
│       ├── config.txt              # Configuration
│       ├── aime24_raw.jsonl        # Raw model outputs
│       ├── aime24.jsonl            # Extracted answers
│       └── ...
└── result/
    └── ModelName_TIMESTAMP/
        ├── aime24.json             # Per-benchmark results
        ├── aime25.json
        ├── math500.json
        └── summary.json            # Aggregated results
```

**Evaluation Metrics**:
- **Pass@1**: Probability of getting the correct answer in a single attempt
- **Pass@k**: Probability of getting at least one correct answer in k attempts
- **Accuracy**: Overall percentage of correctly solved problems

**Note**: The same evaluation procedure applies to other mathematical reasoning benchmarks. Simply place your benchmark data in `eval/data/` following the format:
```json
{"question": "Problem statement", "answer": "Ground truth answer"}
```

## Experimental Settings

### Training Configuration

- **Base Models**: Qwen3 series (1.7B, 4B, 8B, 14B)
- **Training Data**: DAPO17k (deduplicated, multi-step reasoning)
- **Hardware**: 8x NVIDIA L20 GPUs per node
- **Framework**: verl with default GRPO configuration


### S2L-PO Specific Settings

- **Group Size (G)**: 16 rollouts per prompt
- **Progressive Annealing**:
  - Steps 1-8: α decreases linearly from 1.0 to 0.0
  - G_w (small model) = ⌈α × G⌉
  - G_s (large model) = G - G_w
  - Steps 9-16: Full on-policy (G_w = 0, G_s = G)



## Citation

If you find this work useful, please cite our paper:

```bibtex

```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on the [verl](https://github.com/volcengine/verl) framework
- Uses [Qwen3](https://github.com/QwenLM/Qwen3) base models