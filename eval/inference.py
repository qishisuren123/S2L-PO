"""
vLLM inference script for evaluating S2L-PO models on AIME24/25, MATH500, and OlympiadBench.

Supports:
  - Qwen3-series models (thinking / non-thinking mode)
  - Batch inference with n=k parallel sampling
  - Automatic chat-template formatting
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams


BENCHMARK_PATHS: Dict[str, str] = {
    "aime24":       "data/aime24.jsonl",
    "aime25":       "data/aime25.jsonl",
    "math500":      "data/math500.jsonl",
    "olympiadbench": "data/olympiadbench.jsonl",
}

SAMPLING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "think": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": 38912,
    },
    "nothink": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 38912,
    },
}

BENCHMARK_MAX_TOKENS: Dict[str, int] = {
    "aime24":        38912,
    "aime25":        38912,
    "math500":       32768,
    "olympiadbench": 32768,
}


def load_dataset(benchmark: str) -> List[Dict[str, Any]]:
    if benchmark not in BENCHMARK_PATHS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of {list(BENCHMARK_PATHS.keys())}")
    filepath = BENCHMARK_PATHS[benchmark]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} examples from {benchmark}")
    return data


def extract_model_name(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def get_sampling_params(mode: str, benchmark: str, k: int) -> SamplingParams:
    if mode not in SAMPLING_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Must be 'think' or 'nothink'")
    config = SAMPLING_CONFIGS[mode].copy()
    config["max_tokens"] = BENCHMARK_MAX_TOKENS.get(benchmark, 32768)
    return SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        presence_penalty=config["presence_penalty"],
        max_tokens=config["max_tokens"],
        n=k,
    )


def prepare_chat_messages(question: str, mode: str) -> List[Dict[str, str]]:
    """
    Build chat messages for Qwen3 think / nothink mode.
    Following the Qwen3 technical report, the thinking mode is triggered by
    appending '/think' to the user message; nothink mode uses '/no_think'.
    """
    suffix = "/think" if mode == "think" else "/no_think"
    return [{"role": "user", "content": f"{question} {suffix}"}]


def save_config(
    output_dir: Path,
    model_path: str,
    mode: str,
    benchmark: str,
    k: int,
    sampling_params: SamplingParams,
    first_prompt: str,
    first_messages: List[Dict[str, str]],
) -> None:
    config_path = output_dir / "config.txt"
    open_mode = "a" if config_path.exists() else "w"
    with open(config_path, open_mode, encoding="utf-8") as f:
        sep = "=" * 80
        if open_mode == "a":
            f.write(f"\n{sep}\n")
        f.write(f"{sep}\nEvaluation Config — {benchmark.upper()}\n{sep}\n\n")
        f.write(f"Model Path:  {model_path}\n")
        f.write(f"Mode:        {mode}\n")
        f.write(f"Benchmark:   {benchmark}\n")
        f.write(f"K:           {k}\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Temperature:      {sampling_params.temperature}\n")
        f.write(f"Top-p:            {sampling_params.top_p}\n")
        f.write(f"Top-k:            {sampling_params.top_k}\n")
        f.write(f"Presence Penalty: {sampling_params.presence_penalty}\n")
        f.write(f"Max Tokens:       {sampling_params.max_tokens}\n")
        f.write(f"N per prompt:     {sampling_params.n}\n\n")
        f.write("First Messages:\n")
        f.write(json.dumps(first_messages, indent=2, ensure_ascii=False) + "\n\n")
        f.write("First Prompt (after chat template):\n")
        f.write(first_prompt + "\n")
    print(f"Config saved to: {config_path}")


def run_inference(
    model_path: str,
    benchmark: str,
    mode: str,
    k: int,
    timestamp: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    batch_size: Optional[int] = None,
    dtype: str = "bfloat16",
    max_model_len: Optional[int] = None,
) -> None:
    print(f"\n{'='*80}\nInference: {benchmark.upper()} | mode={mode} | k={k}\n{'='*80}\n")

    dataset = load_dataset(benchmark)
    model_name = extract_model_name(model_path)
    output_dir = Path(f"output/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    llm_kwargs: Dict[str, Any] = dict(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print("Model loaded.\n")

    sampling_params = get_sampling_params(mode, benchmark, k)

    prompts, answers, questions, all_messages = [], [], [], []
    for item in tqdm(dataset, desc="Building prompts"):
        messages = prepare_chat_messages(item["question"], mode)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        answers.append(item["answer"])
        questions.append(item["question"])
        all_messages.append(messages)

    save_config(output_dir, model_path, mode, benchmark, k, sampling_params, prompts[0], all_messages[0])

    print(f"Total examples: {len(dataset)} | Samples per example: {k} | Total generations: {len(dataset) * k}")
    batch_size = batch_size or len(prompts)
    output_file = output_dir / f"{benchmark}_raw.jsonl"
    all_results = []

    with tqdm(total=len(prompts), desc=f"Inference ({benchmark})") as pbar:
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_outputs = llm.generate(prompts[start:end], sampling_params)
            for idx, output in enumerate(batch_outputs):
                all_results.append({
                    "question": questions[start + idx],
                    "answer": answers[start + idx],
                    "outputs": [o.text for o in output.outputs],
                })
                pbar.update(1)
            with open(output_file, "w", encoding="utf-8") as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ Results saved to: {output_file} ({len(all_results)} examples, {k} samples each)\n")
