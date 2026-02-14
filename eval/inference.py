
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm


BENCHMARK_PATHS = {
    "aime24": "data/aime24.jsonl",
    "aime25": "data/aime25.jsonl",
    "math500": "data/math500.jsonl",
}

SAMPLING_CONFIGS = {
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
    }
}

BENCHMARK_MAX_TOKENS = {
    "aime24": 38912,
    "aime25": 38912,
    "math500": 32768,
}


def load_dataset(benchmark: str) -> List[Dict[str, Any]]:
    if benchmark not in BENCHMARK_PATHS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of {list(BENCHMARK_PATHS.keys())}")
    
    filepath = BENCHMARK_PATHS[benchmark]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples from {benchmark}")
    return data


def extract_model_name(model_path: str) -> str:
    model_name = model_path.rstrip('/').split('/')[-1]
    return model_name


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
        n=k,  # 直接设置n=k，一次性生成k个输出
    )


def prepare_chat_messages(question: str, mode: str) -> List[Dict[str, str]]:
    if mode == "think":
        user_content = f"{question} /think"  
    else:
        user_content = f"{question} /no_think"

    messages = [
        {"role": "user", "content": user_content},
    ]
    return messages


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def save_config(output_dir: Path, model_path: str, mode: str, benchmark: str, 
                k: int, sampling_params: SamplingParams, first_prompt: str,
                first_messages: List[Dict[str, str]]):
    config_path = output_dir / "config.txt"
    
    mode_flag = 'a' if config_path.exists() else 'w'
    
    with open(config_path, mode_flag, encoding='utf-8') as f:
        if mode_flag == 'a':
            f.write("\n" + "=" * 80 + "\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Qwen3 Evaluation Configuration - {benchmark.upper()}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"K (sampling times): {k}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Sampling Parameters:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Temperature: {sampling_params.temperature}\n")
        f.write(f"Top-p: {sampling_params.top_p}\n")
        f.write(f"Top-k: {sampling_params.top_k}\n")
        f.write(f"Presence Penalty: {sampling_params.presence_penalty}\n")
        f.write(f"Max Tokens: {sampling_params.max_tokens}\n")
        f.write(f"N (samples per prompt): {sampling_params.n}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("First Messages (Chat Format):\n")
        f.write("-" * 80 + "\n")
        f.write(json.dumps(first_messages, indent=2, ensure_ascii=False) + "\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("First Prompt (After Chat Template):\n")
        f.write("-" * 80 + "\n")
        f.write(first_prompt + "\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Configuration saved to: {config_path}")


def run_inference(
    model_path: str,
    benchmark: str,
    mode: str,
    k: int,
    timestamp: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    batch_size: Optional[int] = None,
    dtype: str = "float16",
):
    print(f"\n{'='*80}")
    print(f"Starting inference for {benchmark.upper()} in {mode} mode")
    print(f"{'='*80}\n")
    
    dataset = load_dataset(benchmark)
    
    model_name = extract_model_name(model_path)
    output_dir = Path(f"output/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    print(f"  - tensor_parallel_size: {tensor_parallel_size}")
    print(f"  - gpu_memory_utilization: {gpu_memory_utilization}")
    print(f"  - dtype: {dtype}")

    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            # quantization="gptq_marlin",
        )
    except Exception as e:
        print("\n[ERROR] vLLM engine initialization failed.")
        print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
        try:
            import torch

            print(
                "torch.cuda.is_available="
                f"{torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
            )
        except Exception as torch_e:
            print(f"[WARN] Failed to import torch for CUDA diagnostics: {torch_e}")
        raise

    print("Model loaded successfully!\n")
    
    tokenizer = llm.get_tokenizer()
    
    sampling_params = get_sampling_params(mode, benchmark, k)
    
    print("Preparing prompts with chat template...")
    prompts = []
    answers = []
    questions = []
    all_messages = []
    
    for item in tqdm(dataset, desc="Building prompts", unit="example"):
        question = item["question"]
        answer = item["answer"]
        
        messages = prepare_chat_messages(question, mode)
        
        prompt = apply_chat_template(tokenizer, messages)
        
        prompts.append(prompt)
        answers.append(answer)
        questions.append(question)
        all_messages.append(messages)
    
    first_prompt = prompts[0]
    first_messages = all_messages[0]
    
    print("\n" + "-" * 80)
    print("First messages (chat format):")
    print("-" * 80)
    print(json.dumps(first_messages, indent=2, ensure_ascii=False))
    print("\n" + "-" * 80)
    print("First prompt (after chat template):")
    print("-" * 80)
    print(first_prompt[:500] + "..." if len(first_prompt) > 500 else first_prompt)
    print("-" * 80 + "\n")
    
    save_config(output_dir, model_path, mode, benchmark, k, sampling_params, 
                first_prompt, first_messages)
    
    print(f"Starting batch inference:")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Samples per example: {k}")
    print(f"  - Total generations: {len(dataset) * k}")
    
    if batch_size is None:
        batch_size = len(prompts)
    
    print(f"  - Batch size: {batch_size}\n")
    
    all_results = []
    output_file = output_dir / f"{benchmark}_raw.jsonl"
    
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    with tqdm(total=len(prompts), desc=f"Inference ({benchmark})", unit="example") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(prompts))
            
            batch_prompts = prompts[start_idx:end_idx]
            batch_questions = questions[start_idx:end_idx]
            batch_answers = answers[start_idx:end_idx]
            
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            
            for idx, output in enumerate(batch_outputs):
                generated_texts = [o.text for o in output.outputs]
                
                result = {
                    "question": batch_questions[idx],
                    "answer": batch_answers[idx],
                    "outputs": generated_texts,
                }
                all_results.append(result)
                
                pbar.update(1)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*80}")
    print(f"✓ Inference completed!")
    print(f"  Results saved to: {output_file}")
    print(f"  Total examples: {len(all_results)}")
    print(f"  Samples per example: {k}")
    print(f"  Total generations: {len(all_results) * k}")
    print(f"{'='*80}\n")
