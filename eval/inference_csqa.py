
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm


# 数据集路径配置
BENCHMARK_PATHS = {
    "csqa": "data/commonsenseqa.jsonl",
}

# 采样配置
SAMPLING_CONFIGS = {
    "think": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": 8192,  
    },
    "nothink": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 8192,
    }
}


def load_dataset(benchmark: str) -> List[Dict[str, Any]]:
    """
    {
        "id": "...",
        "question": "...",
        "question_concept": "...",
        "choices": {
            "label": ["A", "B", "C", "D", "E"],
            "text": ["choice1", "choice2", ...]
        },
        "answerKey": "A"
    }
    """
    if benchmark not in BENCHMARK_PATHS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Must be one of {list(BENCHMARK_PATHS.keys())}")
    
    filepath = BENCHMARK_PATHS[benchmark]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            
            assert "question" in item, "Missing 'question' field"
            assert "choices" in item, "Missing 'choices' field"
            assert "answerKey" in item, "Missing 'answerKey' field"
            data.append(item)
    
    print(f"Loaded {len(data)} examples from {benchmark}")
    return data


def extract_model_name(model_path: str) -> str:

    model_name = model_path.rstrip('/').split('/')[-1]
    return model_name


def get_sampling_params(mode: str, k: int) -> SamplingParams:
    if mode not in SAMPLING_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Must be 'think' or 'nothink'")
    
    config = SAMPLING_CONFIGS[mode].copy()
    
    return SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        presence_penalty=config["presence_penalty"],
        max_tokens=config["max_tokens"],
        n=k,
    )


def format_choices(choices: Dict[str, List[str]]) -> str:
    """
    
    Args:
        choices: {"label": ["A", "B", ...], "text": ["...", "...", ...]}
    
    Returns:
        A. choice1
        B. choice2
        ...
    """
    labels = choices["label"]
    texts = choices["text"]
    
    formatted = []
    for label, text in zip(labels, texts):
        formatted.append(f"{label}. {text}")
    
    return "\n".join(formatted)


def prepare_chat_messages(item: Dict[str, Any], mode: str) -> List[Dict[str, str]]:
    """
    Args:
        item: 
        mode: "think" or "nothink"
    
    Returns:
        chat messages
    """
    question = item["question"]
    choices_formatted = format_choices(item["choices"])
    
    # 构建完整的问题文本
    full_question = f"{question}\n\n{choices_formatted}\n\nAnswer with only the letter (A, B, C, D, or E)."
    
    if mode == "think":
        user_content = f"{full_question} /think"
    else:
        user_content = f"{full_question} /no_think"
    
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


def extract_answer(text: str, valid_choices: List[str] = None) -> Optional[str]:
    """
    Args:
        text
        valid_choices: ["A", "B", "C", "D", "E"]
    """
    if valid_choices is None:
        valid_choices = ["A", "B", "C", "D", "E"]
    
    text = text.strip()
    
    patterns = [
        r'(?:answer|Answer|ANSWER)[\s:]*(?:is|IS)?[\s:]*\(?([A-E])\)?',
        r'(?:correct|Correct|CORRECT)[\s:]*(?:answer|option|choice)?[\s:]*\(?([A-E])\)?',
        r'(?:option|Option|OPTION|choice|Choice|CHOICE)[\s:]*\(?([A-E])\)?',
        r'\(([A-E])\)',
        r'\[([A-E])\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).upper()
            if answer in valid_choices:
                return answer
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) == 1 and line.upper() in valid_choices:
            return line.upper()
        if line and line[0].upper() in valid_choices and (len(line) == 1 or not line[1].isalnum()):
            return line[0].upper()
    
    for choice in valid_choices:
        pattern = r'\b' + choice + r'\b'
        if re.search(pattern, text):
            return choice
    
    text_upper = text.upper()
    found_choices = [c for c in valid_choices if c in text_upper]
    if len(found_choices) == 1:
        return found_choices[0]
    
    for choice in reversed(valid_choices):
        if choice in text_upper:
            idx = text_upper.rfind(choice)
            if idx == 0 or not text_upper[idx-1].isalnum():
                if idx == len(text_upper)-1 or not text_upper[idx+1].isalnum():
                    return choice
    
    return None


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    pass_at_1 = sum(1 for r in results if r.get("predicted_answers") and 
                    r["predicted_answers"][0] == r["answerKey"])

    pass_at_k = sum(1 for r in results if r.get("predicted_answers") and 
                    r["answerKey"] in r["predicted_answers"])

    majority_correct = 0
    for r in results:
        if r.get("predicted_answers"):
            from collections import Counter
            vote_counts = Counter(r["predicted_answers"])
            if vote_counts:
                majority_answer = vote_counts.most_common(1)[0][0]
                if majority_answer == r["answerKey"]:
                    majority_correct += 1
    
    extraction_success = sum(1 for r in results if r.get("predicted_answers") and 
                            any(ans is not None for ans in r["predicted_answers"]))
    
    stats = {
        "total": total,
        "pass@1": pass_at_1,
        "pass@1_rate": pass_at_1 / total if total > 0 else 0,
        "pass@k": pass_at_k,
        "pass@k_rate": pass_at_k / total if total > 0 else 0,
        "majority_vote_correct": majority_correct,
        "majority_vote_rate": majority_correct / total if total > 0 else 0,
        "extraction_success": extraction_success,
        "extraction_rate": extraction_success / total if total > 0 else 0,
    }
    
    return stats


def save_config(output_dir: Path, model_path: str, mode: str, benchmark: str,
                k: int, sampling_params: SamplingParams, first_prompt: str,
                first_messages: List[Dict[str, str]]):
    config_path = output_dir / "config.txt"
    
    mode_flag = 'a' if config_path.exists() else 'w'
    
    with open(config_path, mode_flag, encoding='utf-8') as f:
        if mode_flag == 'a':
            f.write("\n" + "=" * 80 + "\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"CommonsenseQA Evaluation Configuration - {benchmark.upper()}\n")
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
        )
    except Exception as e:
        print("\n[ERROR] vLLM engine initialization failed.")
        print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
        raise
    
    print("Model loaded successfully!\n")
    
    tokenizer = llm.get_tokenizer()
    
    sampling_params = get_sampling_params(mode, k)
    
    print("Preparing prompts with chat template...")
    prompts = []
    all_items = []
    all_messages = []
    
    for item in tqdm(dataset, desc="Building prompts", unit="example"):
        messages = prepare_chat_messages(item, mode)
        prompt = apply_chat_template(tokenizer, messages)
        
        prompts.append(prompt)
        all_items.append(item)
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
            batch_items = all_items[start_idx:end_idx]
            
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            
            for idx, output in enumerate(batch_outputs):
                item = batch_items[idx]
                generated_texts = [o.text for o in output.outputs]

                valid_choices = item["choices"]["label"]
                predicted_answers = [extract_answer(text, valid_choices) for text in generated_texts]
                
                result = {
                    "id": item.get("id", ""),
                    "question": item["question"],
                    "question_concept": item.get("question_concept", ""),
                    "choices": item["choices"],
                    "answerKey": item["answerKey"],
                    "outputs": generated_texts,
                    "predicted_answers": predicted_answers,
                }
                all_results.append(result)
                
                pbar.update(1)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in all_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    stats = calculate_accuracy(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    stats_file = output_dir / f"{benchmark}_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Inference completed!")
    print(f"  Results saved to: {output_file}")
    print(f"  Statistics saved to: {stats_file}")
    print(f"\n  Statistics:")
    print(f"  - Total examples: {stats['total']}")
    print(f"  - Pass@1: {stats['pass@1']} ({stats['pass@1_rate']:.2%})")
    print(f"  - Pass@{k}: {stats['pass@k']} ({stats['pass@k_rate']:.2%})")
    print(f"  - Majority Vote: {stats['majority_vote_correct']} ({stats['majority_vote_rate']:.2%})")
    print(f"  - Extraction Success: {stats['extraction_success']} ({stats['extraction_rate']:.2%})")
    print(f"{'='*80}\n")