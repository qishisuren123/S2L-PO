

"""
小模型批量采样脚本 - 使用vLLM批量生成离线Rollout数据
优化点：
1. 批量处理多个prompt，充分利用GPU并行能力
2. 每个prompt生成n个样本，vLLM内部并行处理
3. 支持断点续传和增量保存

输出格式要求：
- prompt_id: 稳定的prompt标识符
- prompt: 原始问题文本
- prompt_ids: tokenized prompt序列
- response_ids: 小模型生成的response tokens（含EOS）
- small_model_log_probs: 每个token的log probability
- reward: scalar奖励分数
- is_correct: 是否正确
- data_source: 数据来源标签
- ground_truth: 参考答案（可选）
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import numpy as np

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd


def load_train_data(data_path: str, max_samples: int = -1) -> List[Dict[str, Any]]:
    """加载训练数据（支持parquet和jsonl）"""
    print(f"Loading data from: {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        data = df.to_dict('records')
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if max_samples > 0:
        data = data[:max_samples]

    print(f"Loaded {len(data)} examples")
    return data


def normalize_answer(answer: str) -> str:
    """标准化答案格式（用于匹配）"""
    answer = answer.strip().lower()
    import re

    # 尝试提取 \boxed{...} 格式
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
    if boxed_match:
        return boxed_match.group(1).strip().lower()

    # 尝试提取 #### 后面的答案
    hash_match = re.search(r'####\s*(.+)', answer)
    if hash_match:
        return hash_match.group(1).strip().lower()

    # 提取最后出现的数字或简单表达式
    number_match = re.findall(r'-?\d+\.?\d*', answer)
    if number_match:
        return number_match[-1]

    return answer


def check_correctness(generated_text: str, ground_truth: str) -> bool:
    """检查生成结果是否正确"""
    pred_answer = normalize_answer(generated_text)
    true_answer = normalize_answer(ground_truth)
    return pred_answer == true_answer or true_answer in pred_answer


def prepare_prompt(item: Dict[str, Any], prompt_format: str = "chat") -> str:
    """准备prompt（支持不同格式）"""
    question = item.get("question", item.get("prompt", ""))

    if prompt_format == "chat":
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        return question


def extract_prompt_ids_from_tokenizer(tokenizer, prompt_text: str) -> List[int]:
    """使用tokenizer获取prompt的token IDs（不添加special tokens）"""
    encoding = tokenizer.encode(prompt_text, add_special_tokens=False)
    return encoding


def process_batch_outputs(
    batch_items: List[Dict[str, Any]],
    batch_outputs: List[Any],
    tokenizer,
    prompt_format: str,
    start_prompt_idx: int,
) -> List[Dict[str, Any]]:
    """
    处理批量生成的输出

    Args:
        batch_items: 输入的数据项列表
        batch_outputs: vLLM生成的输出列表
        tokenizer: tokenizer实例
        prompt_format: prompt格式
        start_prompt_idx: 起始prompt索引（用于生成prompt_id）

    Returns:
        处理后的结果列表
    """
    results = []

    for batch_idx, (item, request_output) in enumerate(zip(batch_items, batch_outputs)):
        question = item.get("question", item.get("prompt", ""))
        answer = item.get("answer", "")
        data_source = item.get("data_source", item.get("type", "unknown"))

        # 格式化prompt（用于提取prompt_ids）
        prompt_text = prepare_prompt(item, prompt_format)
        prompt_ids = extract_prompt_ids_from_tokenizer(tokenizer, prompt_text)

        # 生成唯一的prompt_id
        prompt_idx = start_prompt_idx + batch_idx
        prompt_id = f"prompt_{prompt_idx:06d}"

        # 处理该prompt的所有样本（n个）
        for sample_idx, output in enumerate(request_output.outputs):
            # 提取生成的文本
            generated_text = output.text

            # 提取response的token IDs
            response_ids = output.token_ids

            # 提取log_probs
            log_probs = []
            if output.logprobs is not None and len(output.logprobs) > 0:
                for token_idx, token_logprobs_dict in enumerate(output.logprobs):
                    if token_logprobs_dict and token_idx < len(response_ids):
                        token_id = response_ids[token_idx]
                        if token_id in token_logprobs_dict:
                            log_prob = token_logprobs_dict[token_id].logprob
                        else:
                            log_prob = list(token_logprobs_dict.values())[0].logprob
                        log_probs.append(float(log_prob))

            # 确保长度一致
            if len(log_probs) < len(response_ids):
                log_probs.extend([0.0] * (len(response_ids) - len(log_probs)))
            elif len(log_probs) > len(response_ids):
                log_probs = log_probs[:len(response_ids)]

            # 计算reward
            is_correct = check_correctness(generated_text, answer)
            reward = 1.0 if is_correct else 0.0

            # 构造结果
            result = {
                "prompt_id": prompt_id,
                "prompt": question,
                "prompt_ids": prompt_ids,
                "response": generated_text,
                "response_ids": response_ids,
                "small_model_log_probs": log_probs,
                "reward": reward,
                "is_correct": is_correct,
                "data_source": data_source,
                "ground_truth": answer,
            }

            results.append(result)

    return results


def run_sampling(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_samples_per_prompt: int = 50,
    max_prompt_length: int = 2048,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = -1,
    tensor_parallel_size: int = 1,
    batch_size: int = 16,  # 增大默认batch size
    max_data_samples: int = -1,
    prompt_format: str = "chat",
    checkpoint_every: int = 100,
):
    """
    运行小模型批量采样

    关键优化：
    1. 使用batch_size控制同时处理的prompt数量
    2. 每个prompt生成n个样本由vLLM内部并行处理
    3. 总并行度 = batch_size * num_samples_per_prompt
    """
    # 1. 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "small_model_rollouts.jsonl"
    checkpoint_file = output_path / "checkpoint.txt"
    config_file = output_path / "sampling_config.json"

    # 检查checkpoint
    start_idx = 0
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"Resuming from prompt index {start_idx}")

    # 2. 加载数据
    print("\n" + "=" * 80)
    print("Small Model Batch Sampling for GRPO Training")
    print("=" * 80 + "\n")

    dataset = load_train_data(data_path, max_samples=max_data_samples)

    # 3. 初始化模型
    print(f"\nLoading model: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,  # 提高GPU利用率
        max_model_len=max_prompt_length + max_new_tokens,
        # 优化参数
        max_num_batched_tokens=None,  # 自动计算
        max_num_seqs=batch_size * num_samples_per_prompt,  # 最大并行序列数
    )
    print("Model loaded successfully!")

    # 获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 配置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        n=num_samples_per_prompt,  # 每个prompt生成n个样本
        logprobs=1,
        prompt_logprobs=None,
    )

    # 保存配置
    config = {
        "model_path": model_path,
        "data_path": data_path,
        "num_samples_per_prompt": num_samples_per_prompt,
        "batch_size": batch_size,
        "max_prompt_length": max_prompt_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "tensor_parallel_size": tensor_parallel_size,
        "prompt_format": prompt_format,
        "timestamp": datetime.now().isoformat(),
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n" + "-" * 80)
    print("Sampling Configuration:")
    print("-" * 80)
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Total prompts: {len(dataset)}")
    print(f"  Batch size (prompts): {batch_size}")
    print(f"  Samples per prompt: {num_samples_per_prompt}")
    print(f"  Total parallel sequences: {batch_size * num_samples_per_prompt}")
    print(f"  Total samples to generate: {len(dataset) * num_samples_per_prompt}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Top-k: {top_k}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print("-" * 80 + "\n")

    # 5. 批量采样
    mode = "a" if start_idx > 0 else "w"

    # 统计信息
    total_correct = 0
    total_generated = 0

    with open(output_file, mode, encoding='utf-8') as f_out:
        # 创建批次
        num_batches = (len(dataset) - start_idx + batch_size - 1) // batch_size

        pbar = tqdm(
            total=len(dataset) - start_idx,
            desc="Sampling",
            unit="prompt"
        )

        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(dataset))

            batch_items = dataset[batch_start:batch_end]

            # 准备batch prompts
            batch_prompts = [
                prepare_prompt(item, prompt_format) 
                for item in batch_items
            ]

            try:
                # 批量生成
                # vLLM会自动并行处理 batch_size * num_samples_per_prompt 个序列
                batch_outputs = llm.generate(batch_prompts, sampling_params)

                # 处理输出
                results = process_batch_outputs(
                    batch_items=batch_items,
                    batch_outputs=batch_outputs,
                    tokenizer=tokenizer,
                    prompt_format=prompt_format,
                    start_prompt_idx=batch_start,
                )

                # 写入文件
                for result in results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    total_generated += 1
                    if result["is_correct"]:
                        total_correct += 1

                f_out.flush()

                # 更新进度条
                pbar.update(len(batch_items))

                # 保存checkpoint
                if (batch_end) % checkpoint_every == 0 or batch_end == len(dataset):
                    with open(checkpoint_file, 'w') as cf:
                        cf.write(str(batch_end))

                    # 显示当前统计
                    current_acc = total_correct / total_generated if total_generated > 0 else 0
                    pbar.set_postfix({
                        "acc": f"{current_acc:.2%}",
                        "correct": total_correct,
                        "total": total_generated
                    })

            except Exception as e:
                print(f"\nError processing batch {batch_idx} (prompts {batch_start}-{batch_end}): {e}")
                import traceback
                traceback.print_exc()
                continue

        pbar.close()

    # 6. 删除checkpoint文件
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # 7. 统计信息
    print("\n" + "=" * 80)
    print("Sampling Completed!")
    print("=" * 80)
    print(f"  Output file: {output_file}")
    print(f"  Total prompts: {len(dataset)}")
    print(f"  Samples per prompt: {num_samples_per_prompt}")
    print(f"  Total samples: {total_generated}")
    print(f"  Correct samples: {total_correct}")
    print(f"  Accuracy: {total_correct / total_generated:.2%}" if total_generated > 0 else "  Accuracy: N/A")
    print("=" * 80 + "\n")

    # 8. 验证输出格式
    print("Verifying output format...")
    with open(output_file, 'r') as f:
        first_line = f.readline()
        first_sample = json.loads(first_line)

        required_fields = [
            "prompt_id", "prompt", "prompt_ids", "response_ids",
            "small_model_log_probs", "reward", "is_correct", "data_source"
        ]

        missing_fields = [field for field in required_fields if field not in first_sample]

        if missing_fields:
            print(f"⚠️  Warning: Missing fields in output: {missing_fields}")
        else:
            print("✓ All required fields present!")

        print("\nFirst sample preview:")
        print(json.dumps({k: v if k not in ["prompt_ids", "response_ids", "small_model_log_probs"] 
                         else f"[{len(v)} items]" for k, v in first_sample.items()}, 
                        indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Small model batch sampling for GRPO training")

    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to small model (e.g., Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--data_path", type=str, 
                       default="/mnt/petrelfs/renyiming/verl/data/train.parquet",
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, 
                       default="./small_model_rollouts",
                       help="Output directory")

    # 采样参数
    parser.add_argument("--num_samples_per_prompt", type=int, default=50,
                       help="Number of samples per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                       help="Maximum prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1,
                       help="Top-k sampling (-1 means disabled)")

    # 系统参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (number of prompts processed simultaneously)")
    parser.add_argument("--max_data_samples", type=int, default=-1,
                       help="Maximum data samples to use (-1 means all)")

    # 其他参数
    parser.add_argument("--prompt_format", type=str, default="chat",
                       choices=["chat", "direct"],
                       help="Prompt format")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                       help="Save checkpoint every N prompts")

    args = parser.parse_args()

    run_sampling(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples_per_prompt=args.num_samples_per_prompt,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_data_samples=args.max_data_samples,
        prompt_format=args.prompt_format,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()