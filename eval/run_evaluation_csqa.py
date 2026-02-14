
import argparse
from inference_csqa import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run CommonsenseQA evaluation with vLLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model (e.g., Qwen/Qwen3-8B or /path/to/model)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["think", "nothink"],
        default="nothink",
        help="Reasoning mode: 'think' for chain-of-thought, 'nothink' for direct answer"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of samples to generate per question (default: 8)"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["csqa"],
        help="Benchmarks to evaluate (default: csqa)"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference. None means process all at once (default: None)"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights (default: float16)"
    )
    
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp for output directory (default: auto-generated)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.timestamp is None:
        from datetime import datetime
        args.timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    
    print("\n" + "="*80)
    print("CommonsenseQA Evaluation Configuration")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"K (samples per question): {args.k}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Batch Size: {args.batch_size if args.batch_size else 'Auto (all at once)'}")
    print(f"Data Type: {args.dtype}")
    print(f"Timestamp: {args.timestamp}")
    print("="*80 + "\n")
    
    for benchmark in args.benchmarks:
        try:
            run_inference(
                model_path=args.model_path,
                benchmark=benchmark,
                mode=args.mode,
                k=args.k,
                timestamp=args.timestamp,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                batch_size=args.batch_size,
                dtype=args.dtype,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to run benchmark {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("All evaluations completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()