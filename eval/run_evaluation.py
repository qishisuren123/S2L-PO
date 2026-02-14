"""
Usage:
    python run_evaluation.py --model_path Qwen/Qwen3-4B --mode think --k 10
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import os

# 导入其他模块
try:
    from inference import run_inference
    from extract_answers import process_all_benchmarks
    from scoring import score_all_benchmarks
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure inference.py, extract_answers.py, and scoring.py are in the same directory.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3 evaluation on AIME24/25 and MATH500 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with thinking mode
  python run_evaluation.py --model_path Qwen/Qwen3-4B --mode think --k 10
  
  # Non-thinking mode evaluation
  python run_evaluation.py --model_path Qwen/Qwen3-4B --mode nothink --k 10
  
  # Multi-GPU evaluation (2 GPUs)
  python run_evaluation.py --model_path Qwen/Qwen3-7B --mode think --k 20 --tensor_parallel_size 2
  
  # Evaluate specific benchmarks only
  python run_evaluation.py --model_path Qwen/Qwen3-4B --mode think --k 10 --benchmarks aime24 aime25
  
  # Reprocess existing outputs (skip inference)
  python run_evaluation.py --model_path Qwen/Qwen3-4B --mode think --k 10 \\
      --skip_inference --timestamp 251027123456
  
  # Only re-score (skip inference and extraction)
  python run_evaluation.py --model_path Qwen/Qwen3-4B --mode think --k 10 \\
      --skip_inference --skip_extraction --timestamp 251027123456
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen3 model (e.g., Qwen/Qwen3-4B, /path/to/local/model)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["think", "nothink"],
        help="Evaluation mode: 'think' (thinking mode) or 'nothink' (non-thinking mode)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["aime24", "aime25", "math500"],
        choices=["aime24", "aime25", "math500"],
        help="Benchmarks to evaluate. Default: all three benchmarks"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of samples per question for Pass@k calculation. Default: 10"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism. Default: 1 (single GPU)"
    )
    
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference step and use existing raw outputs. Requires --timestamp"
    )
    
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip answer extraction step and use existing extracted answers. Requires --timestamp"
    )
    
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Use existing timestamp for reprocessing (format: YYMMDDHHMMSS, e.g., 251027123456)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio. Default: 0.9 (90%%)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for inference. None means process all examples at once. Default: None"
    )
    
    return parser.parse_args()


def validate_args(args):
    errors = []
    warnings = []
    
    if args.k < 1:
        errors.append(f"k must be at least 1, got {args.k}")
    elif args.k > 100:
        warnings.append(f"k={args.k} is very large, this will take a long time")
    
    if args.tensor_parallel_size < 1:
        errors.append(f"tensor_parallel_size must be at least 1, got {args.tensor_parallel_size}")
    
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        errors.append(f"gpu_memory_utilization must be between 0 and 1, got {args.gpu_memory_utilization}")
    
    if args.skip_inference and args.timestamp is None:
        errors.append("--timestamp is required when using --skip_inference")
    
    if args.skip_extraction and not args.skip_inference:
        errors.append("--skip_extraction requires --skip_inference (you must skip inference first)")
    
    if args.timestamp is not None:
        if not args.timestamp.isdigit() or len(args.timestamp) != 12:
            errors.append(f"Invalid timestamp format: '{args.timestamp}'. Expected format: YYMMDDHHMMSS (12 digits)")
        
        if args.skip_inference:
            model_name = args.model_path.rstrip('/').split('/')[-1]
            output_dir = Path(f"output/{model_name}_{args.timestamp}")
            if not output_dir.exists():
                errors.append(f"Output directory not found: {output_dir}")
    
    if not args.skip_inference:
        data_dir = Path("data")
        if not data_dir.exists():
            errors.append(f"Data directory not found: {data_dir}. Please create it and add dataset files.")
        else:
            missing_files = []
            for benchmark in args.benchmarks:
                data_file = data_dir / f"{benchmark}.jsonl"
                if not data_file.exists():
                    missing_files.append(str(data_file))
            
            if missing_files:
                errors.append(f"Missing dataset files: {', '.join(missing_files)}")
    
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"   - {error}")
        print()
        sys.exit(1)
    
    if warnings:
        print("⚠ Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    print("✓ Configuration validated successfully!\n")


def print_config_summary(args, model_name, timestamp, output_dir, result_dir):
    print("="*80)
    print("Qwen3 Evaluation Pipeline Configuration")
    print("="*80)
    print(f"Model Path:              {args.model_path}")
    print(f"Model Name:              {model_name}")
    print(f"Evaluation Mode:         {args.mode}")
    print(f"Benchmarks:              {', '.join(args.benchmarks)}")
    print(f"K (samples per Q):       {args.k}")
    print(f"Tensor Parallel Size:    {args.tensor_parallel_size}")
    print(f"GPU Memory Utilization:  {args.gpu_memory_utilization:.1%}")
    print(f"Timestamp:               {timestamp}")
    print(f"Output Directory:        {output_dir}")
    print(f"Result Directory:        {result_dir}")
    print("="*80)
    
    if args.mode == "think":
        print("\nSampling Parameters (Thinking Mode):")
        print("  - Temperature: 0.6")
        print("  - Top-p: 0.95")
        print("  - Top-k: 20")
        print("  - Presence Penalty: 0.0")
        print("  - Max Tokens: 38912 (AIME), 32768 (MATH500)")
    else:
        print("\nSampling Parameters (Non-Thinking Mode):")
        print("  - Temperature: 0.7")
        print("  - Top-p: 0.8")
        print("  - Top-k: 20")
        print("  - Presence Penalty: 1.5")
        print("  - Max Tokens: 38912 (AIME), 32768 (MATH500)")
    
    print()
    
    if args.skip_inference or args.skip_extraction:
        print("Pipeline Steps:")
        print(f"  Step 1 (Inference):   {'⊘ SKIPPED' if args.skip_inference else '✓ ENABLED'}")
        print(f"  Step 2 (Extraction):  {'⊘ SKIPPED' if args.skip_extraction else '✓ ENABLED'}")
        print(f"  Step 3 (Scoring):     ✓ ENABLED")
        print()


def main():
    print("\n" + "="*80)
    print("Qwen3 Model Evaluation System")
    print("="*80 + "\n")
    
    args = parse_args()
    
    validate_args(args)
    
    if args.timestamp:
        timestamp = args.timestamp
        print(f"📅 Using existing timestamp: {timestamp}\n")
    else:
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        print(f"📅 Generated new timestamp: {timestamp}\n")
    
    model_name = args.model_path.rstrip('/').split('/')[-1]
    
    output_dir = Path(f"output/{model_name}_{timestamp}")
    result_dir = Path(f"result/{model_name}_{timestamp}")
    
    print_config_summary(args, model_name, timestamp, output_dir, result_dir)
    
    try:
        if not args.skip_inference:
            print("="*80)
            print("Step 1/3: Running Inference")
            print("="*80 + "\n")
            
            for idx, benchmark in enumerate(args.benchmarks, 1):
                print(f"\n{'─'*80}")
                print(f"[{idx}/{len(args.benchmarks)}] Processing benchmark: {benchmark.upper()}")
                print(f"{'─'*80}\n")
                
                run_inference(
                    model_path=args.model_path,
                    benchmark=benchmark,
                    mode=args.mode,
                    k=args.k,
                    timestamp=timestamp,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    batch_size=args.batch_size,
                )
                
                print(f"\n✓ Completed benchmark: {benchmark.upper()}")
        else:
            print("\n" + "="*80)
            print("Step 1/3: Inference [SKIPPED]")
            print("="*80)
            print("Using existing raw outputs from:", output_dir)
            print()
        
        if not args.skip_extraction:
            print("\n" + "="*80)
            print("Step 2/3: Extracting Answers")
            print("="*80 + "\n")
            
            process_all_benchmarks(output_dir)
        else:
            print("\n" + "="*80)
            print("Step 2/3: Answer Extraction [SKIPPED]")
            print("="*80)
            print("Using existing extracted answers from:", output_dir)
            print()
        
        print("\n" + "="*80)
        print("Step 3/3: Scoring and Evaluation")
        print("="*80 + "\n")
        
        score_all_benchmarks(output_dir, result_dir)
        
        print("\n" + "="*80)
        print("✓ Evaluation Pipeline Completed Successfully!")
        print("="*80 + "\n")
        
        print("📁 Output Locations:")
        print(f"   Raw outputs:       {output_dir}")
        print(f"   Evaluation results: {result_dir}")
        print()
        
        print("📊 View Results:")
        print(f"   python view_results.py --result_dir {result_dir}")
        print()
        
        print("📄 Summary File:")
        print(f"   {result_dir}/summary.json")
        print()
        
        summary_file = result_dir / "summary.json"
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("📈 Quick Results:")
            for result in summary:
                bench = result['benchmark']
                accuracy = result.get('accuracy_any', 0)
                pass_at_1 = result['pass_at_k'].get('pass@1', 0)
                print(f"   {bench:10s} - Pass@1: {pass_at_1:.4f}, Accuracy(any): {accuracy:.4f}")
            print()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user (Ctrl+C)")
        print(f"💾 Partial results may be available in: {output_dir}")
        print()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Error occurred during evaluation:")
        print(f"   {str(e)}")
        print()
        
        import traceback
        print("Full traceback:")
        print("─"*80)
        traceback.print_exc()
        print("─"*80)
        
        print(f"\n💾 Check partial results in: {output_dir}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()