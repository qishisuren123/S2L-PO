set -e

# ====================
MODEL_PATH="/Qwen3-4B-Base"  
DATA_PATH="/data/train_converted_deduplicated.jsonl"
OUTPUT_DIR=${OUTPUT_DIR:-"./small_model_rollouts_4B"}


NUM_SAMPLES=32            
MAX_PROMPT_LENGTH=2048     
MAX_NEW_TOKENS=30000        
TEMPERATURE=0.7            
TOP_P=0.95             


TENSOR_PARALLEL_SIZE=1     
BATCH_SIZE=16            
MAX_DATA_SAMPLES=-1    


PROMPT_FORMAT="chat"       
CHECKPOINT_EVERY=16       

# ===============
echo "========================================"
echo "Small Model Sampling Configuration"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Samples per prompt: $NUM_SAMPLES"
echo "GPUs: $TENSOR_PARALLEL_SIZE"
echo "========================================"
echo ""

python /verl/examples/small_model_rollout2.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples_per_prompt $NUM_SAMPLES \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size $BATCH_SIZE \
    --max_data_samples $MAX_DATA_SAMPLES \
    --prompt_format $PROMPT_FORMAT \
    --checkpoint_every $CHECKPOINT_EVERY

echo ""
echo "========================================"
echo "✓ Sampling completed!"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Next step: Verify the output format"
echo ""
echo "Run verification:"
echo "  python verify_sampling_output.py --input $OUTPUT_DIR/small_model_rollouts.jsonl"
echo "========================================"