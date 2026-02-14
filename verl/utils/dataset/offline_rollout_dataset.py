

import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask


class OfflineRolloutDataset(Dataset):
    """
    {
        "prompt_id": "prompt_000001",
        "prompt":,
        "prompt_ids": [token_id1, token_id2, ...],
        "response": ,
        "response_ids": [token_id1, token_id2, ...],
        "small_model_log_probs": [log_prob1, log_prob2, ...],
"reward": 1.0,
        "is_correct": true,
        "data_source": "gsm8k",
        "ground_truth": 
    }
    """
    
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Any = None, 
        config: Dict[str, Any] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for offline rollout training")
        
        self.config = config if config is not None else {}
        self.max_samples = max_samples

        # ==========
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.max_response_length = self.config.get("max_response_length", 2048)
        self.truncation = self.config.get("truncation", "error")
        self.samples_per_prompt = self.config.get("offline_samples_per_prompt", None)
        self.pad_token_id = tokenizer.pad_token_id
        
        self.validate_data = self.config.get("validate_data", False)

        self.load_rollout_log_probs = bool(self.config.get("offline_load_rollout_log_probs", True))

        # ======
        if not isinstance(data_files, list):
            data_files = [data_files]
        
        self.grouped_data = defaultdict(list)
        
        print(f"\n{'='*80}")
        print(f"Loading Offline Rollout Dataset")
        print(f"{'='*80}")
        print(f"Data files: {len(data_files)}")
        
        total_samples = 0
        for file_idx, file_path in enumerate(data_files):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Offline rollout file not found: {file_path}")
            
            print(f"Loading file {file_idx+1}/{len(data_files)}: {file_path}")
            
            # Robust per-line decode to avoid crashes during iteration
            with open(file_path, 'rb') as fin:
                for line_idx, raw in enumerate(fin):
                    if not raw.strip():
                        continue

                    try:
                        line = raw.decode('utf-8')
                    except UnicodeDecodeError as e:
                        try:
                            line = raw.decode('utf-8', errors='replace')
                            print(f"Warning: Decoding error in {file_path} line {line_idx}: {e}. Replaced invalid bytes.")
                        except Exception:
                            line = raw.decode('latin-1', errors='replace')
                            print(f"Warning: Decoded {file_path} line {line_idx} with Latin-1; invalid UTF-8 bytes replaced.")

                    try:
                        item = json.loads(line)

                        required_fields = ["prompt_ids", "response_ids"]
                        if self.load_rollout_log_probs:
                            required_fields.append("small_model_log_probs")

                        missing_fields = [f for f in required_fields if f not in item]
                        if missing_fields:
                            print(f"Warning: Line {line_idx} missing fields {missing_fields}, skipping")
                            continue

                        prompt_id = item.get("prompt_id", None)
                        if prompt_id is None:
                            prompt_text = item.get("prompt", "")
                            prompt_id = f"hash_{hash(prompt_text):016x}"
                            item["prompt_id"] = prompt_id

                        if not self.load_rollout_log_probs:
                            for k in ("small_model_log_probs", "small_model_logits", "logits", "teacher_logits"):
                                item.pop(k, None)

                        self.grouped_data[prompt_id].append(item)
                        total_samples += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_idx}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing line {line_idx}: {e}")
                        continue
        
        self.prompt_ids_list = sorted(list(self.grouped_data.keys()))
        
        print(f"Loaded {total_samples} samples grouped into {len(self.prompt_ids_list)} prompts")
        
        # ==========
        samples_per_prompt_counts = [len(samples) for samples in self.grouped_data.values()]
        if samples_per_prompt_counts:
            print(f"\nSamples per prompt distribution:")
            print(f"  - Min: {min(samples_per_prompt_counts)}")
            print(f"  - Max: {max(samples_per_prompt_counts)}")
            print(f"  - Mean: {np.mean(samples_per_prompt_counts):.1f}")
            print(f"  - Median: {np.median(samples_per_prompt_counts):.1f}")
        
        if self.samples_per_prompt is not None:
            print(f"\nAdjusting to {self.samples_per_prompt} samples per prompt...")
            import random
            random.seed(self.config.get("seed", 42))
            
            adjusted_count = 0
            for prompt_id in self.prompt_ids_list:
                samples = self.grouped_data[prompt_id]
                current_n = len(samples)
                
                if current_n > self.samples_per_prompt:
                    # Over-sampling: 随机选择
                    self.grouped_data[prompt_id] = random.sample(samples, self.samples_per_prompt)
                    adjusted_count += 1
                elif current_n < self.samples_per_prompt:
                    # Under-sampling: 随机重复
                    additional_needed = self.samples_per_prompt - current_n
                    additional_samples = random.choices(samples, k=additional_needed)
                    self.grouped_data[prompt_id] = samples + additional_samples
                    adjusted_count += 1
            
            print(f"Adjusted {adjusted_count} prompts")
        
        # ========
        total_samples_after = sum(len(samples) for samples in self.grouped_data.values())
        correct_samples = sum(
            1
            for samples in self.grouped_data.values()
            for item in samples
            if item.get("is_correct", False)
        )
        
        data_source_counts = defaultdict(int)
        for samples in self.grouped_data.values():
            for item in samples:
                source = item.get("data_source", "unknown")
                data_source_counts[source] += 1

        print(f"\n{'='*80}")
        print(f"Final Dataset Statistics")
        print(f"{'='*80}")
        print(f"Total prompts: {len(self.prompt_ids_list)}")
        print(f"Total samples: {total_samples_after}")
        print(f"Correct samples: {correct_samples}")
        if total_samples_after > 0:
            print(f"Accuracy: {correct_samples / total_samples_after:.2%}")
        else:
            print("Accuracy: N/A (no samples)")
        
        if self.samples_per_prompt:
            print(f"Samples per prompt: {self.samples_per_prompt} (fixed)")
        
        if data_source_counts:
            print(f"\nData source distribution:")
            for source, count in sorted(data_source_counts.items()):
                print(f"  - {source}: {count} ({count/total_samples_after:.1%})")
        
        print(f"{'='*80}\n")
        
        # ==========
        if max_samples > 0 and len(self.prompt_ids_list) > max_samples:
            import random
            random.seed(self.config.get("seed", 42))
            self.prompt_ids_list = random.sample(self.prompt_ids_list, max_samples)
            print(f"Randomly sampled {max_samples} prompts for training")
    
    def __len__(self) -> int:
        return len(self.prompt_ids_list)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
       
        prompt_id = self.prompt_ids_list[idx]
        samples = self.grouped_data[prompt_id]

        processed_samples = []
        for sample_idx, item in enumerate(samples):
            try:
                processed_sample = self._process_single_sample(item, prompt_id)
                
                if self.validate_data:
                    self._validate_sample(processed_sample, prompt_id, sample_idx)
                
                processed_samples.append(processed_sample)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx} for prompt_id {prompt_id}: {e}")
                if self.validate_data:
                    import traceback
                    traceback.print_exc()
                continue
        
        if not processed_samples:
            raise ValueError(f"No valid samples for prompt_id {prompt_id}")
        
        return processed_samples
    
    def _process_single_sample(self, item: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
        prompt_ids = torch.tensor(item["prompt_ids"], dtype=torch.long)
        prompt_length = prompt_ids.size(0)

        if prompt_length > self.max_prompt_length:
            if self.truncation == "left":
                prompt_ids = prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                prompt_ids = prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                prompt_ids = torch.cat((prompt_ids[:left_half], prompt_ids[-right_half:]), dim=0)
            elif self.truncation == "error":
                print(
                    f"Warning: Prompt length {prompt_length} exceeds max {self.max_prompt_length} "
                    f"for prompt_id {prompt_id}; falling back to right truncation."
                )
                prompt_ids = prompt_ids[:self.max_prompt_length]
            else:
                raise ValueError(f"Unsupported truncation mode: {self.truncation}")
            prompt_length = prompt_ids.size(0)

        if prompt_length < self.max_prompt_length:
            pad_length = self.max_prompt_length - prompt_length
            pad_tensor = torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            prompt_ids = torch.cat((pad_tensor, prompt_ids), dim=0)
            prompt_attention_mask = torch.cat(
                (
                    torch.zeros(pad_length, dtype=torch.long),
                    torch.ones(prompt_length, dtype=torch.long),
                ),
                dim=0,
            )
        else:
            prompt_attention_mask = torch.ones(self.max_prompt_length, dtype=torch.long)

        position_ids = compute_position_id_with_mask(prompt_attention_mask.unsqueeze(0))[0]

        response_ids = torch.tensor(item["response_ids"], dtype=torch.long)
        original_response_length = response_ids.size(0)
        actual_response_length = min(original_response_length, self.max_response_length)
        response_ids = response_ids[:actual_response_length]

        if actual_response_length < self.max_response_length:
            pad_length = self.max_response_length - actual_response_length
            pad_tensor = torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            response_ids = torch.cat((response_ids, pad_tensor), dim=0)

        rollout_log_probs = None
        if self.load_rollout_log_probs:
            if "small_model_log_probs" not in item:
                raise KeyError(
                    f"Missing 'small_model_log_probs' for prompt_id={prompt_id} "
                    f"while offline_load_rollout_log_probs=True"
                )

            rollout_log_probs = torch.tensor(item["small_model_log_probs"], dtype=torch.float32)
            original_log_probs_length = rollout_log_probs.size(0)

            if original_log_probs_length != original_response_length:
                print(
                    f"Warning: log_probs length ({original_log_probs_length}) != "
                    f"response_ids length ({original_response_length}) for prompt_id {prompt_id}"
                )

            rollout_log_probs = rollout_log_probs[:actual_response_length]

            if rollout_log_probs.numel() < actual_response_length:
                pad_length = actual_response_length - rollout_log_probs.numel()
                rollout_log_probs = torch.cat(
                    (rollout_log_probs, torch.zeros(pad_length, dtype=torch.float32)), dim=0
                )

            if rollout_log_probs.numel() < self.max_response_length:
                pad_length = self.max_response_length - rollout_log_probs.numel()
                rollout_log_probs = torch.cat(
                    (rollout_log_probs, torch.zeros(pad_length, dtype=torch.float32)), dim=0
                )

        response_mask = torch.zeros(self.max_response_length, dtype=torch.long)
        if actual_response_length > 0:
            response_mask[:actual_response_length] = 1

        reward_value = float(item.get("reward", 0.0))
        
        token_level_scores = torch.zeros(self.max_response_length, dtype=torch.float32)
        if actual_response_length > 0:
            token_level_scores[actual_response_length - 1] = reward_value

        result = {
            "input_ids": prompt_ids,                    # [max_prompt_length]
            "attention_mask": prompt_attention_mask,    # [max_prompt_length]
            "position_ids": position_ids,               # [max_prompt_length]
            "uid": prompt_id, 
            "data_source": item.get("data_source", "offline_rollout"),

            "responses": response_ids,                  # [max_response_length]
            "response_mask": response_mask,             # [max_response_length]

            "token_level_scores": token_level_scores,   # [max_response_length]
            "offline_reward": torch.tensor(reward_value, dtype=torch.float32),
            "is_correct": item.get("is_correct", False),
        }

        if rollout_log_probs is not None:
            result["rollout_log_probs"] = rollout_log_probs

        if "ground_truth" in item or "reward_model" in item:
            result["reward_model"] = {
                "style": "offline",
                "ground_truth": item.get("ground_truth", None),
            }

        return result
    
    def _validate_sample(self, sample: Dict[str, Any], prompt_id: str, sample_idx: int):
        required_fields = [
            "input_ids", "attention_mask", "position_ids",
            "responses", "response_mask", "uid"
        ]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing required field '{field}' in sample")
        
        expected_shapes = {
            "input_ids": (self.max_prompt_length,),
            "attention_mask": (self.max_prompt_length,),
            "position_ids": (self.max_prompt_length,),
            "responses": (self.max_response_length,),
            "response_mask": (self.max_response_length,),
        }
        
        for field, expected_shape in expected_shapes.items():
            if sample[field].shape != expected_shape:
                raise ValueError(
                    f"Invalid shape for '{field}': expected {expected_shape}, "
                    f"got {sample[field].shape}"
                )
        
        if sample["uid"] != prompt_id:
            raise ValueError(
                f"UID mismatch: expected '{prompt_id}', got '{sample['uid']}'"
            )
        
        if "rollout_log_probs" in sample:
            if sample["rollout_log_probs"].shape != (self.max_response_length,):
                raise ValueError(
                    f"Invalid rollout_log_probs shape: "
                    f"expected ({self.max_response_length},), "
                    f"got {sample['rollout_log_probs'].shape}"
                )
    
    def __getstate__(self):
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def resume_dataset_state(self):
        pass


def offline_rollout_collate_fn(batch_list: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    
    Args:
        batch_list: [
            [sample1, sample2, ..., sampleN], 
            [sample1, sample2, ..., sampleN], 
            ...
        ]
    
    Returns:
        Dict: 
        {
            "input_ids": torch.Tensor [batch_size, max_prompt_length],
            "attention_mask": torch.Tensor [batch_size, max_prompt_length],
            "responses": torch.Tensor [batch_size, max_response_length],
            ...
            
            "uid": np.ndarray [batch_size],  
            "data_source": np.ndarray [batch_size],
            ...
        }
    """
    all_samples = []
    for samples_group in batch_list:
        if isinstance(samples_group, list):
            all_samples.extend(samples_group)
        else:
            all_samples.append(samples_group)
    
    if not all_samples:
        raise ValueError("Empty batch after flattening")
    
    tensor_keys = [
        "input_ids",
        "attention_mask", 
        "position_ids",
        "responses",
        "response_mask",
        "rollout_log_probs",
        "token_level_scores",
        "offline_reward",
    ]
    
    batch_dict = {}
    
    optional_tensor_keys = {"rollout_log_probs"}

    for key in tensor_keys:
        if key not in all_samples[0]:
            continue

        try:
            tensors = []
            none_or_missing = 0
            for s in all_samples:
                v = s.get(key, None)
                if v is None:
                    none_or_missing += 1
                else:
                    tensors.append(v)

            if none_or_missing > 0:
                if key in optional_tensor_keys:
                    continue
                raise TypeError(f"'{key}' has {none_or_missing} None/missing values in batch")

            batch_dict[key] = torch.stack(tensors, dim=0)
        except Exception as e:
            print(f"Error stacking '{key}': {e}")
            print(f"Number of samples: {len(all_samples)}")
            preview = []
            for s in all_samples[:3]:
                v = s.get(key, None)
                preview.append(None if v is None else tuple(v.shape))
            print(f"First 3 shapes: {preview}")
            raise
    
    non_tensor_keys = ["uid", "data_source", "is_correct"]
    for key in non_tensor_keys:
        if key in all_samples[0]:
            batch_dict[key] = np.array([s[key] for s in all_samples], dtype=object)
    

    if "reward_model" in all_samples[0]:
        # batch_dict["reward_model"] = [s["reward_model"] for s in all_samples]
        batch_dict["reward_model"] = np.array([s["reward_model"] for s in all_samples], dtype=object)
    
    return batch_dict