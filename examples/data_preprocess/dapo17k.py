#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preprocess the DAPO-Math-17k dataset to parquet format.

注意：
- Hugging Face 上的 BytedTsinghua-SIA/DAPO-Math-17k 已经是 VERL 格式：
  [data_source, prompt, ability, reward_model, extra_info]
  所以默认直接原样保存即可，不需要重新从 question/answer 构造。

- 只有当你有一份“原始版 dapo 数据”（带 question/answer 字段）时，
  才需要用到 raw_mode=True 和下面的 GSM8K 风格处理逻辑。
"""

import argparse
import os
import re
from typing import Dict, Any

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution_from_text(solution_str: str) -> str:
    """
    如果你用 raw_mode 从包含 '#### answer' 的解答字符串中抽最终答案，
    可以使用这个函数；当前 DAPO HF 官方数据集已经给了 ground_truth，
    实际上用不到这个逻辑。
    """
    m = re.search(r"####\s*([-0-9\.\,]+)", solution_str)
    assert m is not None, f"Cannot find '####' style answer in: {solution_str!r}"
    final_solution = m.group(1)
    return final_solution.replace(",", "").strip()


def make_gsm8k_style_map_fn(
    split: str,
    question_field: str,
    answer_field: str,
    data_source: str,
    instruction_following: str,
):
    """
    只有 raw_mode=True 且数据集中确实有 question/answer 字段时才会用到。
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question_raw = example[question_field]
        answer_raw = example[answer_field]

        question = question_raw + " " + instruction_following
        solution = extract_solution_from_text(answer_raw)

        return {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

    return process_fn


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    # 转换data_source为小写以进行大小写不敏感的比较
    data_source_lower = data_source.lower()
    
    if data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward
        res = math_reward.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument(
        "--local_dir",
        default=None,
        help="(Deprecated) use --local_save_dir instead.",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset, if it exists. "
             "If not set, will load from HF hub (BytedTsinghua-SIA/DAPO-Math-17k by default).",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo17k",
        help="The save directory for the preprocessed dataset.",
    )

    # 数据源 & 字段名
    parser.add_argument(
        "--data_source",
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="HF dataset name & value recorded in 'data_source' field.",
    )

    # raw_mode：只有“原始版 dapo”才用。默认 False：直接使用 HF 官方 VERL 格式
    parser.add_argument(
        "--raw_mode",
        action="store_true",
        help="If set, assume the dataset has `question` / `answer` style fields "
             "and convert it to VERL schema like GSM8K. "
             "If not set (default), assume it's already in VERL format and just save.",
    )
    parser.add_argument(
        "--question_field",
        default="question",
        help="Field name for question when using raw_mode.",
    )
    parser.add_argument(
        "--answer_field",
        default="answer",
        help="Field name for answer when using raw_mode.",
    )
    parser.add_argument(
        "--instruction_following",
        default="Let's think step by step and output the final answer after \"####\".",
        help="Instruction appended to question when using raw_mode.",
    )

    args = parser.parse_args()

    # 1. 加载数据集：本地优先，否则从 HF hub 加载
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(args.data_source)

    available_splits = list(dataset.keys())
    print(f"[INFO] Available splits in dataset: {available_splits}")

    # DAPO HF 官方只有 train 一个 split
    train_split_name = "train" if "train" in dataset else available_splits[0]
    train_dataset = dataset[train_split_name]
    test_dataset = dataset["test"] if "test" in dataset else None
    if test_dataset is None:
        print("[WARN] No 'test' split found; will only export train.parquet")

    # 2. 判断当前 schema 是否已经是 VERL 格式
    features = train_dataset.features
    has_verl_schema = (
        "prompt" in features
        and "reward_model" in features
        and "ability" in features
        and "extra_info" in features
    )

    print(f"[INFO] has_verl_schema = {has_verl_schema}, raw_mode = {args.raw_mode}")

    # 3A. 已是 VERL 且 raw_mode 未开启：直接原样保存（推荐 DAPO 默认）
    if has_verl_schema and not args.raw_mode:
        print("[INFO] Detected VERL-style schema. Will save dataset as-is.")

        local_save_dir = args.local_dir or args.local_save_dir
        if args.local_dir is not None:
            print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
        local_save_dir = os.path.expanduser(local_save_dir)
        os.makedirs(local_save_dir, exist_ok=True)

        train_out = os.path.join(local_save_dir, "train.parquet")
        print(f"[INFO] Saving train split to: {train_out}")
        train_dataset.to_parquet(train_out)

        if test_dataset is not None:
            test_out = os.path.join(local_save_dir, "test.parquet")
            print(f"[INFO] Saving test split to: {test_out}")
            test_dataset.to_parquet(test_out)

        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            print(f"[INFO] Copying {local_save_dir} -> HDFS {args.hdfs_dir}")
            copy(src=local_save_dir, dst=args.hdfs_dir)

    # 3B. 不是 VERL 或你强制 raw_mode=True：走 GSM8K 风格转换
    else:
        print("[INFO] Will convert from raw question/answer style to VERL schema (GSM8K-like).")

        # 检查字段是否存在
        if args.question_field not in train_dataset.column_names or \
           args.answer_field not in train_dataset.column_names:
            raise KeyError(
                f"Expected fields '{args.question_field}' and '{args.answer_field}' "
                f"in dataset columns {train_dataset.column_names}, "
                f"but at least one is missing."
            )

        train_dataset = train_dataset.map(
            function=make_gsm8k_style_map_fn(
                split=train_split_name,
                question_field=args.question_field,
                answer_field=args.answer_field,
                data_source=args.data_source,
                instruction_following=args.instruction_following,
            ),
            with_indices=True,
        )

        if test_dataset is not None:
            test_dataset = test_dataset.map(
                function=make_gsm8k_style_map_fn(
                    split="test",
                    question_field=args.question_field,
                    answer_field=args.answer_field,
                    data_source=args.data_source,
                    instruction_following=args.instruction_following,
                ),
                with_indices=True,
            )

        local_save_dir = args.local_dir or args.local_save_dir
        if args.local_dir is not None:
            print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
        local_save_dir = os.path.expanduser(local_save_dir)
        os.makedirs(local_save_dir, exist_ok=True)

        train_out = os.path.join(local_save_dir, "train.parquet")
        print(f"[INFO] Saving converted train split to: {train_out}")
        train_dataset.to_parquet(train_out)

        if test_dataset is not None:
            test_out = os.path.join(local_save_dir, "test.parquet")
            print(f"[INFO] Saving converted test split to: {test_out}")
            test_dataset.to_parquet(test_out)

        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            print(f"[INFO] Copying {local_save_dir} -> HDFS {args.hdfs_dir}")
            copy(src=local_save_dir, dst=args.hdfs_dir)