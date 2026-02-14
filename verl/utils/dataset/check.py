#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, sys
from typing import Any, Dict, List

import pyarrow.parquet as pq
import pyarrow as pa

REQUIRED_TOP_LEVEL = ["prompt", "reward_model", "data_source"]
REQUIRED_REWARD_KEYS = ["style", "ground_truth"]

def is_chat_prompt(x: Any) -> bool:
    """prompt 应为 [{'role': str, 'content': str}, ...]"""
    if not isinstance(x, list) or len(x) == 0:
        return False
    for m in x:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
        if not isinstance(m["role"], str) or not isinstance(m["content"], str):
            return False
    return True

def brief_prompt(p: List[Dict[str, str]], max_chars: int = 160) -> str:
    # 仅拼接 user 段落做预览
    users = [m["content"] for m in p if m.get("role") == "user"]
    text = "\n---\n".join(users) if users else p[0].get("content", "")
    text = text.strip().replace("\r", " ").replace("\n", " ")
    return (text[:max_chars] + "…") if len(text) > max_chars else text

def validate_row(row: Dict[str, Any], idx: int) -> List[str]:
    errs = []
    for k in REQUIRED_TOP_LEVEL:
        if k not in row:
            errs.append(f"[{idx}] missing key: '{k}'")
    if "prompt" in row and not is_chat_prompt(row["prompt"]):
        errs.append(f"[{idx}] 'prompt' must be a non-empty chat list with role/content")
    if "reward_model" in row:
        rm = row["reward_model"]
        if not isinstance(rm, dict):
            errs.append(f"[{idx}] 'reward_model' must be dict")
        else:
            for rk in REQUIRED_REWARD_KEYS:
                if rk not in rm:
                    errs.append(f"[{idx}] reward_model missing '{rk}'")
            if "style" in rm and not isinstance(rm["style"], str):
                errs.append(f"[{idx}] reward_model.style must be str")
            if "ground_truth" in rm and not isinstance(rm["ground_truth"], str):
                # 允许数字，但统一成字符串更稳
                if not isinstance(rm["ground_truth"], (int, float)):
                    errs.append(f"[{idx}] reward_model.ground_truth must be str/number")
    if "data_source" in row and not isinstance(row["data_source"], str):
        errs.append(f"[{idx}] data_source must be str")
    return errs

def to_pylist(table: pa.Table) -> List[Dict[str, Any]]:
    # 直接转 python list（pyarrow 会把 struct/list 列转为 Python 对象）
    return table.to_pylist()

def main():
    ap = argparse.ArgumentParser(description="Verify VeRL dataset parquet and print 3 samples.")
    ap.add_argument("--parquet", default="/mnt/petrelfs/renyiming/verl/data/train.parquet",
                    help="Path to parquet file (default: ./data/dapo17k_verl/train.parquet)")
    ap.add_argument("--show", type=int, default=3, help="How many examples to print (default: 3)")
    args = ap.parse_args()

    path = args.parquet
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)

    try:
        table = pq.read_table(path)
    except Exception as e:
        print(f"❌ Failed to read parquet: {e}")
        sys.exit(1)

    n_rows = table.num_rows
    n_cols = table.num_columns
    print(f"✅ Loaded parquet: {path}")
    print(f"   Rows: {n_rows}, Cols: {n_cols}")
    print(f"   Columns: {[c for c in table.column_names]}")

    data = to_pylist(table)

    if n_rows == 0:
        print("⚠ Dataset is empty.")
        sys.exit(0)

    # 全量扫一遍做快速验证（可按需抽样）
    errors = []
    for i, row in enumerate(data):
        errors.extend(validate_row(row, i))

    if errors:
        print("❌ Schema or value errors found:")
        for e in errors[:20]:
            print("   -", e)
        if len(errors) > 20:
            print(f"   … and {len(errors)-20} more.")
        sys.exit(2)
    else:
        print("✅ Basic schema/type checks passed.")

    # 打印前 N 条
    k = min(args.show, n_rows)
    print(f"\n=== Show {k} sample(s) ===")
    for i in range(k):
        row = data[i]
        prompt_preview = brief_prompt(row["prompt"])
        rm = row["reward_model"]
        out = {
            "idx": i,
            "data_source": row.get("data_source", ""),
            "prompt_preview": prompt_preview,
            "reward_style": rm.get("style", ""),
            "ground_truth": rm.get("ground_truth", ""),
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n🎉 Verification completed successfully.")

if __name__ == "__main__":
    main()
