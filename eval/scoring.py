
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import math


def normalize_answer(answer) -> str:
    if answer is None:
        return ""

    try:
        s = str(answer)
    except Exception:
        return ""

    s = s.strip().lower()

    s = re.sub(r"\s+", "", s)

    s = s.replace("\\", "")

    s = re.sub(r'frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)

    s = s.replace("$", "")

    s = re.sub(r'(\d),(\d)', r'\1\2', s)

    s = re.sub(r'^=+', '', s)

    for _ in range(3):
        s = re.sub(r'^\((.*)\)$', r'\1', s)

    s = s.rstrip("。．.!！；;，,。")

    return s.strip()


def _try_parse_fraction(x: str):
    if "/" in x:
        parts = x.split("/")
        if len(parts) == 2:
            num, den = parts
            try:
                return float(num) / float(den)
            except Exception:
                return None
    return None


def _try_parse_percent(x: str):
    if x.endswith("%"):
        try:
            return float(x[:-1]) / 100.0
        except Exception:
            return None
    return None


def _try_parse_number(x: str):
    frac = _try_parse_fraction(x)
    if frac is not None:
        return frac

    pct = _try_parse_percent(x)
    if pct is not None:
        return pct

    try:
        return float(x)
    except Exception:
        return None


def is_equiv(pred, gold) -> bool:

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    p_num = _try_parse_number(pred_norm)
    g_num = _try_parse_number(gold_norm)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-6

    if gold_norm and gold_norm in pred_norm:
        return True

    return False


# ------------------------
#  Pass@k 
# ------------------------

def _comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or k > n:
        return 0
    try:
        return math.comb(n, k)  # type: ignore[attr-defined]
    except Exception:

        k = min(k, n - k)
        if k < 0:
            return 0
        res = 1
        for i in range(1, k + 1):
            res = res * (n - i + 1) // i
        return res


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n <= 0:
        return 0.0

    k_eff = min(max(k, 0), n)
    if k_eff == 0:
        return 0.0
    if c <= 0:
        return 0.0

    if k_eff > (n - c):
        return 1.0

    denom = _comb(n, k_eff)
    if denom == 0:
        return 0.0

    num = _comb(n - c, k_eff)
    return 1.0 - (num / denom)



def score_single_example(predicted_answers: List[Any], gold_answer: Any) -> Dict[str, float]:

    n = len(predicted_answers)
    if n == 0:
        return {}

    correct_count = sum(1 for pred in predicted_answers if is_equiv(pred, gold_answer))

    results = {}
    for k in range(1, n + 1):
        results[f"pass@{k}"] = calculate_pass_at_k(n, correct_count, k)

    return results


def score_benchmark(answer_file: Path) -> Dict[str, Any]:
    print(f"Scoring: {answer_file}")

    if not answer_file.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_file}")

    all_scores: List[Dict[str, float]] = []
    total_examples = 0
    correct_at_least_once = 0

    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())


            gold_answer = item.get("answer", "")
            predicted_answers = item.get("predicted_answers", [])
            if not isinstance(predicted_answers, list):
                predicted_answers = [predicted_answers]

            scores = score_single_example(predicted_answers, gold_answer)
            if scores:  
                all_scores.append(scores)

            total_examples += 1

            if any(is_equiv(pred, gold_answer) for pred in predicted_answers):
                correct_at_least_once += 1

    if not all_scores:
        return {
            "benchmark": answer_file.stem,
            "total_examples": total_examples,
            "correct_at_least_once": correct_at_least_once,
            "accuracy_any": (correct_at_least_once / total_examples) if total_examples else 0.0,
            "pass_at_k": {},
        }

    k_values = sorted(int(k.split('@')[1]) for k in all_scores[0].keys())
    avg_scores: Dict[str, float] = {}
    for k in k_values:
        key = f"pass@{k}"
        avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)

    result = {
        "benchmark": answer_file.stem,
        "total_examples": total_examples,
        "correct_at_least_once": correct_at_least_once,
        "accuracy_any": correct_at_least_once / total_examples if total_examples > 0 else 0.0,
        "pass_at_k": avg_scores,
    }
    return result


def score_all_benchmarks(output_dir: Path, result_dir: Path):
    print(f"\n{'='*80}")
    print(f"Scoring benchmarks from: {output_dir}")
    print(f"{'='*80}\n")

    benchmarks = ["aime24", "aime25", "math500"]
    all_results = []

    for benchmark in benchmarks:
        answer_file = output_dir / f"{benchmark}.jsonl"
        if not answer_file.exists():
            print(f"Warning: {answer_file} not found, skipping...\n")
            continue

        result = score_benchmark(answer_file)
        all_results.append(result)

        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{benchmark}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {result_file}\n")

    print(f"{'='*80}")
    print(f"Summary Results for {output_dir.name}")
    print(f"{'='*80}\n")
    for result in all_results:
        print(f"Benchmark: {result['benchmark']}")
        print(f"  Total Examples: {result['total_examples']}")
        print(f"  Correct (at least once): {result['correct_at_least_once']} ({result['accuracy_any']:.2%})")
        print(f"  Pass@k metrics:")
        for k, score in sorted(result['pass_at_k'].items(), key=lambda x: int(x[0].split('@')[1])):
            print(f"    {k}: {score:.4f}")
        print()

    summary_file = result_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")
