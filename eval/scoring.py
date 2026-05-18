"""
打分脚本 - 计算Pass@k指标（修正版，修复 int 没有 strip 的报错）
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import math


# ------------------------
#  答案标准化 & 等价判断
# ------------------------

def normalize_answer(answer) -> str:
    """
    标准化答案格式用于比较

    处理要点：
    - 先统一转成字符串；对 None/NaN/非字符串类型安全处理
    - 统一小写 & 去除首尾空白
    - 去除所有空格与反斜杠（便于处理 LaTeX）
    - \frac{a}{b} -> a/b
    - 去掉 $ 符号、千分位逗号
    - 去掉包裹性的括号和等号（如 "= 42", "(42)"）
    - 去掉句尾常见标点
    """
    if answer is None:
        return ""

    try:
        s = str(answer)
    except Exception:
        # 极端情况下，无法转字符串则置空
        return ""

    s = s.strip().lower()

    # 移除所有空格（含不间断空格）
    s = re.sub(r"\s+", "", s)

    # 移除反斜杠（便于 \frac / \% / \pm 等）
    s = s.replace("\\", "")

    # LaTeX 分数：frac{a}{b} -> a/b
    s = re.sub(r'frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', s)

    # 去除美元符号（数学内联）
    s = s.replace("$", "")

    # 千分位 1,000,000 -> 1000000
    s = re.sub(r'(\d),(\d)', r'\1\2', s)

    # 去掉开头的等号（=42 或 ==42）
    s = re.sub(r'^=+', '', s)

    # 去掉包裹性的括号，例如 "(42)" -> "42"；多层括号也尽量处理
    for _ in range(3):
        s = re.sub(r'^\((.*)\)$', r'\1', s)

    # 去掉句尾中英文标点
    s = s.rstrip("。．.!！；;，,。")

    return s.strip()


def _try_parse_fraction(x: str):
    """尝试解析 a/b 形式为 float。失败返回 None。"""
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
    """尝试解析百分号结尾（50% -> 0.5）。失败返回 None。"""
    if x.endswith("%"):
        try:
            return float(x[:-1]) / 100.0
        except Exception:
            return None
    return None


def _try_parse_number(x: str):
    """尽力把标准化后的字符串解析为数值（含分数、百分比、科学计数法）。
       成功返回 float，失败返回 None。"""
    # 分数优先
    frac = _try_parse_fraction(x)
    if frac is not None:
        return frac

    # 百分比
    pct = _try_parse_percent(x)
    if pct is not None:
        return pct

    # 普通/科学计数法
    try:
        return float(x)
    except Exception:
        return None


def is_equiv(pred, gold) -> bool:
    """
    判断预测答案和标准答案是否等价
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # 1) 直接字符串相等
    if pred_norm == gold_norm:
        return True

    # 2) 数值比较（含分数/百分比/科学计数法）
    p_num = _try_parse_number(pred_norm)
    g_num = _try_parse_number(gold_norm)
    if p_num is not None and g_num is not None:
        return abs(p_num - g_num) < 1e-6

    # 3) 容错：gold 在 pred 中（常见于“答案在解释里”）
    if gold_norm and gold_norm in pred_norm:
        return True

    return False


# ------------------------
#  Pass@k 计算
# ------------------------

def _comb(n: int, k: int) -> int:
    """整型组合数 C(n,k)。优先用 math.comb，否则降级实现。"""
    if k < 0 or n < 0 or k > n:
        return 0
    try:
        return math.comb(n, k)  # type: ignore[attr-defined]
    except Exception:
        # 简单整数实现
        k = min(k, n - k)
        if k < 0:
            return 0
        res = 1
        for i in range(1, k + 1):
            res = res * (n - i + 1) // i
        return res


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 Pass@k（不放回）
    Pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n <= 0:
        return 0.0

    k_eff = min(max(k, 0), n)
    if k_eff == 0:
        return 0.0
    if c <= 0:
        return 0.0

    # 必中条件：抽取数量超过所有错误样本数
    if k_eff > (n - c):
        return 1.0

    denom = _comb(n, k_eff)
    if denom == 0:
        return 0.0

    num = _comb(n - c, k_eff)
    return 1.0 - (num / denom)


# ------------------------
#  单样本与整基准评测
# ------------------------

def score_single_example(predicted_answers: List[Any], gold_answer: Any) -> Dict[str, float]:
    """
    对单个样本计算各个 Pass@k（k = 1..n）
    备注：predicted_answers 里可能包含非字符串（例如数字），统一在 is_equiv 内部处理。
    """
    n = len(predicted_answers)
    if n == 0:
        return {}

    correct_count = sum(1 for pred in predicted_answers if is_equiv(pred, gold_answer))

    results = {}
    for k in range(1, n + 1):
        results[f"pass@{k}"] = calculate_pass_at_k(n, correct_count, k)

    return results


def score_benchmark(answer_file: Path) -> Dict[str, Any]:
    """
    对整个 benchmark 打分；answer_file 为由“答案提取脚本”产出的 xxx.jsonl
    """
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

            # 容错：缺字段时跳过该样本
            gold_answer = item.get("answer", "")
            predicted_answers = item.get("predicted_answers", [])
            if not isinstance(predicted_answers, list):
                predicted_answers = [predicted_answers]

            scores = score_single_example(predicted_answers, gold_answer)
            if scores:  # 跳过空
                all_scores.append(scores)

            total_examples += 1

            if any(is_equiv(pred, gold_answer) for pred in predicted_answers):
                correct_at_least_once += 1

    if not all_scores:
        # 防御性：空文件或全空
        return {
            "benchmark": answer_file.stem,
            "total_examples": total_examples,
            "correct_at_least_once": correct_at_least_once,
            "accuracy_any": (correct_at_least_once / total_examples) if total_examples else 0.0,
            "pass_at_k": {},
        }

    # 计算平均 Pass@k（对齐第一条的 k 列表）
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
    """
    对指定输出目录下的所有 benchmark 打分
    """
    print(f"\n{'='*80}")
    print(f"Scoring benchmarks from: {output_dir}")
    print(f"{'='*80}\n")

    benchmarks = ["aime24", "aime25", "math500", "olympiadbench"]
    all_results = []

    for benchmark in benchmarks:
        answer_file = output_dir / f"{benchmark}.jsonl"
        if not answer_file.exists():
            print(f"Warning: {answer_file} not found, skipping...\n")
            continue

        result = score_benchmark(answer_file)
        all_results.append(result)

        # 保存单个 benchmark 结果
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{benchmark}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {result_file}\n")

    # 汇总打印
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

    # 保存汇总
    summary_file = result_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}\n")


# ------------------------
#  验证（含更多易错案例）
# ------------------------

def verify_normalize_answer():
    """验证 normalize_answer"""
    print("Testing normalize_answer...")
    test_cases = [
        (42, "42"),                      # 数值型
        (3.14, "3.14"),
        ("42", "42"),
        ("  42  ", "42"),
        ("\\frac{2}{3}", "2/3"),
        ("$100$", "100"),
        ("1,000", "1000"),
        ("Answer: 42", "answer:42"),
        ("(  3.14 )", "3.14"),
        ("==42。", "42"),
        ("\\frac{  10 }{  20 }", "10/20"),
    ]
    for raw, expected in test_cases:
        got = normalize_answer(raw)
        assert got == expected, f"normalize '{raw}' -> '{got}', expected '{expected}'"
        print(f"  ✓ '{raw}' -> '{got}'")
    print("✓ normalize_answer OK\n")


def verify_is_equiv():
    """验证 is_equiv（覆盖更多等价场景）"""
    print("Testing is_equiv...")

    tests = [
        (42, "42", True),                # 数字 vs 字符串
        ("42", 42, True),
        (0.5, "50%", True),
        ("2/3", "\\frac{2}{3}", True),
        ("The answer is 42.", "42", True),
        ("0.6666666667", "2/3", True),
        ("1e3", "1000", True),
        ("1,000", "1000", True),
        ("abc", "def", False),
        ("42", "43", False),
    ]
    for pred, gold, expected in tests:
        got = is_equiv(pred, gold)
        assert got == expected, f"is_equiv('{pred}', '{gold}') = {got}, expected {expected}"
        print(f"  ✓ is_equiv('{pred}', '{gold}') = {got}")
    print("✓ is_equiv OK\n")


def verify_calculate_pass_at_k():
    """验证 calculate_pass_at_k（含边界与易错点）"""
    print("Testing calculate_pass_at_k...")

    def close(a, b, eps=1e-9):
        return abs(a - b) < eps

    # 全对 -> 任意 k 都 1
    assert close(calculate_pass_at_k(10, 10, 1), 1.0)
    assert close(calculate_pass_at_k(10, 10, 7), 1.0)

    # 全错 -> 任意 k 都 0
    assert close(calculate_pass_at_k(10, 0, 1), 0.0)
    assert close(calculate_pass_at_k(10, 0, 7), 0.0)

    # 基本概率：n=10, c=1, k=5 -> 0.5
    v = calculate_pass_at_k(10, 1, 5)
    assert close(v, 0.5), f"expected 0.5, got {v}"

    # 关键边界：k > n - c -> 必中
    assert close(calculate_pass_at_k(10, 7, 4), 1.0)

    # 反例：c >= k 并不保证 1
    val = calculate_pass_at_k(20, 6, 6)
    assert 0.0 < val < 1.0

    # k > n：按 k_eff = n 处理；只要 c>0 即为 1
    assert close(calculate_pass_at_k(5, 2, 10), 1.0)

    # n=1 边界
    assert close(calculate_pass_at_k(1, 1, 1), 1.0)
    assert close(calculate_pass_at_k(1, 0, 1), 0.0)

    print("✓ calculate_pass_at_k OK\n")


def verify_score_single_example():
    """验证 score_single_example"""
    print("Testing score_single_example...")

    # 例1：5 次采样，3 次正确（含数值类型）
    predicted = [42, "43", "42", "44", "42"]
    gold = 42
    scores = score_single_example(predicted, gold)
    assert len(scores) == 5
    assert abs(scores["pass@1"] - 0.6) < 1e-9
    assert scores["pass@5"] == 1.0

    # 例2：n=6, c=2
    predicted = ["ok", "good", "42", "x", "y", 42]
    gold = "42"
    scores = score_single_example(predicted, gold)
    assert abs(scores["pass@2"] - (1 - 6/15)) < 1e-9

    # 例3：n=4, c=0 -> 全 0
    predicted = ["a", "b", "c", "d"]
    gold = "42"
    scores = score_single_example(predicted, gold)
    assert all(abs(v - 0.0) < 1e-9 for v in scores.values())

    print("✓ score_single_example OK\n")


def verify_score_benchmark():
    """验证 score_benchmark（含构造数据）"""
    print("Testing score_benchmark...")

    test_dir = Path("output/test_model_scoring")
    test_dir.mkdir(parents=True, exist_ok=True)

    answer_file = test_dir / "test.jsonl"
    test_data = [
        {
            "question": "Q1",
            "answer": 42,                              # 数值型答案
            "predicted_answers": ["42", 42, "43"]
        },
        {
            "question": "Q2",
            "answer": "100",
            "predicted_answers": [100, "99", "100"]    # 混合类型
        },
        {
            "question": "Q3",
            "answer": "50%",
            "predicted_answers": ["0.5", "1/2", "0.49"]
        },
    ]
    with open(answer_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    result = score_benchmark(answer_file)
    assert result["total_examples"] == 3
    assert result["correct_at_least_once"] == 3
    assert "pass_at_k" in result and len(result["pass_at_k"]) > 0
    assert abs(result["pass_at_k"]["pass@1"] - (2/3)) < 1e-9

    print(f"  ✓ Result summary: {result}")
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("✓ score_benchmark OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Running verification tests for scoring.py")
    print("=" * 80 + "\n")

    verify_normalize_answer()
    verify_is_equiv()
    verify_calculate_pass_at_k()
    verify_score_single_example()
    verify_score_benchmark()

    print("=" * 80)
    print("All verification tests passed!")
    print("=" * 80 + "\n")
