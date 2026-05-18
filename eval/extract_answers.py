"""
答案提取脚本 - 从模型输出中提取预测答案
"""
import json
import re
from pathlib import Path
from typing import List, Optional


def extract_latex_fraction(text: str) -> Optional[str]:
    """
    提取\frac{}{}格式的LaTeX分数表达式
    
    Args:
        text: 模型输出文本
    
    Returns:
        提取的分数表达式，如果没有找到返回None
    """
    # 匹配 \frac{分子}{分母} 格式，注意转义字符处理
    pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # 构建完整的分数表达式
        numerator, denominator = matches[-1]  # 取最后一个匹配
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    提取\\boxed{}中的答案
    
    Args:
        text: 模型输出文本
    
    Returns:
        提取的答案，如果没有找到返回None
    """
    # 匹配 \boxed{...} 格式
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # 返回最后一个boxed答案（通常是最终答案）
        return matches[-1].strip()
    
    return None


def _strip_trailing_punct(s: str) -> str:
    """
    去掉句尾常见标点，但保留数值中的有效字符（如 3.14 不受影响）。
    """
    if not s:
        return s
    s = s.strip()
    # 若末尾是中英文句号/叹号/分号/逗号，去掉之
    s = s.rstrip("。.!！；;，,")
    return s.strip()


def extract_answer_marker(text: str) -> Optional[str]:
    """
    提取Answer:或答案:后的内容
    
    Args:
        text: 模型输出文本
    
    Returns:
        提取的答案，如果没有找到返回None
    """
    # 说明：
    # 1) 为了支持小数（如 3.14），不再用 [^\n\.]+ 截断到句号；
    # 2) 改为抓取到行尾，然后统一去掉行尾的句号等标点；
    patterns = [
        r'Answer:\s*(.+)',                     # 英文 Answer:
        r'答案[:：]\s*(.+)',                    # 中文 答案：
        r'final answer is:?\s*(.+)',          # 英文 final answer is:
        r'最终答案[:：是]\s*(.+)',               # 中文 最终答案是：
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            # 仅取到该行结束
            candidate = candidate.splitlines()[0].strip()
            candidate = _strip_trailing_punct(candidate)
            if candidate:
                return candidate
    
    return None


def extract_last_number(text: str) -> Optional[str]:
    """
    提取文本中最后一个数字或数学表达式（分数 > 小数 > 整数）
    
    Args:
        text: 模型输出文本
    
    Returns:
        提取的数字/表达式，如果没有找到返回None
    """
    # 1) 分数（优先级最高）
    frac_matches = re.findall(r'-?\d+/\d+', text)
    if frac_matches:
        return frac_matches[-1].strip()
    
    # 2) 小数
    # 使用负向前后查找，避免把更长数字拆成多段
    dec_matches = re.findall(r'(?<![\d])(-?\d+\.\d+)(?![\d])', text)
    if dec_matches:
        return dec_matches[-1].strip()
    
    # 3) 整数（避免与小数的片段混淆：用边界限定）
    int_matches = re.findall(r'(?<![\d])(-?\d+)(?![\d\.])', text)
    if int_matches:
        return int_matches[-1].strip()
    
    return None


def extract_answer(text: str) -> str:
    """
    从模型输出中提取答案（综合多种策略）
    
    Args:
        text: 模型输出文本
    
    Returns:
        提取的答案，如果无法提取则返回原文本
    """
    # 策略1: 尝试提取\boxed{}
    answer = extract_boxed_answer(text)
    if answer:
        return answer
    
    # 策略2: 尝试提取LaTeX分数表达式
    answer = extract_latex_fraction(text)
    if answer:
        return answer
    
    # 策略3: 尝试提取Answer:标记
    answer = extract_answer_marker(text)
    if answer:
        return answer
    
    # 策略4: 提取最后一个数字（分数/小数/整数）
    answer = extract_last_number(text)
    if answer:
        return answer
    
    # 策略5: 返回最后一行非空内容
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    # 如果都失败，返回整个文本
    return text.strip()


def process_raw_outputs(input_file: Path, output_file: Path):
    """
    处理原始输出文件，提取答案
    
    Args:
        input_file: 原始输出文件路径 (xxx_raw.jsonl)
        output_file: 答案输出文件路径 (xxx.jsonl)
    """
    print(f"Processing: {input_file}")
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            item = json.loads(line.strip())
            
            question = item["question"]
            answer = item["answer"]
            raw_outputs = item["outputs"]
            
            # 从每个输出中提取答案
            extracted_answers = []
            for output in raw_outputs:
                extracted = extract_answer(output)
                extracted_answers.append(extracted)
            
            result = {
                "question": question,
                "answer": answer,
                "predicted_answers": extracted_answers,
            }
            results.append(result)
            
            if line_num % 50 == 0:
                print(f"  Processed {line_num} examples...")
    
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Extracted answers saved to: {output_file}")
    print(f"Total examples: {len(results)}\n")


def process_all_benchmarks(output_dir: Path):
    """
    处理指定输出目录下的所有benchmark
    
    Args:
        output_dir: 输出目录 (如 output/Qwen3-4B_2510251214/)
    """
    print(f"\n{'='*80}")
    print(f"Extracting answers from: {output_dir}")
    print(f"{'='*80}\n")
    
    benchmarks = ["aime24", "aime25", "math500", "olympiadbench"]
    processed_count = 0
    
    for benchmark in benchmarks:
        raw_file = output_dir / f"{benchmark}_raw.jsonl"
        answer_file = output_dir / f"{benchmark}.jsonl"
        
        if raw_file.exists():
            process_raw_outputs(raw_file, answer_file)
            processed_count += 1
        else:
            print(f"Warning: {raw_file} not found, skipping...\n")
    
    print(f"{'='*80}")
    print(f"Answer extraction completed! Processed {processed_count} benchmarks.")
    print(f"{'='*80}\n")


def verify_extract_boxed_answer():
    """验证extract_boxed_answer函数"""
    print("Testing extract_boxed_answer function...")
    
    test_cases = [
        ("The answer is \\boxed{42}", "42"),
        ("Step 1... Step 2... Therefore \\boxed{3.14}", "3.14"),
        ("\\boxed{1/2} is the first, \\boxed{3/4} is final", "3/4"),
        ("No boxed answer here", None),
    ]
    
    for text, expected in test_cases:
        result = extract_boxed_answer(text)
        assert result == expected, f"Failed: got {result}, expected {expected}"
        print(f"  ✓ '{text[:40]}...' -> {result}")
    
    print("✓ extract_boxed_answer function works correctly\n")


def verify_extract_answer_marker():
    """验证extract_answer_marker函数"""
    print("Testing extract_answer_marker function...")
    
    test_cases = [
        ("The calculation shows Answer: 42", "42"),
        ("答案：123", "123"),
        ("The final answer is: 3.14", "3.14"),
        ("最终答案是：2/3", "2/3"),
        ("No answer marker", None),
    ]
    
    for text, expected in test_cases:
        result = extract_answer_marker(text)
        if expected is None:
            assert result is None, f"Failed: got {result}, expected None"
        else:
            assert expected in str(result), f"Failed: {expected} not in {result}"
        print(f"  ✓ '{text[:40]}...' -> {result}")
    
    print("✓ extract_answer_marker function works correctly\n")


def verify_extract_last_number():
    """验证extract_last_number函数"""
    print("Testing extract_last_number function...")
    
    test_cases = [
        ("The answer is 42 and that's final", "42"),
        ("We get 3.14159 as result", "3.14159"),
        ("The fraction is 2/3", "2/3"),
        ("First 10, then 20, finally 30", "30"),
        ("No numbers here", None),
    ]
    
    for text, expected in test_cases:
        result = extract_last_number(text)
        assert result == expected, f"Failed: got {result}, expected {expected}"
        print(f"  ✓ '{text[:40]}...' -> {result}")
    
    print("✓ extract_last_number function works correctly\n")


def verify_extract_latex_fraction():
    """验证extract_latex_fraction函数"""
    print("Testing extract_latex_fraction function...")
    
    test_cases = [
        ("The answer is \\frac{14}{3}", "\\frac{14}{3}"),
        ("Step 1: \\frac{1}{2}, Step 2: \\frac{2}{3}", "\\frac{2}{3}"),
        ("No LaTeX fraction here", None),
        ("Partial: \\frac{14}", None),  # 不完整的表达式
    ]
    
    for text, expected in test_cases:
        result = extract_latex_fraction(text)
        assert result == expected, f"Failed: got {result}, expected {expected}"
        print(f"  ✓ '{text[:40]}...' -> {result}")
    
    print("✓ extract_latex_fraction function works correctly\n")


def verify_extract_answer():
    """验证extract_answer函数（综合测试）"""
    print("Testing extract_answer function (integrated)...")
    
    test_cases = [
        ("Therefore \\boxed{42}", "42"),
        ("The answer is \\frac{14}{3}", "\\frac{14}{3}"),
        ("Answer: 100", "100"),
        ("The result is 3.14", "3.14"),
        ("Just text\nFinal line", "Final line"),
    ]
    
    for text, expected in test_cases:
        result = extract_answer(text)
        assert expected in result, f"Failed: {expected} not in {result}"
        print(f"  ✓ '{text[:40]}...' -> {result}")
    
    print("✓ extract_answer function works correctly\n")


def verify_process_raw_outputs():
    """验证process_raw_outputs函数"""
    print("Testing process_raw_outputs function...")
    
    # 创建测试数据
    test_dir = Path("output/test_model")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    raw_file = test_dir / "test_raw.jsonl"
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "outputs": ["The answer is \\boxed{4}", "Answer: 4"]
        },
        {
            "question": "Calculate fraction",
            "answer": "\\frac{14}{3}",
            "outputs": ["The result is \\frac{14}{3}"]
        }
    ]
    
    with open(raw_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # 处理
    answer_file = test_dir / "test.jsonl"
    process_raw_outputs(raw_file, answer_file)
    
    # 验证输出
    with open(answer_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
        assert len(results) == 2
        assert results[0]["predicted_answers"][0] == "4"
        assert results[1]["predicted_answers"][0] == "\\frac{14}{3}"
    
    print("✓ process_raw_outputs function works correctly\n")
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running verification tests for extract_answers.py")
    print("="*80 + "\n")
    
    verify_extract_boxed_answer()
    verify_extract_latex_fraction()  # 新增的验证函数
    verify_extract_answer_marker()
    verify_extract_last_number()
    verify_extract_answer()
    verify_process_raw_outputs()
    
    print("="*80)
    print("All verification tests passed!")
    print("="*80 + "\n")