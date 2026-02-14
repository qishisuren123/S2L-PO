
import json
import re
from pathlib import Path
from typing import List, Optional


def extract_latex_fraction(text: str) -> Optional[str]:
    pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        numerator, denominator = matches[-1]  
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        return matches[-1].strip()
    
    return None


def _strip_trailing_punct(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = s.rstrip("。.!！；;，,")
    return s.strip()


def extract_answer_marker(text: str) -> Optional[str]:
    patterns = [
        r'Answer:\s*(.+)',                   
        r'答案[:：]\s*(.+)',                   
        r'final answer is:?\s*(.+)',          
        r'最终答案[:：是]\s*(.+)',             
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            candidate = candidate.splitlines()[0].strip()
            candidate = _strip_trailing_punct(candidate)
            if candidate:
                return candidate
    
    return None


def extract_last_number(text: str) -> Optional[str]:
    frac_matches = re.findall(r'-?\d+/\d+', text)
    if frac_matches:
        return frac_matches[-1].strip()
    
    dec_matches = re.findall(r'(?<![\d])(-?\d+\.\d+)(?![\d])', text)
    if dec_matches:
        return dec_matches[-1].strip()
    
    int_matches = re.findall(r'(?<![\d])(-?\d+)(?![\d\.])', text)
    if int_matches:
        return int_matches[-1].strip()
    
    return None


def extract_answer(text: str) -> str:
    answer = extract_boxed_answer(text)
    if answer:
        return answer
    
    answer = extract_latex_fraction(text)
    if answer:
        return answer
    
    answer = extract_answer_marker(text)
    if answer:
        return answer
    
    answer = extract_last_number(text)
    if answer:
        return answer

    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if lines:
        return lines[-1]
    
    return text.strip()


def process_raw_outputs(input_file: Path, output_file: Path):
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
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Extracted answers saved to: {output_file}")
    print(f"Total examples: {len(results)}\n")


def process_all_benchmarks(output_dir: Path):
    print(f"\n{'='*80}")
    print(f"Extracting answers from: {output_dir}")
    print(f"{'='*80}\n")
    
    benchmarks = ["aime24", "aime25", "math500"]
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
    print("Testing process_raw_outputs function...")
    
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
    
    answer_file = test_dir / "test.jsonl"
    process_raw_outputs(raw_file, answer_file)

    with open(answer_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f]
        assert len(results) == 2
        assert results[0]["predicted_answers"][0] == "4"
        assert results[1]["predicted_answers"][0] == "\\frac{14}{3}"
    
    print("✓ process_raw_outputs function works correctly\n")

    import shutil
    shutil.rmtree(test_dir)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Running verification tests for extract_answers.py")
    print("="*80 + "\n")
    
    verify_extract_boxed_answer()
    verify_extract_latex_fraction() 
    verify_extract_answer_marker()
    verify_extract_last_number()
    verify_extract_answer()
    verify_process_raw_outputs()
    
    print("="*80)
    print("All verification tests passed!")
    print("="*80 + "\n")