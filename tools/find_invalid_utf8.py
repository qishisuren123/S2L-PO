import sys

def _is_cont(b: int) -> bool:
    return 0x80 <= b <= 0xBF

def _hex(bs: bytes) -> str:
    return " ".join(f"{x:02x}" for x in bs)

def scan_invalid_utf8(b: bytes):
    i = 0
    issues = []
    while i < len(b):
        x = b[i]
        if x <= 0x7F:
            i += 1
            continue
        if 0xC2 <= x <= 0xDF:
            if i + 1 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为2字节)"))
                i += 1
            else:
                b1 = b[i+1]
                if not _is_cont(b1):
                    issues.append((i, b[i:i+2], "续字节非法(2字节)"))
                    i += 1
                else:
                    i += 2
            continue
        if x == 0xE0:
            if i + 2 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为3字节)"))
                i += 1
            else:
                b1, b2 = b[i+1], b[i+2]
                if not (0xA0 <= b1 <= 0xBF and _is_cont(b2)):
                    issues.append((i, b[i:i+3], "续字节/范围非法(E0)"))
                    i += 1
                else:
                    i += 3
            continue
        if 0xE1 <= x <= 0xEC or 0xEE <= x <= 0xEF:
            if i + 2 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为3字节)"))
                i += 1
            else:
                b1, b2 = b[i+1], b[i+2]
                if not (_is_cont(b1) and _is_cont(b2)):
                    issues.append((i, b[i:i+3], "续字节非法(3字节)"))
                    i += 1
                else:
                    i += 3
            continue
        if x == 0xED:
            if i + 2 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为3字节)"))
                i += 1
            else:
                b1, b2 = b[i+1], b[i+2]
                if not (0x80 <= b1 <= 0x9F and _is_cont(b2)):
                    issues.append((i, b[i:i+3], "代理项范围非法(ED)"))
                    i += 1
                else:
                    i += 3
            continue
        if x == 0xF0:
            if i + 3 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为4字节)"))
                i += 1
            else:
                b1, b2, b3 = b[i+1], b[i+2], b[i+3]
                if not (0x90 <= b1 <= 0xBF and _is_cont(b2) and _is_cont(b3)):
                    issues.append((i, b[i:i+4], "续字节/范围非法(F0)"))
                    i += 1
                else:
                    i += 4
            continue
        if 0xF1 <= x <= 0xF3:
            if i + 3 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为4字节)"))
                i += 1
            else:
                b1, b2, b3 = b[i+1], b[i+2], b[i+3]
                if not (_is_cont(b1) and _is_cont(b2) and _is_cont(b3)):
                    issues.append((i, b[i:i+4], "续字节非法(4字节)"))
                    i += 1
                else:
                    i += 4
            continue
        if x == 0xF4:
            if i + 3 >= len(b):
                issues.append((i, b[i:i+1], "序列被截断(应为4字节)"))
                i += 1
            else:
                b1, b2, b3 = b[i+1], b[i+2], b[i+3]
                if not (0x80 <= b1 <= 0x8F and _is_cont(b2) and _is_cont(b3)):
                    issues.append((i, b[i:i+4], "续字节/范围非法(F4)"))
                    i += 1
                else:
                    i += 4
            continue
        # 其它情况：无效引导字节，如 0x80..0xC1 或 0xF5..0xFF
        issues.append((i, b[i:i+1], "无效引导字节"))
        i += 1
    return issues

def main():
    if len(sys.argv) < 2:
        print("用法: python find_invalid_utf8.py <jsonl文件> [--check-json]")
        sys.exit(1)
    path = sys.argv[1]
    check_json = len(sys.argv) >= 3 and sys.argv[2] == "--check-json"

    total_invalid_lines = 0
    total_issues = 0

    with open(path, "rb") as fin:
        for line_no, raw in enumerate(fin, start=1):
            issues = scan_invalid_utf8(raw)
            if issues:
                total_invalid_lines += 1
                for pos, seq, reason in issues:
                    total_issues += 1
                    ctx_start = max(0, pos - 8)
                    ctx_end = min(len(raw), pos + len(seq) + 8)
                    hex_ctx = _hex(raw[ctx_start:ctx_end])
                    text_ctx = raw[ctx_start:ctx_end].decode("utf-8", errors="replace")
                    print(f"[invalid] line={line_no} byte_offset={pos} bytes={_hex(seq)} reason={reason}")
                    print(f"          hex_ctx: {hex_ctx}")
                    print(f"          text_ctx: {text_ctx}")
            elif check_json:
                try:
                    s = raw.decode("utf-8")
                except UnicodeDecodeError as e:
                    print(f"[decode-error] line={line_no}: {e}")
                    continue
                import json
                try:
                    json.loads(s)
                except Exception as e:
                    snippet = s.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    print(f"[json-error] line={line_no}: {e}")
                    print(f"             snippet: {snippet}")

    print(f"Done. invalid_lines={total_invalid_lines} issues={total_issues}")

if __name__ == "__main__":
    main()