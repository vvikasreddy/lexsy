import argparse
import sys
from typing import List

try:
    from blank_filler import find_blanks_in_docx, DOCX_ENABLED
except Exception as e:  # pragma: no cover
    print(f"Failed to import blank_filler: {e}")
    sys.exit(1)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Print blanks with 20-word context before/after from a .docx file.",
    )
    parser.add_argument("docx", help="Path to the .docx file")
    args = parser.parse_args(argv)

    if not DOCX_ENABLED:
        print("python-docx is not installed. Install with: pip install python-docx")
        return 2

    path = "./doc.docx"
    matches = find_blanks_in_docx(path, ctx_words=20)
    if not matches:
        print("No blanks found or file unreadable.")
        return 0

    for i, b in enumerate(matches, 1):
        before = " ".join(b.before_words)
        after = " ".join(b.after_words)
        print(f"#{i}")
        print(f" label: {b.label}")
        print(f" kind: {b.kind}")
        if b.kind == "underscore":
            print(f" underscore_len: {b.underscore_len}")
        print(f" position: paragraph={b.paragraph_index}, span=({b.start},{b.end})")
        print(f" before(20): {before}")
        print(f" blank: {b.text}")
        print(f" after(20): {after}")
        print("-")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

