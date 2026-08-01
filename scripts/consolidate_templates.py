#!/usr/bin/env python3
"""One-time migration: standardize bootstrap imports and parse_common_config usage."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STANDARD_BOOTSTRAP = """import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""

BOOTSTRAP_PATTERNS = [
    re.compile(
        r"import sys\nfrom pathlib import Path\n\n# Add src to path\n"
        r"sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\)\)\n+"
    ),
    re.compile(
        r"from pathlib import Path\n\nimport sys\n\n# Add src to path\n"
        r"sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\)\)\n+"
    ),
]

PARSE_CONFIG_PATTERN = re.compile(
    r"(def parse_config\(config_dict: dict, script_dir: Path\) -> Config:\n"
    r'    """Parse config dictionary into Config dataclass\."""\n)'
    r"    repo_root = script_dir\.parent\n"
    r'    data_path = repo_root / "data" / config_dict\["data"\]\["input_file"\]\n'
    r"    output_dir = ensure_output_dir\(Path\(script_dir\) / (?:config_dict\[\"output\"\]\[\"output_dir\"\]|\"outputs\")\)\n+",
    re.MULTILINE,
)

PARSE_REPLACEMENT = (
    r"\1"
    r"    common = parse_common_config(config_dict, script_dir)\n\n"
)


def standardize_bootstrap(text: str) -> str:
    for pattern in BOOTSTRAP_PATTERNS:
        if pattern.search(text):
            return pattern.sub(STANDARD_BOOTSTRAP, text, count=1)
    return text


def add_parse_common_import(text: str) -> str:
    if "parse_common_config" in text:
        return text
    if "def parse_config" not in text:
        return text

    if "from src.config import parse_common_config" in text:
        return text

    # Add import after src import block
    marker = "from src import ("
    if marker in text:
        # Multi-line from src import — add separate import after closing paren
        end = text.find(")", text.find(marker))
        if end != -1:
            insert_at = text.find("\n", end) + 1
            return (
                text[:insert_at]
                + "from src.config import parse_common_config\n"
                + text[insert_at:]
            )

    marker2 = "from src import"
    if marker2 in text:
        line_end = text.find("\n", text.find(marker2))
        return (
            text[: line_end + 1]
            + "from src.config import parse_common_config\n"
            + text[line_end + 1 :]
        )

    return text


def update_parse_config(text: str) -> str:
    if "def parse_config" not in text:
        return text

    text = PARSE_CONFIG_PATTERN.sub(PARSE_REPLACEMENT, text)

    # Replace common field references
    replacements = [
        ("data_path=data_path,", "data_path=common.data_path,"),
        ("date_col=config_dict[\"data\"][\"date_col\"],", "date_col=common.date_col,"),
        ("value_col=config_dict[\"data\"][\"value_col\"],", "value_col=common.value_col,"),
        ("output_dir=output_dir,", "output_dir=common.output_dir,"),
    ]
    for old, new in replacements:
        if "def parse_config" in text:
            text = text.replace(old, new)

    return text


def process_main_py(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    updated = standardize_bootstrap(updated)
    updated = add_parse_common_import(updated)
    updated = update_parse_config(updated)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for main_py in sorted(REPO_ROOT.glob("*_Python/main.py")):
        if process_main_py(main_py):
            changed.append(main_py.relative_to(REPO_ROOT))

    print(f"Updated {len(changed)} templates:")
    for path in changed:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
