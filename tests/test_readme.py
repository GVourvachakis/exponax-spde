"""Run the documentation's Python examples in their original order."""

import re
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK = re.compile(
    r"^(?P<template><!-- non-executable template -->\n)?"
    r"```python[ \t]*\n(?P<code>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def test_documentation_examples():
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        namespace = {"__name__": "__main__"}
        document = path.read_text(encoding="utf-8")
        for block in PYTHON_BLOCK.finditer(document):
            if block["template"]:
                continue
            padding = "\n" * document.count("\n", 0, block.start("code"))
            exec(compile(padding + block["code"], str(path), "exec"), namespace)
        jax.block_until_ready(
            tuple(value for value in namespace.values() if isinstance(value, jax.Array))
        )
