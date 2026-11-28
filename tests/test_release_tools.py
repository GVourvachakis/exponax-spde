"""Check the link rewrite used for the published package description."""

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rewrite = runpy.run_path(str(ROOT / ".github/scripts/absolutize_readme.py"))["rewrite"]
REF = "abc0123456" * 4


def test_release_rewrites_relative_image_and_doc_links():
    input_markdown = (
        "![Preview](docs/imgs/step.gif)\n"
        '<img src="docs/imgs/logo.svg">\n'
        "[Guide](./docs/noise_model.md#additive-migration)\n"
        "[Notebook](validation/example.ipynb)\n"
    )
    raw = f"https://raw.githubusercontent.com/GVourvachakis/exponax-spde/{REF}"
    blob = f"https://github.com/GVourvachakis/exponax-spde/blob/{REF}"
    assert rewrite(input_markdown, REF) == (
        f"![Preview]({raw}/docs/imgs/step.gif)\n"
        f'<img src="{raw}/docs/imgs/logo.svg">\n'
        f"[Guide]({blob}/docs/noise_model.md#additive-migration)\n"
        f"[Notebook]({blob}/validation/example.ipynb)\n"
    )


def test_release_preserves_absolute_urls_anchors_and_code():
    input_markdown = (
        "[Project](https://example.org/docs/location.md)\n"
        "[Quick start](#quickstart)\n"
        '<a href="#quickstart">Quick start</a>\n'
        "```markdown\n[Example](docs/example.md)\n```\n"
    )
    assert rewrite(input_markdown, REF) == input_markdown


def test_release_rewrite_is_idempotent():
    once = rewrite("[Guide](docs/noise_model.md)", REF)
    assert rewrite(once, REF) == once
