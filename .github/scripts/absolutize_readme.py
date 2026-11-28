"""Pin README image and documentation URLs to the release commit."""

import argparse
import difflib
import posixpath
import re
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

REPO = "GVourvachakis/exponax-spde"
IMAGE_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png", ".svg", ".webp"}
DOC_SUFFIXES = {".md", ".txt", ".rst", ".ipynb"}
MARKDOWN = re.compile(r"(!?\[[^\]\n]*\]\()(?P<url><?[^)\s>]+>?)")
REFERENCE = re.compile(r"(^[ \t]*\[[^\]\n]*\]:[ \t]*)(?P<url><?[^\s>]+>?)")
HTML_ATTR = re.compile(r"""(\b(?:src|href)\s*=\s*["'])(?P<url>[^"']+)(?=["'])""")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def rewrite(markdown: str, sha: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("Supply a full 40-digit Git commit SHA.")

    def absolute(match):
        value = match["url"]
        wrapped = value.startswith("<") and value.endswith(">")
        target = value[1:-1] if wrapped else value
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or parts.path.startswith("/"):
            return match[0]
        path = posixpath.normpath(parts.path)
        if path in {".", ".."} or path.startswith("../"):
            return match[0]
        ext = Path(path).suffix.lower()
        safe_path = quote(path, safe="/%")
        if ext in IMAGE_SUFFIXES:
            host, resolved = "raw.githubusercontent.com", f"/{REPO}/{sha}/{safe_path}"
        elif ext in DOC_SUFFIXES or path.startswith(("docs/", "validation/")):
            host, resolved = "github.com", f"/{REPO}/blob/{sha}/{safe_path}"
        else:
            return match[0]
        url = urlunsplit(("https", host, resolved, parts.query, parts.fragment))
        return match[1] + (f"<{url}>" if wrapped else url)

    result = []
    fence = None
    for line in markdown.splitlines(keepends=True):
        found = FENCE.match(line)
        if found:
            token = found[1]
            if fence is None:
                fence = token
            elif (
                token[0] == fence[0]
                and len(token) >= len(fence)
                and not line[found.end() :].strip()
            ):
                fence = None
            result.append(line)
            continue
        if fence is None:
            segments = re.split(r"(`+[^`\n]*`+)", line)
            for index in range(0, len(segments), 2):
                for pattern in (MARKDOWN, REFERENCE, HTML_ATTR):
                    segments[index] = pattern.sub(absolute, segments[index])
            line = "".join(segments)
        result.append(line)
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sha", help="Full commit SHA for the README targets")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--write", action="store_true", help="Apply the rewrite")
    args = parser.parse_args()
    old = args.readme.read_text(encoding="utf-8")
    try:
        new = rewrite(old, args.sha.lower())
    except ValueError as exc:
        parser.error(str(exc))
    if args.write:
        args.readme.write_text(new, encoding="utf-8")
    else:
        sys.stdout.writelines(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=str(args.readme),
                tofile=f"{args.readme} (release links)",
            )
        )


if __name__ == "__main__":
    main()
