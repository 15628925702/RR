# -*- coding: utf-8 -*-
"""Export the Chinese progress report markdown to a clean A4 PDF via Chrome."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent
HTML_OUT = ROOT / "report.html"
PDF_TMP = ROOT / "report.pdf"
CSS_PATH = ROOT / "style.css"


def resolve_paths() -> tuple[Path, list[Path]]:
    matches = sorted(DOCS.glob("*P4阻塞分析.md"))
    if not matches:
        matches = sorted(DOCS.glob("*P4*.md"))
    if not matches:
        sys.exit(f"no source markdown under {DOCS}")
    src = matches[0]
    pdf_names = [
        src.with_suffix(".pdf"),
        DOCS / "汇报给师兄_项目进展与P4阻塞分析.pdf",
    ]
    # de-duplicate while keeping order
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for p in pdf_names:
        if p not in seen:
            seen.add(p)
            pdfs.append(p)
    return src, pdfs

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def autolink(html: str) -> str:
    return re.sub(
        r'(?<!href=")(https?://[^\s<]+)',
        r'<a href="\1">\1</a>',
        html,
    )


def mark_headings(html: str) -> str:
    html = html.replace("<h3>✅ ", '<h3><span class="dot ok"></span>')
    html = html.replace("<h3>🔶 ", '<h3><span class="dot warn"></span>')
    html = html.replace("<h3>❌ ", '<h3><span class="dot bad"></span>')
    return html


def wrap_callouts(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return f'<div class="callout">{match.group(0)}</div>'

    for pattern in (
        r"<p><strong>核心矛盾</strong>：.*?</p>",
        r"<p><strong>诚实结论</strong>：.*?</p>",
        r"<p><strong>关键推论</strong>：.*?</p>",
    ):
        html = re.sub(pattern, repl, html, count=1, flags=re.S)
    return html


def wrap_last_paragraph(html: str) -> str:
    return re.sub(
        r"(<p>代码、结果、诊断与交接文档[\s\S]*?</p>)\s*</body>",
        r'<div class="footer-note">\1</div></body>',
        html,
        count=1,
    )


def build_html(md_text: str) -> str:
    lines = md_text.splitlines()
    title_line = lines[0].lstrip("# ").strip()
    body_md = "\n".join(lines[1:]).lstrip("\n")
    if body_md.startswith("---"):
        body_md = body_md[3:].lstrip("\n")

    if "：" in title_line:
        main, sub = title_line.split("：", 1)
    else:
        main, sub = title_line, ""

    body_html = markdown.markdown(
        body_md,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body_html = autolink(body_html)
    body_html = re.sub(
        r"<code>(https?://[^<]+)</code>",
        r'<a href="\1">\1</a>',
        body_html,
    )
    body_html = mark_headings(body_html)
    body_html = wrap_callouts(body_html)

    css = CSS_PATH.read_text(encoding="utf-8")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{main}</title>
<style>
{css}
</style>
</head>
<body>
<header class="masthead">
  <p class="kicker">RR-GID · 阶段性内部汇报</p>
  <h1>{main}</h1>
  <p class="subtitle">{sub}</p>
  <p class="meta">2026年8月26日<span class="sep">|</span>面向师兄的进展与阻塞说明</p>
</header>
<main>
{body_html}
</main>
</body>
</html>
"""
    return wrap_last_paragraph(html)


def export_pdf() -> None:
    src, pdfs = resolve_paths()
    if not CHROME.exists():
        sys.exit(f"missing Chrome: {CHROME}")

    HTML_OUT.write_text(build_html(src.read_text(encoding="utf-8")), encoding="utf-8")
    if PDF_TMP.exists():
        PDF_TMP.unlink()

    cmd = [
        str(CHROME),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--disable-extensions",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=12000",
        f"--print-to-pdf={PDF_TMP}",
        HTML_OUT.resolve().as_uri(),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not PDF_TMP.exists():
        sys.stderr.write((proc.stdout or "") + "\n" + (proc.stderr or ""))
        sys.exit(proc.returncode or 1)

    size_kb = PDF_TMP.stat().st_size / 1024
    for pdf_out in pdfs:
        shutil.copy2(PDF_TMP, pdf_out)
        print(f"wrote {pdf_out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    export_pdf()
