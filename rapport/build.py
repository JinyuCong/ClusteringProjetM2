"""Compile the report and create a self-contained Overleaf source archive.

Usage from the repository root: python rapport/build.py
Requires pdfLaTeX on PATH. No scientific Python packages are needed to compile.
"""
from pathlib import Path
import re
import shutil
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    compiler = shutil.which("pdflatex")
    if not compiler:
        raise SystemExit("pdfLaTeX is required; alternatively upload overleaf.zip to Overleaf.")
    for number in range(1, 4):
        result = subprocess.run(
            [compiler, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "-output-directory=build", "-jobname=rapport", "main.tex"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        (build / f"pass-{number}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"LaTeX failed: see {build / f'pass-{number}.txt'}")
    log = (build / "rapport.log").read_text(encoding="utf-8", errors="replace")
    if re.search(r"undefined references|(?:Citation|Reference) .+ undefined|Overfull", log):
        raise SystemExit("Unresolved references or overflowing content: inspect build/rapport.log.")
    shutil.copyfile(build / "rapport.pdf", root / "rapport.pdf")
    sources = list(root.glob("*.tex"))
    for folder in ("sections", "tables"):
        sources.extend((root / folder).glob("*.tex"))
    sources.extend((root / "figures").glob("*.pdf"))
    sources.extend((root / "assets").glob("*.pdf"))
    sources.extend((root / "assets").glob("*.md"))
    with zipfile.ZipFile(root / "overleaf.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):
            archive.write(source, source.relative_to(root).as_posix())
    print(f"PDF: {root / 'rapport.pdf'}")
    print(f"Overleaf sources: {root / 'overleaf.zip'}")


if __name__ == "__main__":
    main()
