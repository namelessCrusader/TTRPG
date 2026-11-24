"""Build cases.tex (the case-bank table) from docs/magic-science.md, so the PDF and the markdown cannot drift.
Run from the repo root: python3 docs/magic-science-tex/make_cases.py"""
import re, pathlib
root = pathlib.Path(__file__).resolve().parents[2]
md = (root / "docs/magic-science.md").read_text()
def esc(s):
    s = s.replace("\\", "/")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s
rows = []
for line in md.splitlines():
    m = re.match(r"\|\s*C(\d+)\s*\|(.*)\|(.*)\|(.*)\|\s*$", line)
    if m:
        rows.append((int(m.group(1)), [esc(x.strip()) for x in m.group(2, 3, 4)]))
rows.sort()
out = [r"\begin{longtable}{@{}p{0.06\textwidth}p{0.32\textwidth}p{0.40\textwidth}p{0.12\textwidth}@{}}",
       r"\toprule \textbf{Id} & \textbf{Case} & \textbf{A sensible outcome} & \textbf{Tests}\\ \midrule \endfirsthead",
       r"\toprule \textbf{Id} & \textbf{Case} & \textbf{A sensible outcome} & \textbf{Tests}\\ \midrule \endhead",
       r"\bottomrule \endfoot"]
for n, (case, outcome, tests) in rows:
    out.append(f"C{n:02d} & {case} & {outcome} & {tests}\\\\")
out.append(r"\end{longtable}")
(pathlib.Path(__file__).parent / "cases.tex").write_text("\n".join(out) + "\n")
print(len(rows), "cases")
