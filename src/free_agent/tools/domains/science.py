"""Scientific-research tools — literature search, citation lookup, computation, statistics."""
from __future__ import annotations

import math
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from langchain_core.tools import tool

from free_agent.tools.domains._common import (
    http_get,
    http_json,
    net_error,
    parse_numbers,
    truncate,
)

_ATOM = "{http://www.w3.org/2005/Atom}"


@tool
def arxiv_search(query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
    """Search arXiv preprints. Returns title, authors, date, arXiv id, PDF link and abstract.

    Args:
      query: arXiv query, e.g. `transformer protein folding` or `au:hinton AND ti:capsule`.
      max_results: Number of papers (1-25, default 5).
      sort_by: `relevance`, `submittedDate` or `lastUpdatedDate`.
    """
    if sort_by not in ("relevance", "submittedDate", "lastUpdatedDate"):
        sort_by = "relevance"
    params = {
        "search_query": query if ":" in query else " AND ".join(f"all:{w}" for w in query.split()),
        "max_results": max(1, min(int(max_results), 25)),
        "sortBy": sort_by,
    }
    try:
        root = ET.fromstring(http_get("https://export.arxiv.org/api/query", params))
    except Exception as exc:
        return net_error(exc)
    entries = root.findall(f"{_ATOM}entry")
    if not entries:
        return f"[no arXiv results for {query!r}]"
    out = []
    for i, e in enumerate(entries, 1):
        title = " ".join((e.findtext(f"{_ATOM}title") or "").split())
        authors = [a.findtext(f"{_ATOM}name") or "" for a in e.findall(f"{_ATOM}author")]
        published = (e.findtext(f"{_ATOM}published") or "")[:10]
        abs_url = e.findtext(f"{_ATOM}id") or ""
        summary = " ".join((e.findtext(f"{_ATOM}summary") or "").split())
        author_str = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
        out.append(
            f"{i}. {title}\n   {author_str} · {published}\n   {abs_url} · "
            f"pdf: {abs_url.replace('/abs/', '/pdf/')}\n   {summary[:600]}"
        )
    return truncate("\n\n".join(out))


@tool
def pubmed_search(query: str, max_results: int = 5) -> str:
    """Search PubMed (biomedical literature). Returns title, journal, date, authors, PMID and DOI.

    Args:
      query: PubMed query, supports field tags like `crispr[Title] AND 2023[dp]`.
      max_results: Number of articles (1-25, default 5).
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        ids = http_json(
            f"{base}/esearch.fcgi",
            {"db": "pubmed", "term": query, "retmode": "json", "sort": "relevance",
             "retmax": max(1, min(int(max_results), 25))},
        )["esearchresult"]["idlist"]
        if not ids:
            return f"[no PubMed results for {query!r}]"
        summ = http_json(f"{base}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    except Exception as exc:
        return net_error(exc)
    out = []
    for i, pmid in enumerate(ids, 1):
        doc = summ.get("result", {}).get(pmid, {})
        authors = [a.get("name", "") for a in doc.get("authors", [])]
        doi = next((a["value"] for a in doc.get("articleids", []) if a.get("idtype") == "doi"), "")
        out.append(
            f"{i}. {doc.get('title', '(untitled)')}\n   {doc.get('fulljournalname', '')} · "
            f"{doc.get('pubdate', '')}\n   {', '.join(authors[:5])}{' et al.' if len(authors) > 5 else ''}\n"
            f"   PMID {pmid} · https://pubmed.ncbi.nlm.nih.gov/{pmid}/" + (f" · doi:{doi}" if doi else "")
        )
    return truncate("\n\n".join(out))


@tool
def crossref_lookup(doi_or_query: str, max_results: int = 3) -> str:
    """Resolve a DOI (or search Crossref by bibliographic text) and return citation metadata.

    Use it to verify references and build citations — never invent DOIs.

    Args:
      doi_or_query: A DOI like `10.1038/nature14539`, or free text (title, authors, year).
      max_results: Results for free-text searches (1-10, default 3).
    """
    text = doi_or_query.strip().removeprefix("https://doi.org/").removeprefix("doi:")
    try:
        if text.startswith("10."):
            items = [http_json(f"https://api.crossref.org/works/{text}")["message"]]
        else:
            items = http_json(
                "https://api.crossref.org/works",
                {"query.bibliographic": text, "rows": max(1, min(int(max_results), 10))},
            )["message"]["items"]
    except Exception as exc:
        return net_error(exc)
    if not items:
        return f"[no Crossref match for {doi_or_query!r}]"
    out = []
    for it in items:
        authors = [f"{a.get('family', '')}, {a.get('given', '')[:1]}." for a in it.get("author", [])]
        year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        out.append(
            f"{' '.join(it.get('title') or ['(untitled)'])}\n"
            f"   {'; '.join(authors[:6])}{' et al.' if len(authors) > 6 else ''} ({year})\n"
            f"   {' '.join(it.get('container-title') or [])} · {it.get('type', '')} · "
            f"cited by {it.get('is-referenced-by-count', '?')}\n   https://doi.org/{it.get('DOI', '')}"
        )
    return truncate("\n\n".join(out))


@tool
def python_compute(code: str, timeout: int = 120) -> str:
    """Run a Python script in a fresh subprocess and return stdout/stderr.

    Use it for numerical work, simulations, data analysis and plotting
    (numpy/scipy/pandas/matplotlib if installed). `print` what you need —
    only printed output comes back. Save figures with `plt.savefig(...)`.

    Args:
      code: Complete Python source to execute.
      timeout: Seconds before the run is killed (max 600).
    """
    timeout = max(1, min(int(timeout), 600))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        script = fh.name
    try:
        res = subprocess.run(
            [sys.executable, script], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    parts = [f"exit={res.returncode}"]
    if res.stdout:
        parts.append(f"stdout:\n{res.stdout}")
    if res.stderr:
        parts.append(f"stderr:\n{res.stderr}")
    return truncate("\n".join(parts))


def _t_crit95(df: int) -> float:
    # Two-sided 95% Student-t critical values; normal approximation beyond 30.
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
             9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
    if df in table:
        return table[df]
    if df > 30:
        return 1.96
    return table[max(k for k in table if k <= df)]


@tool
def stats_summary(values: str) -> str:
    """Descriptive statistics for a numeric sample: n, mean, sd, sem, 95% CI, median, quartiles, skewness.

    Args:
      values: Numbers separated by commas, spaces, semicolons or newlines.
    """
    try:
        xs = parse_numbers(values)
    except ValueError as exc:
        return f"[{exc}]"
    n = len(xs)
    if n < 2:
        return "[need at least 2 values]"
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    sem = sd / math.sqrt(n)
    h = _t_crit95(n - 1) * sem
    q1, q2, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    skew = (sum((x - mean) ** 3 for x in xs) / n) / (statistics.pstdev(xs) ** 3) if sd else 0.0
    cv = f"{sd / mean:.4g}" if mean else "n/a"
    return (
        f"n={n}\nmean={mean:.6g}\nsd={sd:.6g}\nsem={sem:.6g}\n95% CI=[{mean - h:.6g}, {mean + h:.6g}]\n"
        f"min={min(xs):.6g}  Q1={q1:.6g}  median={q2:.6g}  Q3={q3:.6g}  max={max(xs):.6g}\n"
        f"IQR={q3 - q1:.6g}\nskewness={skew:.4g}\ncv={cv}"
    )


@tool
def linear_fit(x_values: str, y_values: str) -> str:
    """Ordinary least-squares fit y = a + b·x with r, R², standard errors and residual sd.

    Args:
      x_values: Independent variable, numbers separated by commas/spaces.
      y_values: Dependent variable, same length as x_values.
    """
    try:
        xs, ys = parse_numbers(x_values), parse_numbers(y_values)
    except ValueError as exc:
        return f"[{exc}]"
    n = len(xs)
    if n != len(ys) or n < 3:
        return "[x and y must have the same length (≥ 3)]"
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return "[x has zero variance]"
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    s = math.sqrt(sse / (n - 2))
    r = sxy / math.sqrt(sxx * syy) if syy else 0.0
    se_b = s / math.sqrt(sxx)
    se_a = s * math.sqrt(1 / n + mx**2 / sxx)
    t_b = b / se_b if se_b else float("inf")
    return (
        f"y = {a:.6g} + {b:.6g}·x\nslope={b:.6g} ± {se_b:.4g} (t={t_b:.4g}, df={n - 2})\n"
        f"intercept={a:.6g} ± {se_a:.4g}\nr={r:.5f}  R²={r * r:.5f}\nresidual sd={s:.6g}  n={n}"
    )


SCIENCE_TOOLS = [arxiv_search, pubmed_search, crossref_lookup, python_compute, stats_summary, linear_fit]
