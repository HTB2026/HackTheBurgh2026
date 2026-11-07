"""A self-contained, printable HTML case report."""

import html
import time
from typing import Any

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "notice": 2, "info": 3}
_SEVERITY_COLOUR = {
    "critical": "#ff3b5c",
    "warning": "#ffb020",
    "notice": "#38d6ff",
    "info": "#8aa0b4",
}


def _fmt_time(value: Any) -> str:
    if value is None:
        return ""

    seconds = float(value)

    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def render(case: Any) -> str:
    events = case.events
    meta = next((e["case"] for e in events if e["type"] == "case"), {}) or {}
    findings = [e["finding"] for e in events if e["type"] == "finding"]
    findings.sort(
        key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 9), f.get("time_s") or 0)
    )
    brief = next(
        (
            e["artifact"]["data"]
            for e in events
            if e["type"] == "artifact" and e["artifact"]["id"] == "brief:brief"
        ),
        None,
    )
    images = [
        e["artifact"]
        for e in events
        if e["type"] == "artifact"
        and e["artifact"]["kind"] in ("image", "spectrogram", "heatmap")
        and e["artifact"].get("url")
    ]
    rows = "\n".join(
        f"""<tr><td><span class="sev" style="background:{_SEVERITY_COLOUR.get(f["severity"], "#888")}">{f["severity"].upper()}</span></td>
<td>{html.escape(f["module"])}</td><td>{_fmt_time(f.get("time_s"))}</td>
<td><b>{html.escape(f["title"])}</b><br><span class="d">{html.escape(f["detail"])}</span>
{'<br><i class="inf">AI-inferred</i>' if f.get("kind") == "inferred" else ""}</td><td>{f["id"]}</td></tr>"""
        for f in findings
    )
    figures = "\n".join(
        f"<figure><img src='{a['url']}'><figcaption>{html.escape(a['title'])}</figcaption></figure>"
        for a in images[:12]
    )
    brief_html = ""

    if brief:
        brief_html = f"<section><h2>Analyst brief</h2><p>{html.escape(brief.get('text', '')).replace(chr(10), '<br>')}</p></section>"

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Case {case.id}</title>
<style>
body{{font:14px/1.5 'IBM Plex Mono',ui-monospace,monospace;background:#fff;color:#111;max-width:1100px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:20px;letter-spacing:.08em}} h2{{font-size:15px;border-bottom:1px solid #ccc;padding-bottom:4px;margin-top:2rem}}
table{{border-collapse:collapse;width:100%}} td{{border-bottom:1px solid #eee;padding:6px;vertical-align:top}}
.sev{{color:#000;padding:1px 6px;font-size:11px;font-weight:bold}} .d{{color:#444}} .inf{{color:#7a4fd0}}
figure{{display:inline-block;width:48%;margin:0 1% 1rem 0}} figure img{{width:100%;border:1px solid #ccc}}
figcaption{{font-size:12px;color:#555}} .meta td:first-child{{color:#666;width:180px}}
</style></head><body>
<h1>EARWITNESS // CASE REPORT {html.escape(case.id)}</h1>
<table class="meta">
<tr><td>Evidence</td><td>{html.escape(case.filename)}</td></tr>
<tr><td>Duration</td><td>{meta.get("duration_s", 0):.2f} s</td></tr>
<tr><td>Video</td><td>{html.escape(str(meta.get("video")))}</td></tr>
<tr><td>Generated</td><td>{time.strftime("%Y-%m-%d %H:%M:%S %Z")}</td></tr>
<tr><td>Findings</td><td>{len(findings)}</td></tr>
</table>
{brief_html}
<section><h2>Findings</h2><table>{rows}</table></section>
<section><h2>Figures</h2>{figures}</section>
<p style="color:#888;font-size:12px;margin-top:3rem">Measured findings come from signal analysis of the file.
Findings marked AI-inferred come from a language/vision model and should be independently verified.</p>
</body></html>"""
