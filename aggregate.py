"""Aggregate per-host benchmark results into a comparison table + SVG chart.

Reads results/*.yaml (one per server, produced by run.sh) and emits:
  - a console table
  - assets/benchmarks.svg   (roomy horizontal bar chart, per-metric; no deps)
  - assets/benchmarks.html  (raw-number table, ready to paste into the Quarto page)

No third-party dependencies beyond PyYAML, so it runs anywhere (laptop included).
"""
import os
import glob
import html
import yaml

RESULTS = "results"
ASSETS = "assets"

# (display label, yaml section, metric key, higher-is-better)
# Grouped: raw compute, then training, then inference, then storage — so a viewer can see
# *why* numbers differ (e.g. a big GPU dominates compute/training even if small-batch
# inference looks flat).
METRICS = [
    ("Raw compute (TFLOPS fp16)", "Compute", "Matmul_FP16_TFLOPS", True),
    ("Training: ResNet50 (img/s)", "Training", "Images_per_Sec", True),
    ("Inference: ResNet50 (img/s)", "Classification", "Images_per_Sec", True),
    ("Inference: detection (FPS)", "Detection", "Steady_FPS", True),
    ("Inference: LLM (tok/s)", "LLM", "Steady_Tokens_per_Sec", True),
    ("Data loading (img/s)", "Storage", "DataLoader_Images_per_Sec", True),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load_hosts():
    hosts = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        name = data.get("Meta", {}).get("host") or os.path.splitext(os.path.basename(path))[0]
        hosts[name] = data
    return hosts


def metric_value(data, section, key):
    return (data.get(section) or {}).get(key)


def console_table(hosts):
    cols = list(hosts.keys())
    width = max([len(m[0]) for m in METRICS] + [12])
    header = "Metric".ljust(width) + "".join(h.rjust(16) for h in cols)
    print(header)
    print("-" * len(header))
    for label, section, key, _ in METRICS:
        row = label.ljust(width)
        for h in cols:
            v = metric_value(hosts[h], section, key)
            row += (f"{v:.1f}" if isinstance(v, (int, float)) else "-").rjust(16)
        print(row)
    print()
    for h in cols:
        m = hosts[h].get("Meta", {})
        print(f"  {h}: {m.get('gpu','?')} x{m.get('gpu_count','?')}  ({m.get('vram_gb_each','?')} GB each)")


def html_table(hosts):
    cols = list(hosts.keys())
    out = ['<table>', '  <tr><th>Benchmark</th>' + "".join(f'<th>{html.escape(h)}</th>' for h in cols) + '</tr>']
    out.append('  <tr><td><b>GPU</b></td>' + "".join(
        f'<td>{html.escape(str(hosts[h].get("Meta",{}).get("gpu","-")))}</td>' for h in cols) + '</tr>')
    for label, section, key, _ in METRICS:
        cells = ""
        for h in cols:
            v = metric_value(hosts[h], section, key)
            cells += f'<td>{v:.1f}</td>' if isinstance(v, (int, float)) else '<td>-</td>'
        out.append(f'  <tr><td>{html.escape(label)}</td>{cells}</tr>')
    out.append('</table>')
    return "\n".join(out)


def _fmt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def svg_chart(hosts):
    """Roomy horizontal bar chart. Bars are normalised per-metric (longest = best in that
    metric) and labelled with the raw value, so different unit scales stay readable."""
    cols = list(hosts.keys())
    groups = [(label, [metric_value(hosts[h], section, key) for h in cols])
              for label, section, key, _ in METRICS]

    n = len(cols)
    LM, RM, TM, BM = 250, 80, 58, 64           # generous left margin for metric labels
    bar_h, bar_gap, grp_gap = 17, 5, 26
    grp_h = n * (bar_h + bar_gap)
    W = 960
    plot_w = W - LM - RM
    H = TM + len(groups) * (grp_h + grp_gap) + BM

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2:.0f}" y="32" text-anchor="middle" font-size="19" font-weight="700" '
             f'fill="#1f2937">ML benchmark by metric — longest bar = best</text>')

    y = TM
    for gi, (label, vals) in enumerate(groups):
        nums = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        mx = max(nums) if nums else 1.0
        # zebra band behind the group
        if gi % 2 == 0:
            s.append(f'<rect x="0" y="{y-bar_gap:.0f}" width="{W}" height="{grp_h+bar_gap:.0f}" fill="#fafbfc"/>')
        # metric label (left, vertically centred)
        s.append(f'<text x="{LM-14}" y="{y + grp_h/2 + 4:.0f}" text-anchor="end" font-size="13" '
                 f'font-weight="600" fill="#374151">{html.escape(label)}</text>')
        s.append(f'<line x1="{LM}" y1="{y-bar_gap:.0f}" x2="{LM}" y2="{y+grp_h:.0f}" stroke="#dfe3e8"/>')
        for i, h in enumerate(cols):
            v = vals[i]
            by = y + i * (bar_h + bar_gap)
            c = COLORS[i % len(COLORS)]
            if isinstance(v, (int, float)) and v > 0:
                bw = max(3, v / mx * plot_w)
                s.append(f'<rect x="{LM}" y="{by:.0f}" width="{bw:.1f}" height="{bar_h}" fill="{c}" rx="2"/>')
                s.append(f'<text x="{LM+bw+7:.1f}" y="{by+bar_h-4:.0f}" font-size="12" fill="#374151">{_fmt(v)}</text>')
            else:
                s.append(f'<text x="{LM+5}" y="{by+bar_h-4:.0f}" font-size="11" fill="#b0b4ba">n/a</text>')
        y += grp_h + grp_gap

    # legend (wraps if needed)
    lx, ly = LM, H - 30
    for i, h in enumerate(cols):
        gpu = str(hosts[h].get("Meta", {}).get("gpu", "")).replace("NVIDIA ", "").replace("-SXM4-80GB", "")
        lbl = f"{h} · {gpu}" if gpu else h
        wlen = 34 + int(len(lbl) * 7.0)
        if lx + wlen > W - 10:
            lx = LM
            ly += 20
        s.append(f'<rect x="{lx:.0f}" y="{ly-11}" width="12" height="12" rx="2" fill="{COLORS[i % len(COLORS)]}"/>')
        s.append(f'<text x="{lx+17:.0f}" y="{ly:.0f}" font-size="12" fill="#4b5563">{html.escape(lbl)}</text>')
        lx += wlen
    s.append('</svg>')
    return "\n".join(s)


def main():
    hosts = load_hosts()
    if not hosts:
        print(f"No results found in {RESULTS}/*.yaml — run ./run.sh on each server and copy them here.")
        return
    console_table(hosts)
    os.makedirs(ASSETS, exist_ok=True)
    with open(os.path.join(ASSETS, "benchmarks.svg"), "w") as f:
        f.write(svg_chart(hosts))
    with open(os.path.join(ASSETS, "benchmarks.html"), "w") as f:
        f.write(html_table(hosts))
    print(f"\nWrote {ASSETS}/benchmarks.svg and {ASSETS}/benchmarks.html")


if __name__ == "__main__":
    main()
