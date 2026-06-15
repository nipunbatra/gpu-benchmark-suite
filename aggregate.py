"""Aggregate per-host benchmark results into a comparison table + SVG chart.

Reads results/*.yaml (one per server, produced by run.sh) and emits:
  - a console table
  - assets/benchmarks.svg   (crisp grouped bar chart, normalised speedup; no deps)
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
METRICS = [
    ("LLM (tok/s)", "LLM", "Steady_Tokens_per_Sec", True),
    ("Detection (FPS)", "Detection", "Steady_FPS", True),
    ("Classification (img/s)", "Classification", "Images_per_Sec", True),
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
    # GPU row
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


def svg_chart(hosts, width=820, height=460, pad=70):
    """Grouped bar chart of speedup relative to the slowest host per metric."""
    cols = list(hosts.keys())
    groups = []
    for label, section, key, _ in METRICS:
        vals = [metric_value(hosts[h], section, key) for h in cols]
        nums = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        base = min(nums) if nums else 1.0
        speed = [(v / base) if isinstance(v, (int, float)) and v > 0 else 0 for v in vals]
        groups.append((label, speed))

    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    max_speed = max([s for _, sp in groups for s in sp] + [1.0])
    gw = plot_w / len(groups)
    bw = gw / (len(cols) + 1)
    y0 = height - pad

    def y(v):
        return y0 - (v / max_speed) * plot_h

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif">']
    s.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    s.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">'
             f'Relative ML performance (× vs slowest server)</text>')
    # y gridlines
    ticks = max(2, int(max_speed))
    for t in range(ticks + 1):
        yy = y(t)
        s.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{width-pad}" y2="{yy:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{pad-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#666">{t}×</text>')
    # bars
    for gi, (label, speed) in enumerate(groups):
        gx = pad + gi * gw
        for ci, v in enumerate(speed):
            x = gx + (ci + 0.5) * bw
            yy = y(v)
            s.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw*0.9:.1f}" height="{y0-yy:.1f}" '
                     f'fill="{COLORS[ci % len(COLORS)]}"/>')
            if v > 0:
                s.append(f'<text x="{x+bw*0.45:.1f}" y="{yy-4:.1f}" text-anchor="middle" '
                         f'font-size="10" fill="#333">{v:.1f}</text>')
        s.append(f'<text x="{gx+gw/2:.1f}" y="{y0+18:.1f}" text-anchor="middle" font-size="12">'
                 f'{html.escape(label)}</text>')
    # legend
    lx = pad
    for ci, h in enumerate(cols):
        ly = height - 16
        s.append(f'<rect x="{lx:.1f}" y="{ly-10:.1f}" width="12" height="12" fill="{COLORS[ci % len(COLORS)]}"/>')
        s.append(f'<text x="{lx+18:.1f}" y="{ly:.1f}" font-size="12">{html.escape(h)}</text>')
        lx += 30 + len(h) * 8
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
