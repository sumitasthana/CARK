"""Draw the figures for a trajectory report edition.

Each edition of the trajectory report keeps its own dated figures, so a later
edition never changes the charts an earlier one was read with. This script
writes one SVG per chart into docs/reports/trajectory/figures/<date>/, which is
the path the report markdown links to relatively.

Every SVG carries its own palette plus a dark override, because GitHub serves
them as plain images with no page styles around them.

    python scripts/build_trajectory_figures.py --date 2026-10-04
    python scripts/build_trajectory_figures.py --date 2026-10-04 --check
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "docs" / "experiments" / "forgetting_traces.csv"
FIGURES = ROOT / "docs" / "reports" / "trajectory" / "figures"
DEFAULT_HISTORY = (ROOT / "ops-docs" / "experiment_evidence" / "asof_20261004"
                   / "drive" / "full_sequence_27step_candidate"
                   / "history_seq1_resnet50_seed0.json")

# Runs shown in the per-run comparison, in the order they were performed.
RUN_ORDER = ["E08", "E09", "E10", "E11", "E12", "E13", "E14", "E15",
             "forgetting-30step-20260929"]

LIGHT = dict(bg="#ffffff", ink="#1b2027", muted="#5e6878", line="#d5dae2",
             soft="#e6eaf0", target="#4a63b5", retain="#2f7f72", fail="#9c4a3c",
             bandr="#2f7f7219", bandp="#2f7f4f17")
DARK = dict(bg="#1a1f26", ink="#e4e8ee", muted="#9aa5b5", line="#333c49",
            soft="#28303a", target="#8ba0e0", retain="#58b7a6", fail="#dd8b78",
            bandr="#58b7a61f", bandp="#6cc48d1c")

STYLE_RULES = """
  svg{font-family:ui-monospace,Consolas,monospace}
  text{fill:var(--muted);font-size:12px}
  .ct{fill:var(--ink);font-family:system-ui,sans-serif;font-weight:600}
  .sub{fill:var(--muted);font-family:system-ui,sans-serif}
  .grid{stroke:var(--soft);stroke-width:1}
  .axis{stroke:var(--line);stroke-width:1}
  .chance{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 3;opacity:.7}
  .target-line{stroke:var(--target);stroke-width:1.4;stroke-dasharray:5 4}
  .band-retain{fill:var(--bandr)} .band-pass{fill:var(--bandp)}
  .ln-target{stroke:var(--target);stroke-width:2.2;fill:none}
  .ln-retain{stroke:var(--retain);stroke-width:2.2;fill:none}
  .dot-target{fill:var(--target)} .dot-retain{fill:var(--retain)}
  .bar-target{fill:var(--target);opacity:.85}
  .lbl-target{fill:var(--target)} .lbl-retain{fill:var(--retain)}
  .lbl-chance{fill:var(--muted)} .lbl-fail{fill:var(--fail)}
  .bar-learn{fill:var(--retain);opacity:.8}
  .bar-before{fill:var(--fail);opacity:.8}
  .bar-spill{fill:var(--fail);opacity:.8}
  .stem{stroke:var(--line);stroke-width:1.5}
  .val{fill:var(--ink)} .val-sm{fill:var(--muted)}
"""


def read_traces(path=TRACES):
    """Per-step task 3 and task 0 accuracy for every recorded forgetting run."""
    runs = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            runs.setdefault(row["run"], []).append({
                "s": int(row["step"]),
                "t3": float(row["task3_accuracy_pct"]),
                "t0": float(row["task0_accuracy_pct"]),
            })
    return {run: sorted(steps, key=lambda r: r["s"]) for run, steps in runs.items()}


def tokens(values):
    return ";".join("--%s:%s" % (key, value) for key, value in values.items())


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;")


def wrap(w, h, body, label):
    style = ("<style>\n  svg{%s}\n"
             "  @media (prefers-color-scheme:dark){svg{%s}}\n%s</style>\n"
             % (tokens(LIGHT), tokens(DARK), STYLE_RULES))
    return ('<svg viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
            'aria-label="%s" xmlns="http://www.w3.org/2000/svg">\n'
            '%s<rect width="%d" height="%d" fill="var(--bg)"/>\n%s</svg>\n'
            % (w, h, w, h, esc(label), style, w, h, body))


def txt(x, y, s, cls="", anchor="start", size=12):
    return ('<text x="%.1f" y="%.1f" text-anchor="%s" font-size="%d" '
            'class="%s">%s</text>\n' % (x, y, anchor, size, cls, esc(s)))


def trace_panel(traces, run, title, xmax, ymax=50, w=430, h=300):
    """One run's accuracy over its forgetting updates."""
    v = traces[run]
    L, R, T, B = 46, 14, 40, 44
    pw, ph = w - L - R, h - T - B
    fx = lambda s: L + pw * s / xmax
    fy = lambda a: T + ph * (1 - a / ymax)
    o = [txt(0, 16, title, "ct", size=14)]
    t0s = v[0]["t0"]
    top = fy(min(ymax, t0s + 5))
    o.append('<rect x="%d" y="%.1f" width="%d" height="%.1f" class="band-retain"/>\n'
             % (L, top, pw, abs(fy(t0s - 5) - top)))
    for a in range(0, ymax + 1, 10):
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>\n'
                 % (L, fy(a), L + pw, fy(a)))
        o.append(txt(L - 7, fy(a) + 4, a, "ax", "end"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="chance"/>\n'
             % (L, fy(10), L + pw, fy(10)))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="target-line"/>\n'
             % (L, fy(12), L + pw, fy(12)))
    for s in range(0, xmax + 1, 10 if xmax > 10 else 2):
        o.append(txt(fx(s), T + ph + 18, s, "ax", "middle"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>\n'
             % (L, T + ph, L + pw, T + ph))
    for key, cls in (("t0", "ln-retain"), ("t3", "ln-target")):
        d = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", fx(r["s"]), fy(r[key]))
                     for i, r in enumerate(v))
        o.append('<path d="%s" class="%s"/>\n' % (d, cls))
    last = v[-1]
    for key, cls in (("t0", "dot-retain"), ("t3", "dot-target")):
        o.append('<circle cx="%.1f" cy="%.1f" r="3.5" class="%s"/>\n'
                 % (fx(last["s"]), fy(last[key]), cls))
        o.append(txt(fx(last["s"]) - 6, fy(last[key]) - 8, "%.1f" % last[key],
                     cls.replace("dot", "lbl"), "end"))
    o.append(txt(L, h - 6, "unlearning update", "ax"))
    o.append(txt(0, T - 10, "accuracy (%)", "ax"))
    return w, h, "".join(o), title


def chart_best(traces):
    """Lowest target accuracy each run reached while retention still held."""
    best = []
    for run in RUN_ORDER:
        v = traces[run]
        t0s = v[0]["t0"]
        ok = [r for r in v if r["s"] > 0 and abs(r["t0"] - t0s) < 5]
        name = "30-step run" if run.startswith("forget") else run
        best.append((name, min(ok, key=lambda r: r["t3"]) if ok else None))
    w, h, L, R, T, B, xmax = 760, 340, 112, 70, 34, 42, 28
    pw, ph = w - L - R, h - T - B
    fx = lambda a: L + pw * a / xmax
    row = ph / len(best)
    title = "Lowest task 3 accuracy reached while task 0 stayed within 5 points"
    o = [txt(0, 16, title, "ct", size=14)]
    o.append('<rect x="%d" y="%d" width="%.1f" height="%.1f" class="band-pass"/>\n'
             % (L, T, fx(12) - L, ph))
    for a in range(0, xmax + 1, 4):
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" class="grid"/>\n'
                 % (fx(a), T, fx(a), T + ph))
        o.append(txt(fx(a), T + ph + 18, a, "ax", "middle"))
    o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" class="target-line"/>\n'
             % (fx(12), T - 6, fx(12), T + ph + 2))
    o.append(txt(fx(12) + 5, T - 10, "criterion: 12% or lower", "lbl-target"))
    for i, (name, b) in enumerate(best):
        y = T + row * i + row / 2
        o.append(txt(L - 10, y + 4, name, "ax", "end"))
        if b is None:
            o.append(txt(L + 6, y + 4, "no update held task 0 within 5 points", "lbl-fail"))
            continue
        o.append('<rect x="%d" y="%.1f" width="%.1f" height="18" rx="2" '
                 'class="bar-target"/>\n' % (L, y - 9, fx(b["t3"]) - L))
        o.append(txt(fx(b["t3"]) + 7, y + 4,
                     "%.1f%%  at update %d" % (b["t3"], b["s"]), "val"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>\n'
             % (L, T + ph, L + pw, T + ph))
    o.append(txt(L, h - 6,
                 "task 3 validation accuracy (%).  10% is random guessing.", "ax"))
    return w, h, "".join(o), title


def chart_decay(history):
    """How far each task had already fallen before its unlearn request ran."""
    learned, pairs = {}, []
    for r in history:
        if r["action"] == "learn":
            learned[r["task"]] = r["after"][r["task"]]
        else:
            pairs.append((r["task"], learned[r["task"]], r["before"][r["task"]]))
    w, h, L, R, T, B, ymax = 760, 350, 46, 12, 54, 62, 50
    pw, ph = w - L - R, h - T - B
    fy = lambda a: T + ph * (1 - a / ymax)
    gw, bw = pw / len(pairs), 20
    title = "Task accuracy after learning and immediately before its unlearn request"
    o = [txt(0, 16, title, "ct", size=14)]
    o.append(txt(0, 34, "left bar: accuracy just after the task was learned.  "
                        "right bar: accuracy just before its unlearn request.", "sub"))
    for a in range(0, ymax + 1, 10):
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>\n'
                 % (L, fy(a), L + pw, fy(a)))
        o.append(txt(L - 7, fy(a) + 4, a, "ax", "end"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="chance"/>\n'
             % (L, fy(10), L + pw, fy(10)))
    o.append(txt(L + pw, fy(10) - 6, "random guessing = 10%", "lbl-chance", "end"))
    for i, (task, after, before) in enumerate(pairs):
        cx = L + gw * i + gw / 2
        o.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" rx="2" '
                 'class="bar-learn"/>\n' % (cx - bw - 2, fy(after), bw, T + ph - fy(after)))
        o.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" rx="2" '
                 'class="bar-before"/>\n' % (cx + 2, fy(before), bw, T + ph - fy(before)))
        o.append(txt(cx - bw / 2 - 2, fy(after) - 5, "%.0f" % after, "val-sm", "middle", 11))
        o.append(txt(cx + bw / 2 + 2, fy(before) - 5, "%.0f" % before, "val-sm", "middle", 11))
        o.append(txt(cx, T + ph + 18, "U%s" % task, "ax", "middle"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>\n'
             % (L, T + ph, L + pw, T + ph))
    o.append(txt(L, h - 24, "unlearn requests, in the order the sequence ran", "ax"))
    o.append(txt(0, T - 10, "accuracy (%)", "ax"))
    return w, h, "".join(o), title


def chart_spill(history):
    """Total accuracy change across other tasks at each unlearn request."""
    spills = [(r["task"], sum(abs(r["after"][k] - v)
                              for k, v in r["before"].items() if k != r["task"]))
              for r in history if r["action"] == "forget"]
    w, h, L, R, T, B, ymax = 760, 300, 46, 12, 50, 56, 55
    pw, ph = w - L - R, h - T - B
    fy = lambda a: T + ph * (1 - a / ymax)
    gw, bw = pw / len(spills), 30
    title = "Spill at each unlearn request"
    o = [txt(0, 16, title, "ct", size=14)]
    o.append(txt(0, 34, "total accuracy change across all other tasks, "
                        "in percentage points", "sub"))
    for a in range(0, ymax + 1, 10):
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>\n'
                 % (L, fy(a), L + pw, fy(a)))
        o.append(txt(L - 7, fy(a) + 4, a, "ax", "end"))
    for i, (task, sp) in enumerate(spills):
        cx = L + gw * i + gw / 2
        o.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" rx="2" '
                 'class="bar-spill"/>\n' % (cx - bw / 2, fy(sp), bw, T + ph - fy(sp)))
        o.append(txt(cx, fy(sp) - 6, "%.1f" % sp, "val-sm", "middle", 11))
        o.append(txt(cx, T + ph + 18, "U%s" % task, "ax", "middle"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="target-line"/>\n'
             % (L, fy(3), L + pw, fy(3)))
    o.append(txt(L + pw, fy(3) - 6, "criterion: under 3 points", "lbl-target", "end"))
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>\n'
             % (L, T + ph, L + pw, T + ph))
    o.append(txt(0, T - 10, "points", "ax"))
    return w, h, "".join(o), title


def chart_replay():
    """Spread of the repeated task-9 runs. Values are transcribed from the audits."""
    groups = [("original run, inside the full sequence", [34.4]),
              ("replays from the saved model, separate sessions", [23.6, 31.8]),
              ("two runs inside one session", [32.0, 32.4])]
    w, h, L, R, T, B = 760, 270, 300, 70, 56, 40
    pw, ph = w - L - R, h - T - B
    lo, hi = 20, 42
    fx = lambda a: L + pw * (a - lo) / (hi - lo)
    title = "Task 0 accuracy after learning task 9, across repeated runs"
    o = [txt(0, 16, title, "ct", size=14)]
    o.append(txt(0, 34, "every run restored the identical saved model, with matching "
                        "file hashes and settings", "sub"))
    for a in range(lo, hi + 1, 4):
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" class="grid"/>\n'
                 % (fx(a), T, fx(a), T + ph))
        o.append(txt(fx(a), T + ph + 18, a, "ax", "middle"))
    rows = sum(len(values) for _, values in groups)
    step, i = ph / rows, 0
    for label, values in groups:
        o.append(txt(L - 14, T + step * i + 13, label, "ax", "end"))
        for j, val in enumerate(values):
            y = T + step * (i + j) + step / 2
            o.append('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" class="stem"/>\n'
                     % (L, y, fx(val), y))
            o.append('<circle cx="%.1f" cy="%.1f" r="6" class="dot-retain"/>\n'
                     % (fx(val), y))
            o.append(txt(fx(val) + 12, y + 4, "%.1f%%" % val, "val"))
        i += len(values)
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>\n'
             % (L, T + ph, L + pw, T + ph))
    o.append(txt(L, h - 6, "task 0 accuracy after learning task 9 (%)", "ax"))
    return w, h, "".join(o), title


def build(traces, history):
    """Every chart, keyed by the filename the report markdown links to."""
    return {
        "c1a": trace_panel(traces, "E08", "E08: ten updates, gamma 1e-4", 10),
        "c1b": trace_panel(traces, "forgetting-30step-20260929",
                           "30-step run: gamma 5e-6", 30),
        "c2": chart_best(traces),
        "c3": chart_decay(history),
        "c4": chart_spill(history),
        "c5": chart_replay(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True,
                        help="Edition date, as in docs/reports/trajectory/<date>.md")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY,
                        help="Full-sequence request history JSON")
    parser.add_argument("--check", action="store_true",
                        help="Compare against the committed figures without writing")
    args = parser.parse_args()

    traces = read_traces()
    missing = [run for run in RUN_ORDER if run not in traces]
    if missing:
        raise ValueError(f"Runs missing from {TRACES.name}: {', '.join(missing)}")
    history = json.loads(args.history.read_text(encoding="utf-8"))

    output = FIGURES / args.date
    for name, (w, h, body, label) in build(traces, history).items():
        text = wrap(w, h, body, label)
        destination = output / f"{name}.svg"
        if args.check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
                raise ValueError(f"Figure needs regeneration: {destination}")
        else:
            output.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
    print(f"{'Checked' if args.check else 'Wrote'} 6 figures in {output}")


if __name__ == "__main__":
    main()
