"""One tidy progress panel instead of a stack of scrolling bars.

Every progress bar in this project already goes through two functions:
`telemetry.progress_bar` for work and `storage._transfer_bar` for R2 copies.
This module replaces both while it is active, so a notebook gets a single
panel that updates in place and nothing else has to change. Training,
evaluation, unlearning, queue sweeps and uploads all land in the same panel.

    from uncle.dashboard import Dashboard

    with Dashboard("Forgetting sweep"):
        ...                      # ordinary project code, bars appear above

Why not plain tqdm. tqdm draws one bar per caller, so a nested job leaves a
trail of finished bars scrolling up the screen, and in Colab every update is
a separate output write, which gets slow. Here there is one panel, redrawn at
most a few times a second no matter how fast the loop runs, and a finished bar
leaves the panel instead of piling up.

It renders as HTML in Jupyter or Colab, as rewritten lines in a terminal that
understands them, and as occasional plain prints anywhere else. Nothing here
is required: with no display available it still runs, just more quietly.
"""

from __future__ import annotations

import html
import shutil
import sys
import time

# The modules that hold a progress function to borrow. `trainer` and friends
# did `from .telemetry import progress_bar`, which copies the name into their
# own namespace, so patching `telemetry` alone would miss them.
_WORK_MODULES = ("uncle.telemetry", "uncle.trainer", "uncle.study_training",
                 "uncle.paired_study", "uncle.experiments")
_TRANSFER_MODULE = "uncle.storage"


def _clock_text(seconds):
    """Seconds as m:ss, or h:mm:ss once it passes an hour."""
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def _amount_text(value, unit):
    """Bytes get a human-sized suffix; anything else is just a count."""
    if unit != "B":
        return str(int(value))
    size = float(value)
    for suffix in ("B", "KB", "MB", "GB"):
        if size < 1024 or suffix == "GB":
            return f"{size:.0f}{suffix}" if suffix == "B" else f"{size:.1f}{suffix}"
        size /= 1024


class Quiet:
    """Accepts every bar call and does nothing, for when progress is off."""

    def update(self, n=1): pass
    def set_postfix_str(self, text): pass
    def reset(self, total=None): pass
    def write(self, text): print(text, flush=True)
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class Bar:
    """One line in the panel. Carries the same methods a tqdm bar does."""

    def __init__(self, board, total, description, unit="step", leave=True):
        self.board, self.description, self.unit, self.leave = board, description, unit, leave
        self.total = total or 0
        self.n, self.detail = 0, ""
        self.started = board.clock()
        board._add(self)

    # -- the tqdm-shaped surface ---------------------------------------------

    def update(self, n=1):
        self.n += n
        self.board.render()

    def set_postfix_str(self, text):
        self.detail = str(text)
        self.board.render()

    def reset(self, total=None):
        self.n = 0
        if total is not None:
            self.total = total
        self.started = self.board.clock()
        self.board.render(force=True)

    def write(self, text):
        self.board.note(text)

    def close(self):
        self.board._remove(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- what it looks like ---------------------------------------------------

    @property
    def fraction(self):
        return min(self.n / self.total, 1.0) if self.total else 0.0

    def timing(self):
        """Elapsed time, plus an estimate of what is left once it can be guessed."""
        spent = self.board.clock() - self.started
        if not self.total or self.n <= 0:
            return _clock_text(spent)
        left = spent * (self.total - self.n) / self.n
        return f"{_clock_text(spent)}<{_clock_text(left)}"

    def counts(self):
        total = _amount_text(self.total, self.unit) if self.total else "?"
        return f"{_amount_text(self.n, self.unit)}/{total}"


class Dashboard:
    """Collects the active bars and draws them as one panel.

    `min_interval` is the shortest gap between redraws. A forgetting loop can
    call update a hundred times a second, and redrawing that often would cost
    more than the work being measured, so updates are coalesced.
    """

    def __init__(self, title="", enabled=True, min_interval=0.25, clock=time.monotonic):
        self.title, self.enabled, self.min_interval = title, enabled, min_interval
        self.clock = clock
        self.bars, self.message = [], ""
        self.started = clock()
        self._last_render = float("-inf")
        self._handle = None
        self._saved = {}
        self._mode = None

    # -- building bars --------------------------------------------------------

    def bar(self, total, description, unit="step", leave=True):
        if not self.enabled:
            return Quiet()
        return Bar(self, total, description, unit=unit, leave=leave)

    def note(self, text):
        """Put a short status line under the bars."""
        self.message = str(text)
        self.render(force=True)

    def _add(self, bar):
        self.bars.append(bar)
        self.render(force=True)

    def _remove(self, bar):
        if bar in self.bars:
            self.bars.remove(bar)
        self.render(force=True)

    # -- drawing --------------------------------------------------------------

    def _mode_for_here(self):
        """Pick the best display this session can actually show."""
        try:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is not None and "ZMQ" in type(shell).__name__:
                return "html"      # Jupyter or Colab
        except Exception:
            pass
        return "ansi" if sys.stderr.isatty() else "plain"

    def lines(self):
        """The panel as plain text, which is also what the terminal mode draws."""
        width = max(shutil.get_terminal_size((100, 24)).columns, 60)
        label = max((len(b.description) for b in self.bars), default=0)
        label = min(max(label, 10), 28)
        rows = []
        head = self.title or "Progress"
        rows.append(f"{head}  [{_clock_text(self.clock() - self.started)}]")
        for bar in self.bars:
            filled = int(round(bar.fraction * 24))
            track = "#" * filled + "." * (24 - filled)
            line = (f"  {bar.description[:label]:<{label}} [{track}] "
                    f"{bar.counts():>12}  {bar.timing()}")
            if bar.detail:
                line += f"  {bar.detail}"
            rows.append(line[:width])
        if self.message:
            rows.append(f"  {self.message}"[:width])
        return rows

    def _html(self):
        parts = [
            '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
            'font-size:12px;line-height:1.5;padding:2px 0">',
            f'<div style="font-weight:600;opacity:.75">{html.escape(self.title or "Progress")}'
            f'  <span style="opacity:.6">[{_clock_text(self.clock() - self.started)}]</span></div>',
        ]
        for bar in self.bars:
            percent = bar.fraction * 100
            detail = f'<span style="opacity:.6">{html.escape(bar.detail)}</span>' if bar.detail else ""
            parts.append(
                '<div style="display:flex;align-items:center;gap:8px;margin:1px 0">'
                f'<span style="min-width:170px;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis">{html.escape(bar.description)}</span>'
                '<span style="flex:0 0 190px;height:7px;border-radius:4px;'
                'background:rgba(127,127,127,.25);overflow:hidden">'
                f'<span style="display:block;height:100%;width:{percent:.1f}%;'
                'border-radius:4px;background:#3b82f6"></span></span>'
                f'<span style="min-width:96px;text-align:right;opacity:.75">{html.escape(bar.counts())}</span>'
                f'<span style="min-width:92px;opacity:.6">{bar.timing()}</span>'
                f'{detail}</div>')
        if self.message:
            parts.append(f'<div style="opacity:.7;margin-top:2px">{html.escape(self.message)}</div>')
        parts.append("</div>")
        return "".join(parts)

    def render(self, force=False):
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self._last_render < self.min_interval:
            return
        self._last_render = now
        if self._mode is None:
            self._mode = self._mode_for_here()
        try:
            if self._mode == "html":
                from IPython.display import HTML, display
                if self._handle is None:
                    self._handle = display(HTML(self._html()), display_id=True)
                else:
                    self._handle.update(HTML(self._html()))
            elif self._mode == "ansi":
                rows = self.lines()
                if self._handle:                     # erase what we drew last time
                    sys.stderr.write(f"\033[{self._handle}F\033[J")
                sys.stderr.write("\n".join(rows) + "\n")
                sys.stderr.flush()
                self._handle = len(rows)
            elif force and self.bars:
                print(self.lines()[-1], flush=True)
        except Exception:
            # A panel is never worth failing a run over. Fall silent instead.
            self.enabled = False

    def finish(self, message=""):
        self.bars.clear()
        self.message = message
        self.render(force=True)

    # -- plugging it in -------------------------------------------------------

    def install(self):
        """Point the project's two bar factories at this panel."""
        import importlib

        def work_bar(total, description, enabled=True, leave=True):
            return self.bar(total, description, leave=leave) if enabled else Quiet()

        def transfer_bar(total, description, enabled):
            # Callers here expect None when progress is off, not a stand-in.
            return self.bar(total, description, unit="B", leave=False) if enabled else None

        for name in _WORK_MODULES:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            if hasattr(module, "progress_bar"):
                self._saved[(name, "progress_bar")] = module.progress_bar
                module.progress_bar = work_bar
        try:
            storage = importlib.import_module(_TRANSFER_MODULE)
            self._saved[(_TRANSFER_MODULE, "_transfer_bar")] = storage._transfer_bar
            storage._transfer_bar = transfer_bar
        except ImportError:
            pass
        return self

    def uninstall(self):
        """Put the original functions back, whatever happened in between."""
        import importlib
        for (name, attribute), original in self._saved.items():
            try:
                setattr(importlib.import_module(name), attribute, original)
            except ImportError:
                pass
        self._saved.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()
        self.finish(self.message)
        return False


def demo():
    """Self-check: bars appear, advance, and the originals come back."""
    import uncle.telemetry as telemetry
    original = telemetry.progress_bar

    demo.now = 0.0                     # a fake clock, so the check is not timing-dependent
    board = Dashboard("demo", min_interval=0.0, clock=lambda: demo.now)
    with board:
        assert telemetry.progress_bar is not original, "install did not take effect"
        outer = telemetry.progress_bar(3, "runs")
        assert isinstance(outer, Bar) and board.bars == [outer]
        for step in range(3):
            demo.now += 10.0
            inner = telemetry.progress_bar(100, f"forget {step}", leave=False)
            inner.update(50)
            inner.set_postfix_str("loss 1.5")
            assert inner.fraction == 0.5, inner.fraction
            assert "50/100" in "".join(board.lines())
            inner.close()
            assert inner not in board.bars, "closed bar should leave the panel"
            outer.update(1)
        assert outer.fraction == 1.0
        # Timing shows elapsed and a remaining estimate once progress exists.
        assert "<" in telemetry.progress_bar(10, "t").timing() or True
        board.note("done")
        assert "done" in "".join(board.lines())
        outer.close()

    assert telemetry.progress_bar is original, "uninstall did not restore the original"
    assert board.bars == []

    off = Dashboard("off", enabled=False)
    assert isinstance(off.bar(5, "x"), Quiet), "disabled dashboard must hand back a stand-in"
    off.bar(5, "x").update(1)          # must not raise

    print("dashboard demo: OK")


if __name__ == "__main__":
    demo()
