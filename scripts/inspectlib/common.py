"""Shared terminal formatting: colors, sparklines, bars, and layout helpers."""

import os
import sys


# ── Color handling ─────────────────────────────────────────────────────────────

def colorEnabled (noColorFlag: bool = False) -> bool:
   if noColorFlag or os.environ.get("NO_COLOR"):
      return False
   return sys.stdout.isatty()


class Palette:
   """ANSI codes, or empty strings when color is disabled."""

   def __init__ (self, enabled: bool):
      pick = (lambda code: code) if enabled else (lambda code: "")
      self.reset  = pick("\033[0m")
      self.bold   = pick("\033[1m")
      self.dim    = pick("\033[2m")
      self.green  = pick("\033[32m")
      self.red    = pick("\033[31m")
      self.yellow = pick("\033[33m")
      self.cyan   = pick("\033[36m")

   def good (self, s: str) -> str:  return f"{self.green}{s}{self.reset}"
   def bad (self, s: str) -> str:   return f"{self.red}{s}{self.reset}"
   def warn (self, s: str) -> str:  return f"{self.yellow}{s}{self.reset}"
   def faint (self, s: str) -> str: return f"{self.dim}{s}{self.reset}"
   def head (self, s: str) -> str:  return f"{self.bold}{s}{self.reset}"


# ── Sparklines and bars ────────────────────────────────────────────────────────

SPARK_CHARS = "▁▂▃▄▅▆▇█"
MISSING_CHAR = "—"


def sparkline (values: list, lo: float = None, hi: float = None) -> str:
   """One character per entry; None entries render as MISSING_CHAR.
   Scale is min..max of present values unless lo/hi are given."""
   present = [v for v in values if v is not None]
   if not present:
      return MISSING_CHAR * len(values)
   lo = min(present) if lo is None else lo
   hi = max(present) if hi is None else hi
   span = hi - lo
   out = []
   for v in values:
      if v is None:
         out.append(MISSING_CHAR)
      elif span <= 0:
         out.append(SPARK_CHARS[3])
      else:
         idx = int((v - lo) / span * (len(SPARK_CHARS) - 1) + 0.5)
         out.append(SPARK_CHARS[max(0, min(len(SPARK_CHARS) - 1, idx))])
   return "".join(out)


def bar (fraction: float, width: int = 10) -> str:
   """Horizontal bar with 1/8-block resolution, for category charts."""
   fraction = max(0.0, min(1.0, fraction))
   eighths = int(fraction * width * 8 + 0.5)
   full, rem = divmod(eighths, 8)
   partial = "▏▎▍▌▋▊▉█"[rem - 1] if rem else ""
   return "█" * full + partial


# ── Layout helpers ─────────────────────────────────────────────────────────────

RULE_WIDTH = 84


def sectionRule (title: str) -> str:
   label = f"─ {title} " if title else ""
   return label + "─" * max(0, RULE_WIDTH - len(label))


def banner (title: str, right: str = "") -> str:
   inner = RULE_WIDTH - 2                      # content width between the ║ borders
   left = f"  {title}"
   pad = inner - len(left) - len(right) - 2    # right label ends 2 cols before ║
   line = (left + " " * max(1, pad) + right + "  ")[:inner].ljust(inner)
   return ("╔" + "═" * inner + "╗\n"
           + "║" + line + "║\n"
           + "╚" + "═" * inner + "╝")


def humanBytes (n: int) -> str:
   for unit in ("B", "KB", "MB", "GB"):
      if n < 1024 or unit == "GB":
         return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
      n /= 1024.0
   return f"{n:.1f} GB"


def humanDuration (seconds: float) -> str:
   seconds = int(seconds)
   h, rem = divmod(seconds, 3600)
   m, s = divmod(rem, 60)
   if h: return f"{h}h {m:02d}m"
   if m: return f"{m}m {s:02d}s"
   return f"{s}s"


# ── Small math (no numpy) ──────────────────────────────────────────────────────

def linearFit (ys: list) -> tuple:
   """Least-squares fit y = a + b*x over x = 0..n-1.
   Returns (slope, residualStd). Requires >= 3 points."""
   n = len(ys)
   if n < 3:
      return 0.0, 0.0
   xs = range(n)
   mx = (n - 1) / 2.0
   my = sum(ys) / n
   sxx = sum((x - mx) ** 2 for x in xs)
   sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
   slope = sxy / sxx if sxx else 0.0
   resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
   var = sum(r * r for r in resid) / n
   return slope, var ** 0.5


def pearson (xs: list, ys: list) -> float:
   pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
   n = len(pairs)
   if n < 3:
      return 0.0
   mx = sum(p[0] for p in pairs) / n
   my = sum(p[1] for p in pairs) / n
   cov = sum((x - mx) * (y - my) for x, y in pairs)
   vx = sum((x - mx) ** 2 for x, _ in pairs)
   vy = sum((y - my) ** 2 for _, y in pairs)
   denom = (vx * vy) ** 0.5
   return cov / denom if denom else 0.0
