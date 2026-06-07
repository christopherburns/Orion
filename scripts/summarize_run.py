#!/usr/bin/env python3
"""Summarize an iterative_gameplay run.json as a flat table.

Reads the master run.json and prints one row per cycle, showing win rates
against each evaluation opponent, the best training losses, average turn
counts in the generated training data, and average turn counts in eval play.

Usage:
   summarize_run.py path/to/run.json
"""

import json
import sys
from pathlib import Path


# ── Cell extractors ────────────────────────────────────────────────────────────

def winRate (cycle, evalKey):
   """Decisive-game win rate for the trained model in the named eval, or None."""
   e = cycle.get(evalKey)
   if e is None:
      return None
   return e["results"]["perPlayer"][0]["winRateOverDecisive"]


def avgEvalTurns (cycle):
   """Mean avg-turns-per-game across whatever eval opponents this cycle has."""
   keys = ("evalVsRandom", "evalVsHeuristic", "evalVsPrev")
   turns = [cycle[k]["results"]["avgTurnsPerGame"]
            for k in keys if cycle.get(k) is not None]
   return sum(turns) / len(turns) if turns else None


def trainDataTurns (cycle):
   """Average turns per game in this cycle's generated training data, or None
   for the data-start first cycle (no generation happened)."""
   g = cycle.get("generate")
   return g["results"]["avgExamplesPerGame"] if g else None


# ── Cell formatters ────────────────────────────────────────────────────────────

def pct (x):
   return f"{x*100:5.1f}%" if x is not None else "    —"


def num (x, fmt=".3f"):
   return format(x, fmt) if x is not None else "    —"


# ── Main ───────────────────────────────────────────────────────────────────────

def main ():
   if len(sys.argv) != 2:
      sys.stderr.write(__doc__)
      sys.exit(1)
   path = Path(sys.argv[1])
   if not path.exists():
      sys.stderr.write(f"File not found: {path}\n")
      sys.exit(1)

   r = json.load(open(path))
   cfg = r.get("config", {})

   print(f"Run:      {r.get('runDir', '?')}")
   print(f"Started:  {r.get('startedAt', '?')}")
   print(f"Config:   games/cycle={cfg.get('gamesPerCycle')}  "
         f"mc={cfg.get('monteCarloSamples')}  epochs={cfg.get('epochs')}  "
         f"batch={cfg.get('trainingBatchSize')}  lr={cfg.get('learningRate')}  "
         f"eval-games={cfg.get('evalGames')}  accumulate={cfg.get('accumulateData')}")
   print()

   header = (f"{'Gen':>3}  {'vs Rand':>7}  {'vs Heur':>7}  {'vs Prev':>7}  "
             f"{'Gate':>4}  {'PolLoss':>7}  {'ValLoss':>7}  "
             f"{'DataTurns':>9}  {'EvalTurns':>9}")
   print(header)
   print("-" * len(header))

   for c in r.get("cycles", []):
      gate = "yes" if c.get("championAccepted") else " no"
      train = c.get("train", {}).get("results", {})
      row = (f"{c['cycleIndex']:>3}  "
             f"{pct(winRate(c, 'evalVsRandom')):>7}  "
             f"{pct(winRate(c, 'evalVsHeuristic')):>7}  "
             f"{pct(winRate(c, 'evalVsPrev')):>7}  "
             f"{gate:>4}  "
             f"{num(train.get('bestValidationPolicyLoss')):>7}  "
             f"{num(train.get('bestValidationValueLoss')):>7}  "
             f"{num(trainDataTurns(c), '.1f'):>9}  "
             f"{num(avgEvalTurns(c), '.1f'):>9}")
      print(row)


if __name__ == "__main__":
   main()
