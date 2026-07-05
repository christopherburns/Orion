"""Dashboard for an iterative_gameplay run.json.

Top-down structure:
   render(runJson, palette, genRange)   — entry point, prints the whole dashboard
      _header
      _trajectory
      _moveMix
      _generationsTable (+ glossary)
      _checks
   Series extraction lives in _seriesFrom; each check in _checks is one
   hardcoded rule with its trigger arithmetic printed inline.
"""

from . import common as C


# ── Per-generation series extraction ───────────────────────────────────────────

def _winRate (gen, key):
   e = gen.get(key)
   return e["results"]["perPlayer"][0]["winRateOverDecisive"] * 100 if e else None


def _seriesFrom (gens):
   """Build aligned per-generation series (None where a value is absent)."""
   s = {
      "gen":       [g["generationIndex"] for g in gens],
      "rand":      [_winRate(g, "evalVsRandom") for g in gens],
      "heur":      [_winRate(g, "evalVsHeuristic") for g in gens],
      "prev":      [_winRate(g, "evalVsPrev") for g in gens],
      "accepted":  [g.get("championAccepted") for g in gens],
      "polL":      [], "valL": [], "epochs": [], "bestEp": [], "early": [], "pool": [],
      "dataTurns": [], "evalTurns": [], "discard": [], "genSecs": [], "trainSecs": [],
   }
   for g in gens:
      tr = (g.get("train") or {}).get("results", {})
      s["polL"].append(tr.get("bestValidationPolicyLoss"))
      s["valL"].append(tr.get("bestValidationValueLoss"))
      s["epochs"].append(tr.get("epochsCompleted"))
      s["bestEp"].append(tr.get("bestEpoch"))
      s["early"].append(tr.get("earlyStopped"))
      pool = tr.get("trainingExamples")
      s["pool"].append(pool + tr.get("validationExamples", 0) if pool is not None else None)
      s["trainSecs"].append((g.get("train") or {}).get("elapsedSeconds"))

      gr = g.get("generate") or {}
      res = gr.get("results", {})
      # avgTurnsPerGame is the corrected player-turn count; older reports only
      # have avgExamplesPerGame (decisions incl. discards) — over-reports turns.
      s["dataTurns"].append(res.get("avgTurnsPerGame", res.get("avgExamplesPerGame")))
      s["genSecs"].append(gr.get("elapsedSeconds"))
      s["discard"].append(_discardPct(res.get("moveStatistics")))

      evalTurns = [g[k]["results"]["avgTurnsPerGame"]
                   for k in ("evalVsRandom", "evalVsHeuristic", "evalVsPrev") if g.get(k)]
      s["evalTurns"].append(sum(evalTurns) / len(evalTurns) if evalTurns else None)
   return s


def _discardPct (ms):
   if not ms:
      return None
   tot = ms["totals"]
   total = tot["winner"] + tot["loser"] + tot["tied"]
   d = ms["byMoveType"].get("discardGem", {})
   disc = d.get("winner", 0) + d.get("loser", 0) + d.get("tied", 0)
   return disc / total * 100 if total else None


def _last (xs):
   present = [x for x in xs if x is not None]
   return present[-1] if present else None


def _first (xs):
   present = [x for x in xs if x is not None]
   return present[0] if present else None


# ── Sections ───────────────────────────────────────────────────────────────────

def _header (run, gens, p):
   cfg = run.get("config", {})
   archVersions = sorted({g[k]["parameters"]["architectureVersion"]
                          for g in gens for k in ("generate", "train")
                          if g.get(k) and "architectureVersion" in g[k].get("parameters", {})})
   arch = f"arch v{'/'.join(str(v) for v in archVersions)}" if archVersions else "arch —"
   print(C.banner(f"ORION RUN · {run.get('runDir', '?')}",
                  f"{arch} · schema {run.get('schemaVersion', '?')}"))

   started, completed = run.get("startedAt", "?"), run.get("completedAt")
   wall = ""
   try:
      from datetime import datetime
      t0 = datetime.fromisoformat(started)
      t1 = datetime.fromisoformat(completed)
      wall = C.humanDuration((t1 - t0).total_seconds())
   except Exception:
      wall = "?"
   print(f"  started   {started} · finished  {completed or 'in progress'} · wall  {wall}")

   def opt (key, label, fmt="{}"):
      v = cfg.get(key)
      return f"{label}={fmt.format(v)}" if v is not None else None

   line1 = [opt("monteCarloSamples", "mc"), opt("mctsLeafBatch", "leaf"),
            (f"dirichlet α={cfg['dirichletAlpha']} ε={cfg['dirichletEpsilon']}"
             if "dirichletAlpha" in cfg else None),
            (f"target-ex={cfg['targetExamples']//1000}k" if cfg.get("targetExamples") else
             opt("gamesPerGeneration", "games/gen")),
            (opt("accumulateWindow", "window") if "accumulateWindow" in cfg else
             f"accumulate={cfg.get('accumulateData')}")]
   line2 = [f"epochs≤{cfg.get('epochs')}", opt("trainingBatchSize", "bs"),
            opt("learningRate", "lr"), opt("weightDecay", "wd"),
            (f"eval mc={cfg.get('evalMonteCarloSamples')} det={cfg.get('evalDeterminizations')}"
             if cfg.get("evalMonteCarloSamples") is not None else None)]
   print(f"  config    {' · '.join(x for x in line1 if x)}")
   print(f"            {' · '.join(x for x in line2 if x)}")

   accepted = sum(1 for a in _seriesFrom(gens)["accepted"] if a)
   champion = next((g["trainedModel"].rsplit("/", 1)[-1]
                    for g in reversed(gens) if g.get("championAccepted")), "?")
   print(f"  champion  {champion} · {len(gens)} generations · "
         f"{accepted} accepted · {len(gens) - accepted} rejected")


def _trajRow (label, values, valueFmt, note, p):
   spark = C.sparkline(values)
   value = _last(values)
   valueStr = valueFmt.format(value) if value is not None else "—"
   print(f"  {label:<13s} {spark:<22s} {valueStr:>7s}   {note}")


def _trajectory (s, p):
   print()
   print(C.sectionRule("trajectory"))
   heur0, heurN = _first(s["heur"]), _last(s["heur"])
   heurNote = f"▲ {heurN - heur0:+.1f} pts" if heur0 is not None and heurN is not None else ""
   gates = [a for a in s["accepted"] if a is not None]
   prevPresent = [v for v in s["prev"] if v is not None]
   gateNote = (f"✓ {sum(gates)}/{len(gates)} gates, avg {sum(prevPresent)/len(prevPresent):.1f}%"
               if prevPresent else "")
   dt0, dtN = _first(s["dataTurns"]), _last(s["dataTurns"])
   et0, etN = _first(s["evalTurns"]), _last(s["evalTurns"])
   di0, diN = _first(s["discard"]), _last(s["discard"])

   _trajRow("vs random",    s["rand"], "{:.1f}%", "", p)
   _trajRow("vs heuristic", s["heur"], "{:.1f}%", heurNote, p)
   _trajRow("vs prev gate", s["prev"], "{:.1f}%", gateNote, p)
   _trajRow("policy loss",  s["polL"], "{:.3f}", "", p)
   _trajRow("value loss",   s["valL"], "{:.3f}", p.warn("⚠ see checks below"), p)
   if dt0 is not None:
      _trajRow("data turns", s["dataTurns"], "{:.1f}", f"▼ {dt0:.0f}→{dtN:.0f}" if dtN < dt0 else "", p)
   if et0 is not None:
      _trajRow("eval turns", s["evalTurns"], "{:.1f}", f"▼ {et0:.0f}→{etN:.0f}" if etN < et0 else "", p)
   if di0 is not None:
      _trajRow("discard rate", s["discard"], "{:.1f}%", f"▼ {di0:.0f}%→{diN:.0f}%" if diN < di0 else "", p)


MOVE_CATEGORIES = [("purchaseTier1", "buy1"), ("purchaseTier2", "buy2"),
                   ("purchaseTier3", "buy3"), ("purchaseReserved", "buyR"),
                   ("takeThreeGems", "take3"), ("takeTwoGems", "take2"),
                   ("reserveCard", "rsrv"), ("discardGem", "disc")]


def _moveMix (gens, p):
   withStats = [(g["generationIndex"], g["generate"]["results"]["moveStatistics"])
                for g in gens if g.get("generate")
                and g["generate"]["results"].get("moveStatistics")]
   if not withStats:
      return
   print()
   print(C.sectionRule("move mix (self-play, share of all decisions)"))
   picks = sorted({0, len(withStats) // 2, len(withStats) - 1})
   print("           " + "".join(f"{label:>8s}" for _, label in MOVE_CATEGORIES))
   for i in picks:
      genIdx, ms = withStats[i]
      tot = ms["totals"]
      total = tot["winner"] + tot["loser"] + tot["tied"]
      row = f"  gen {genIdx:3d} "
      for key, _ in MOVE_CATEGORIES:
         c = ms["byMoveType"].get(key, {})
         n = c.get("winner", 0) + c.get("loser", 0) + c.get("tied", 0)
         row += f"{n / total * 100:7.1f}%"
      print(row)

   # Winner-vs-loser tier-3 signal from the newest generation
   _, msLast = withStats[-1]
   t3 = msLast["byMoveType"].get("purchaseTier3", {})
   w, l = t3.get("winner", 0), t3.get("loser", 0)
   if l > 0 and w / l >= 1.5:
      print(f"  winners buy {w / l:.1f}× more tier-3 than losers "
            f"(latest gen: {w} vs {l})")


def _generationsTable (gens, s, p):
   print()
   print(C.sectionRule("generations"))
   print("  gen  acc   rand   heur   prev   polL    valL   ep(best)    pool   turns    t·gen")
   for i, g in enumerate(gens):
      acc = "●" if s["accepted"][i] else ("○" if s["accepted"][i] is not None else " ")
      accStr = p.good(acc) if s["accepted"][i] else p.bad(acc)

      def num (v, fmt):
         return fmt.format(v) if v is not None else "    —"

      ep = (f"{s['epochs'][i]:3d}({s['bestEp'][i]})" + ("!" if s["early"][i] else " ")
            if s["epochs"][i] is not None else "     —")
      pool = f"{s['pool'][i]/1000:6.0f}k" if s["pool"][i] is not None else "      —"
      tGen = f"{s['genSecs'][i]:6.0f}s" if s["genSecs"][i] is not None else "      —"
      print(f"  {s['gen'][i]:3d}   {accStr}   "
            f"{num(s['rand'][i], '{:5.1f}')}  {num(s['heur'][i], '{:5.1f}')}  "
            f"{num(s['prev'][i], '{:5.1f}')}  "
            f"{num(s['polL'][i], '{:5.3f}')}  {num(s['valL'][i], '{:6.4f}')}  "
            f"{ep:>9s} {pool} {num(s['dataTurns'][i], '{:6.1f}')}  {tGen}")

   print()
   print(C.sectionRule("glossary"))
   for line in [
      "acc      ● champion accepted (beat prev ≥ threshold) · ○ rejected, prev retained",
      "rand     win % vs random agent, decisive games only",
      "heur     win % vs heuristic agent (same eval settings)",
      "prev     win % vs previous champion — the gating match",
      "polL     best-epoch validation policy loss (cross-entropy vs MCTS visits, nats)",
      "valL     best-epoch validation value loss (MSE vs ±1 outcome) — pool-relative,",
      "         NOT comparable across generations when the data window shifts",
      "ep(best) epochs run (best-checkpoint epoch); '!' = early-stopped",
      "pool     training examples in this generation's window (train + validation)",
      "turns    avg completed player turns per self-play game (discards excluded)",
      "t·gen    wall-clock of the generate step",
   ]:
      print(f"  {line}")


def _checks (gens, s, p):
   print()
   print(C.sectionRule("checks (hardcoded rules; trigger arithmetic shown)"))
   lines = []

   # Plateau: slope of vs-heur over last 10 gens
   heur = [v for v in s["heur"] if v is not None]
   window = heur[-10:]
   if len(window) >= 5:
      slope, resid = C.linearFit(window)
      g0 = s["gen"][len(s["gen"]) - len(window)]
      if abs(slope) < 0.3 and resid < 3.0:
         lines.append(p.warn("⚠ plateau       ") +
                      f" vs-heur slope {slope:+.2f} pts/gen over g{g0}–g{s['gen'][-1]} (|slope| < 0.3, σ {resid:.1f} < 3)")
         plateaued = True
      else:
         lines.append(p.good("✓ still climbing") +
                      f" vs-heur slope {slope:+.2f} pts/gen over g{g0}–g{s['gen'][-1]}")
         plateaued = False
   else:
      plateaued = False

   # Value-loss / data-turns correlation (pool-composition artifact)
   r = C.pearson(s["valL"], s["dataTurns"])
   if r < -0.7:
      lines.append(p.warn("⚠ valL artifact ") +
                   f" corr(valLoss, dataTurns) = {r:.2f} — pool-mix effect, not regression")
   elif abs(r) > 0:
      lines.append(p.good("✓ valL clean    ") + f" corr(valLoss, dataTurns) = {r:.2f}")

   # Nontransitive drift: healthy gate winrate while plateaued
   prev = [v for v in s["prev"] if v is not None][-10:]
   if prev and plateaued:
      avg = sum(prev) / len(prev)
      if avg >= 58:
         lines.append(p.warn("⚠ drift suspect ") +
                      f" vs-prev avg {avg:.1f}% while plateaued — gains may be nontransitive")

   # Stale pool: near-instant convergence
   bestEps = [b for b in s["bestEp"] if b is not None]
   if len(bestEps) >= 3 and all(b <= 3 for b in bestEps[-3:]):
      lines.append(p.warn("⚠ stale pool    ") +
                   f" best epoch ≤ 3 for last 3 gens ({bestEps[-3:]}) — window may be stale")
   elif bestEps:
      lines.append(p.good("✓ pool fresh    ") + f" recent best epochs {bestEps[-3:]}")

   # Wall-clock split
   gen = sum(v for v in s["genSecs"] if v)
   train = sum(v for v in s["trainSecs"] if v)
   evals = sum(g[k]["elapsedSeconds"] for g in gens
               for k in ("evalVsRandom", "evalVsHeuristic", "evalVsPrev") if g.get(k))
   total = gen + train + evals
   if total > 0:
      lines.append(p.faint(f"· wall-clock: generate {gen/total*100:.0f}% · "
                           f"train {train/total*100:.0f}% · eval {evals/total*100:.0f}%"))

   for line in lines:
      print(f"  {line}")


# ── Entry point ────────────────────────────────────────────────────────────────

def render (run: dict, palette, genRange=None):
   gens = run.get("generations", [])
   if genRange:
      lo, hi = genRange
      gens = [g for g in gens if lo <= g["generationIndex"] <= hi]
   if not gens:
      print("No generations in range — nothing to show.")
      return
   s = _seriesFrom(gens)
   _header(run, gens, palette)
   _trajectory(s, palette)
   _moveMix(gens, palette)
   _generationsTable(gens, s, palette)
   _checks(gens, s, palette)
