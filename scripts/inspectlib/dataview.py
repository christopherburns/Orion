"""Report on an ORIN training-data file (.bin.lz4 compressed or .bin raw).

Top-down structure:
   render(path, palette, sampleLimit)   — entry point
      _loadExamples     header parse + (partial) decompression + unpack
      _outcomeSection
      _policySection
      _stateSection

The binary stores packed (state, π, value) examples — it does NOT store the
moves actually played or game boundaries. All move-distribution numbers here
are therefore derived from the stored MCTS policy targets π: "π mass" is the
expected share of probability each move category receives; "argmax π" is what
a greedy player following the targets would pick.
"""

import math
import os
import struct

from . import common as C
from . import applelz4

MAGIC = 0x4F52494E  # "ORIN"
HEADER_BYTES = 24

MOVE_CATEGORIES = [  # canonical move index ranges → category label
   ("buy tier1", 0, 4), ("buy tier2", 4, 8), ("buy tier3", 8, 12),
   ("buy rsrvd", 12, 15), ("take three", 15, 25), ("take two", 25, 30),
   ("reserve", 30, 42), ("discard", 42, 48),
]


# ── Loading ────────────────────────────────────────────────────────────────────

def _loadExamples (path: str, sampleLimit: int):
   """Returns (header dict, list of (stateSlice-accessor, policy list, value), sampledCount).
   Decompresses only as much as the sample needs."""
   raw = open(path, "rb").read()
   compressedSize = len(raw)

   if path.endswith(".lz4"):
      # Decode enough blocks for the header first, then for the sample.
      head = applelz4.decompress(raw, maxBytes=HEADER_BYTES)
      header = _parseHeader(head[:HEADER_BYTES])
      bytesPerExample = (header["stateDim"] + header["policyDim"] + 1) * 4
      wanted = header["examples"] if sampleLimit == 0 else min(sampleLimit, header["examples"])
      body = applelz4.decompress(raw, maxBytes=HEADER_BYTES + wanted * bytesPerExample)
   else:
      body = raw
      header = _parseHeader(body[:HEADER_BYTES])
      bytesPerExample = (header["stateDim"] + header["policyDim"] + 1) * 4
      wanted = header["examples"] if sampleLimit == 0 else min(sampleLimit, header["examples"])

   header["compressedSize"] = compressedSize
   header["bytesPerExample"] = bytesPerExample
   header["uncompressedSize"] = HEADER_BYTES + header["examples"] * bytesPerExample

   available = (len(body) - HEADER_BYTES) // bytesPerExample
   count = min(wanted, available)
   floatsPerExample = header["stateDim"] + header["policyDim"] + 1
   examples = []
   for k in range(count):
      off = HEADER_BYTES + k * bytesPerExample
      vals = struct.unpack_from(f"<{floatsPerExample}f", body, off)
      state = vals[: header["stateDim"]]
      policy = vals[header["stateDim"] : header["stateDim"] + header["policyDim"]]
      value = vals[-1]
      examples.append((state, policy, value))
   return header, examples


def _parseHeader (buf: bytes) -> dict:
   magic, version, stateDim, policyDim, games, examples = struct.unpack("<6I", buf)
   if magic != MAGIC:
      raise ValueError(f"Bad magic 0x{magic:08X} (expected ORIN 0x{MAGIC:08X}) — not an ORIN data file")
   return {"version": version, "stateDim": stateDim, "policyDim": policyDim,
           "games": games, "examples": examples}


# ── Sections ───────────────────────────────────────────────────────────────────

def _headerSection (path, header, sampled, p):
   print(C.sectionRule(f"dataset · {os.path.basename(path)}"))
   print(f"  format    ORIN v{header['version']} · state {header['stateDim']} · "
         f"policy {header['policyDim']} · {header['games']:,} games · "
         f"{header['examples']:,} examples · {header['bytesPerExample']} B/ex")
   print(f"  size      {C.humanBytes(header['uncompressedSize'])} packed → "
         f"{C.humanBytes(header['compressedSize'])} on disk "
         f"({header['uncompressedSize'] / max(1, header['compressedSize']):.1f}×)")
   if sampled < header["examples"]:
      print(f"  sampled   first {sampled:,} of {header['examples']:,} examples "
            f"(--sample 0 for all; pure-python lz4 is slow)")


def _outcomeSection (examples, p):
   # Sign-based bucketing: with length-discounted targets (value-discount < 1),
   # magnitudes shrink below 1 but the sign still encodes win/loss.
   win = sum(1 for _, _, v in examples if v > 0.01)
   loss = sum(1 for _, _, v in examples if v < -0.01)
   tie = len(examples) - win - loss
   n = len(examples)
   mean = sum(v for _, _, v in examples) / n
   meanAbs = sum(abs(v) for _, _, v in examples if abs(v) > 0.01)
   meanAbs = meanAbs / max(1, win + loss)
   discounted = meanAbs < 0.999
   print()
   print(C.sectionRule("outcomes (value targets)"))
   print(f"  win  (+)  {win:>9,}  ({win/n*100:4.1f}%)     mean value {mean:+.4f}"
         + ("  (balanced ✓)" if abs(mean) < 0.05 * meanAbs + 0.01 else p.warn("  ⚠ imbalanced")))
   print(f"  loss (−)  {loss:>9,}  ({loss/n*100:4.1f}%)     mean |value| {meanAbs:.4f}"
         + ("  (length-discounted targets)" if discounted else "  (raw ±1 targets)"))
   print(f"  tie   0   {tie:>9,}  ({tie/n*100:4.1f}%)")


def _policySection (examples, policyDim, p):
   n = len(examples)
   catMass = [0.0] * len(MOVE_CATEGORIES)
   catArgmax = [0] * len(MOVE_CATEGORIES)
   entropies = []
   sharp = 0
   for _, pi, _ in examples:
      h = -sum(x * math.log(x) for x in pi if x > 1e-12)
      entropies.append(h)
      if max(pi) > 0.9:
         sharp += 1
      am = max(range(policyDim), key=lambda i: pi[i])
      for ci, (_, lo, hi) in enumerate(MOVE_CATEGORIES):
         catMass[ci] += sum(pi[lo:hi])
         if lo <= am < hi:
            catArgmax[ci] += 1

   meanH = sum(entropies) / n
   maxH = math.log(policyDim)
   print()
   print(C.sectionRule("policy targets (π, MCTS visit distributions)"))
   print(f"  mean entropy   {meanH:.2f} nats  (uniform over {policyDim} = {maxH:.2f}; "
         f"~{math.exp(meanH):.1f} effective moves)")
   print(f"  sharp (π>0.9)  {sharp/n*100:.1f}% of examples")
   print()
   print(f"  {'':12s}  {'π mass':>7s}             {'argmax π':>8s}")
   maxFrac = max(max(catMass) / n, max(catArgmax) / n, 1e-9)
   for ci, (label, _, _) in enumerate(MOVE_CATEGORIES):
      massFrac = catMass[ci] / n
      amFrac = catArgmax[ci] / n
      print(f"  {label:12s} {massFrac*100:6.1f}%  {C.bar(massFrac/maxFrac, 12):<12s} "
            f"{amFrac*100:7.1f}%  {C.bar(amFrac/maxFrac, 12)}")


def _stateSection (examples, stateDim, p):
   if stateDim < 496:
      return  # unknown encoding layout, skip feature spot-checks
   turnFeats = [st[495] for st, _, _ in examples]
   turnFeats.sort()
   n = len(turnFeats)

   def decodeTurn (f):  # inverse of tanh(min(1, t/100)), clamped
      f = min(f, 0.9999)
      t = 0.5 * math.log((1 + f) / (1 - f)) * 100  # atanh × 100
      return min(t, 100)

   # Affordability flags: visible cards at 351..494, 12 floats each, flag last
   affordable = [sum(1 for k in range(12) if st[351 + 12 * k + 11] > 0.5)
                 for st, _, _ in examples]
   print()
   print(C.sectionRule("state features (spot checks)"))
   print(f"  turn feature (495): min {turnFeats[0]:.2f} · median {turnFeats[n//2]:.2f} · "
         f"max {turnFeats[-1]:.2f}   (≈ turns {decodeTurn(turnFeats[0]):.0f}–{decodeTurn(turnFeats[-1]):.0f})")
   print(f"  affordable visible cards/state: mean {sum(affordable)/n:.1f} of 12")


# ── Entry point ────────────────────────────────────────────────────────────────

DEFAULT_SAMPLE = 20000


def render (path: str, palette, sampleLimit: int = DEFAULT_SAMPLE):
   header, examples = _loadExamples(path, sampleLimit)
   if not examples:
      print("No examples decoded.")
      return
   _headerSection(path, header, len(examples), palette)
   _outcomeSection(examples, palette)
   _policySection(examples, header["policyDim"], palette)
   _stateSection(examples, header["stateDim"], palette)
