"""Report on a saved model directory (architecture.json / metadata.json / weights.json).

Top-down structure:
   render(path, palette)   — entry point
      _loadModel           parse the three json files, decode weights
      _identitySection
      _layerTable
      _sanitySection
"""

import base64
import json
import math
import os
import struct

from . import common as C

# Canonical layer order for display (matches PolicyValueNetwork structure)
LAYER_ORDER = ["dense1", "dense2", "dense3", "policyHead", "valueHidden", "valueOutput"]


# ── Loading ────────────────────────────────────────────────────────────────────

def _loadModel (path: str) -> dict:
   def readJson (name):
      p = os.path.join(path, name)
      return json.load(open(p)) if os.path.exists(p) else None

   arch = readJson("architecture.json")
   meta = readJson("metadata.json")
   weightsPath = os.path.join(path, "weights.json")
   weightsRaw = json.load(open(weightsPath)) if os.path.exists(weightsPath) else {}

   layers = {}
   for key, info in weightsRaw.items():
      data = base64.b64decode(info["data"])
      floats = struct.unpack(f"<{len(data)//4}f", data)
      layers[key] = {"shape": info["shape"], "values": floats}

   return {
      "arch": arch or {},
      "meta": meta or {},
      "layers": layers,
      "weightsBytes": os.path.getsize(weightsPath) if os.path.exists(weightsPath) else 0,
   }


# ── Sections ───────────────────────────────────────────────────────────────────

def _identitySection (path, model, p):
   arch, meta = model["arch"], model["meta"]
   print(C.sectionRule(f"model · {os.path.basename(os.path.normpath(path))}"))
   print(f"  architecture   v{arch.get('architectureVersion', '?')} · "
         f"input {arch.get('inputDimensions', '?')} · "
         f"policy {arch.get('policyDimensions', '?')} · "
         f"dropout {arch.get('dropoutRate', 0):.2f}")
   created = (meta.get("createdAt") or "?").replace("T", " ").rstrip("Z")
   trained = meta.get("trainingEpochs")
   loss = meta.get("trainingLoss")
   print(f"  created        {created}"
         + (f" · best epoch {trained}" if trained is not None else "")
         + (f" · val loss {loss:.4f}" if loss is not None else ""))
   print(f"  size           weights.json {C.humanBytes(model['weightsBytes'])} (fp32 base64)")


def _layerTable (model, p):
   weights = {k: v for k, v in model["layers"].items() if k.endswith(".weight")}
   biases = {k: v for k, v in model["layers"].items() if k.endswith(".bias")}

   print()
   print(f"  {'layer':<14s} {'shape':>10s} {'params':>10s} {'‖W‖₂':>8s}")
   total = 0
   names = [n for n in LAYER_ORDER if f"{n}.weight" in weights]
   names += sorted(set(k.rsplit(".", 1)[0] for k in weights) - set(names))
   for name in names:
      w = weights[f"{name}.weight"]
      b = biases.get(f"{name}.bias")
      shape = w["shape"]
      params = len(w["values"]) + (len(b["values"]) if b else 0)
      total += params
      norm = math.sqrt(sum(x * x for x in w["values"]))
      shapeStr = "×".join(str(d) for d in shape)
      print(f"  {name:<14s} {shapeStr:>10s} {params:>10,} {norm:>8.2f}")
   print(f"  {'total':<14s} {'':>10s} {total:>10,}")


def _sanitySection (model, p):
   allVals = [x for layer in model["layers"].values() for x in layer["values"]]
   bad = sum(1 for x in allVals if math.isnan(x) or math.isinf(x))
   notes = []
   notes.append(p.good("no NaN/Inf ✓") if bad == 0 else p.bad(f"⚠ {bad} NaN/Inf values"))

   # Dead output units in the first layer: rows whose weights are ~all zero
   d1 = model["layers"].get("dense1.weight")
   if d1:
      rows, cols = d1["shape"]
      vals = d1["values"]
      dead = sum(1 for r in range(rows)
                 if all(abs(vals[r * cols + c]) < 1e-7 for c in range(cols)))
      frac = dead / rows * 100
      msg = f"dead dense1 rows: {dead}/{rows} ({frac:.1f}%)"
      notes.append(p.good(msg + " ✓") if frac < 5 else p.warn("⚠ " + msg))

   print()
   print("  sanity: " + " · ".join(notes))


# ── Entry point ────────────────────────────────────────────────────────────────

def render (path: str, palette):
   model = _loadModel(path)
   if not model["layers"] and not model["arch"]:
      print(f"No model files found in {path}")
      return
   _identitySection(path, model, palette)
   _layerTable(model, palette)
   _sanitySection(model, palette)
