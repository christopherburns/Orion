#!/usr/bin/env python3
"""Iterative self-play training loop for Orion.

Starts from an existing bootstrap model and runs N self-play cycles. Each cycle
generates fresh data using the current champion, trains a new model from the
prior cycle's weights, and gates acceptance on a win-rate threshold vs. the
champion. Run scripts/bootstrap.sh first to produce an initial model.

All outputs for this run land under RUN_DIR (created if missing): generated
training data under RUN_DIR/trainingdata/, saved models under RUN_DIR/models/,
per-call reports as RUN_DIR/cycle_NN_*.json, and the master aggregate as
RUN_DIR/run.json. The master file is rewritten atomically after every cycle.

Usage:
   iterative_gameplay.py RUN_DIR [options]

Options:
   --initial-model PATH    Initial model to start self-play from                 [default: none]
   --games-per-cycle N     Games to generate per cycle                           [default: 5000]
   --epochs N              Training epochs per cycle                             [default: 100]
   --training-batch-size N Training batch size                                   [default: 128]
   --generate-batch-size N Games to run in parallel during MCTS generation       [default: 128]
   --cycles N              Total number of cycles to run                         [default: 15]
   --eval-games N          Games to play when evaluating                         [default: 500]
   --champion-threshold N  Min win rate vs previous to accept new model (0=off)  [default: 0.52]
   --early-stopping N      Stop training after N epochs w/out improvement (0=off)[default: 10]
   --initial-temp TEMP     Sampling temperature for cycle 1                      [default: 1.5]
   --final-temp TEMP       Sampling temperature for the last cycle               [default: 0.5]
   --learning-rate R       Learning rate for cycle 1                             [default: 0.0003]
   --lr-decay R            Multiplicative LR decay per cycle (1.0 = no decay)    [default: 0.95]
   --weight-decay N        Weight decay rate                                     [default: 0.0]
   --eval-temp TEMP        Sampling temperature during evaluation (0=greedy)     [default: 0.1]
   --dropout N             Dropout rate for trunk layers (0=disabled)            [default: 0.1]
   --monte-carlo-samples N MCTS monteCarloSamples per move (0=disabled)          [default: 25]
   --c-puct N              MCTS exploration constant                             [default: 1.5]
   --accumulate-data       Train on all previous cycles' data, not just the latest
   --binary PATH           Path to the orion binary                              [default: .build/release/orion]
   -h --help               Show this help message
"""

import datetime
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Optional

from docopt import docopt


# ── Terminology ───────────────────────────────────────────────────────────────
#
#  Step  — one forward+backward pass through a single batch of training examples.
#  Epoch — one full pass through the current training dataset (many steps).
#  Cycle — one full generate → train → evaluate round orchestrated by this script
#          (many epochs). The model improves and the next cycle's data is generated
#          by the updated model.
#
# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
   runDir:            str
   initialModel:      str
   gamesPerCycle:     int
   epochs:            int
   trainingBatchSize: int
   generateBatchSize: int
   maxCycles:         int
   evalGames:         int
   championThreshold: float
   earlyStopping:     int
   initialTemp:       float
   finalTemp:         float
   learningRate:      float
   lrDecay:           float
   weightDecay:       float
   evalTemp:          float
   dropout:           float
   monteCarloSamples: int
   cPuct:             float
   accumulateData:    bool
   binary:            str


# ── Shell helpers ──────────────────────────────────────────────────────────────

_command_log: Optional[str] = None   # set in main()

BOLD_CYAN  = "\033[1;36m"
RESET      = "\033[0m"

def _logCommand (args: list[str], suffix: str = ""):
   """Print command prominently and append to the command log."""
   cmd = " ".join(args) + (f"  {suffix}" if suffix else "")
   print(f"\n{BOLD_CYAN}▶ {cmd}{RESET}")
   if _command_log:
      with open(_command_log, "a") as f:
         f.write(cmd + "\n")


def writeJsonAtomic (path: str, obj) -> None:
   """Atomic JSON write: temp file + rename."""
   os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
   tmp = path + ".tmp"
   with open(tmp, "w") as f:
      json.dump(obj, f, indent=2, sort_keys=True)
   os.replace(tmp, path)


def readJson (path: str) -> dict:
   with open(path) as f:
      return json.load(f)


def runOrion (args: list[str], label: str, reportPath: str) -> Optional[dict]:
   """Run an orion subcommand with --report-json and return the parsed report dict.
   Returns None if the subprocess failed or didn't produce the report file."""
   fullArgs = args + ["--report-json", reportPath]
   _logCommand(fullArgs)
   result = subprocess.run(fullArgs)
   if result.returncode != 0:
      print(f"[ERROR] '{label}' exited with code {result.returncode}", file=sys.stderr)
      return None
   if not os.path.exists(reportPath):
      print(f"[ERROR] '{label}' did not produce report at {reportPath}", file=sys.stderr)
      return None
   return readJson(reportPath)


# ── Step functions (each returns the structured report dict, or None on failure) ─

def generateData (cfg: Config, outputPath: str, agent: str, temperature: float, reportPath: str) -> Optional[dict]:
   args = [
      cfg.binary, "generate",
      "-o", outputPath,
      "-n", str(cfg.gamesPerCycle),
      "-a", agent,
      "-t", f"{temperature:.2f}",
   ]
   if cfg.monteCarloSamples > 0:
      args += ["--monte-carlo-samples", str(cfg.monteCarloSamples), "--c-puct", str(cfg.cPuct),
               "-b", str(cfg.generateBatchSize)]
   return runOrion(args, "generate", reportPath)


def trainModel (cfg: Config, inputPath: str, outputPath: str, learningRate: float,
                prevModelPath: Optional[str], reportPath: str) -> Optional[dict]:
   args = [
      cfg.binary, "train",
      "-i", inputPath,
      "-e", str(cfg.epochs),
      "-b", str(cfg.trainingBatchSize),
      "-o", outputPath,
      "--learning-rate", str(learningRate),
      "--weight-decay", str(cfg.weightDecay),
      "--early-stopping", str(cfg.earlyStopping),
      "--dropout", str(cfg.dropout),
   ]
   if prevModelPath is not None:
      args += ["-m", prevModelPath]
   return runOrion(args, "train", reportPath)


def evaluatePlay (cfg: Config, agentSpecs: list[str], reportPath: str) -> Optional[dict]:
   args = [
      cfg.binary, "play",
      "-n", str(cfg.evalGames),
      "-a", *agentSpecs,
      "-t", f"{cfg.evalTemp:.2f}",
   ]
   return runOrion(args, "play", reportPath)


def cycleLearningRate (cfg: Config, cycle: int) -> float:
   return cfg.learningRate * (cfg.lrDecay ** (cycle - 1))


def computeTemperature (cfg: Config, cycle: int) -> float:
   if cfg.maxCycles <= 1:
      return cfg.finalTemp
   progress = (cycle - 1) / (cfg.maxCycles - 1)
   return cfg.initialTemp - (cfg.initialTemp - cfg.finalTemp) * progress


# ── Path helpers ───────────────────────────────────────────────────────────────

def cycleStr (cfg: Config, cycle: int) -> str:
   width = len(str(cfg.maxCycles))
   return str(cycle).zfill(width)

def modelPath (cfg: Config, cycle: int) -> str:
   return f"{cfg.runDir}/models/model_c{cycleStr(cfg, cycle)}_e{cfg.epochs}_b{cfg.trainingBatchSize}"

def dataPath (cfg: Config, cycle: int) -> str:
   """Path for data generated by the model at the START of `cycle` (the previous
   champion). Cycle 1's data is generated by the bootstrap model and labelled c00."""
   return f"{cfg.runDir}/trainingdata/data_c{cycleStr(cfg, cycle - 1)}_{cfg.gamesPerCycle}"

def reportFile (cfg: Config, cycle: int, kind: str) -> str:
   return f"{cfg.runDir}/cycle_{cycleStr(cfg, cycle)}_{kind}.json"

def masterFile (cfg: Config) -> str:
   return f"{cfg.runDir}/run.json"


# ── Main loop ──────────────────────────────────────────────────────────────────

def runCycle (cfg: Config, cycle: int, prevModel: str) -> tuple[str, dict]:
   """One self-play cycle. Returns (next champion path, cycle entry dict for the master record)."""
   temp = computeTemperature(cfg, cycle)
   lr = cycleLearningRate(cfg, cycle)
   print(f"\n=== Cycle {cycle} (temperature: {temp:.2f}, LR: {lr:.6f}) ===")

   data = dataPath(cfg, cycle)
   currentModel = modelPath(cfg, cycle)

   print(f"Generating {cfg.gamesPerCycle} games with {prevModel}...")
   generateReport = generateData(cfg, data, f"{prevModel}/", temp, reportFile(cfg, cycle, "generate"))
   if generateReport is None:
      sys.exit(1)

   print(f"Training model (continuing from {prevModel})...")
   trainingInput = f"{cfg.runDir}/trainingdata" if cfg.accumulateData else f"{data}.bin.lz4"
   trainReport = trainModel(cfg, trainingInput, currentModel, lr,
                            prevModelPath=f"{prevModel}/",
                            reportPath=reportFile(cfg, cycle, "train"))
   if trainReport is None:
      sys.exit(1)

   print("Evaluating model vs random...")
   evalRandomReport = evaluatePlay(cfg, [f"{currentModel}/", "random"],
                                   reportFile(cfg, cycle, "eval_vs_random"))
   if evalRandomReport is None:
      sys.exit(1)

   print(f"Evaluating model vs {prevModel}...")
   evalPrevReport = evaluatePlay(cfg, [f"{currentModel}/", f"{prevModel}/"],
                                 reportFile(cfg, cycle, "eval_vs_prev"))
   if evalPrevReport is None:
      sys.exit(1)

   # Champion gating: read win rate directly from the structured report.
   winRateVsPrev = evalPrevReport["results"]["perPlayer"][0]["winRateOverDecisive"]
   accepted = True
   if cfg.championThreshold > 0:
      if winRateVsPrev < cfg.championThreshold:
         print(f"New model win rate {winRateVsPrev:.1%} < threshold {cfg.championThreshold:.1%} — keeping previous champion")
         accepted = False
      else:
         print(f"New model win rate {winRateVsPrev:.1%} >= threshold {cfg.championThreshold:.1%} — accepting new champion")

   cycleEntry = {
      "cycleIndex":        cycle,
      "previousModel":     prevModel,
      "trainedModel":      currentModel,
      "temperature":       temp,
      "learningRate":      lr,
      "winRateVsPrev":     winRateVsPrev,
      "championAccepted":  accepted,
      "generate":          generateReport,
      "train":             trainReport,
      "evalVsRandom":      evalRandomReport,
      "evalVsPrev":        evalPrevReport,
   }

   nextChampion = currentModel if accepted else prevModel
   return nextChampion, cycleEntry


def configFromArgs (args: dict) -> Config:
   return Config(
      runDir             = args["RUN_DIR"],
      initialModel       = args["--initial-model"],
      gamesPerCycle      = int(args["--games-per-cycle"]),
      epochs             = int(args["--epochs"]),
      trainingBatchSize  = int(args["--training-batch-size"]),
      generateBatchSize  = int(args["--generate-batch-size"]),
      maxCycles          = int(args["--cycles"]),
      evalGames          = int(args["--eval-games"]),
      championThreshold  = float(args["--champion-threshold"]),
      earlyStopping      = int(args["--early-stopping"]),
      initialTemp        = float(args["--initial-temp"]),
      finalTemp          = float(args["--final-temp"]),
      learningRate       = float(args["--learning-rate"]),
      lrDecay            = float(args["--lr-decay"]),
      weightDecay        = float(args["--weight-decay"]),
      evalTemp           = float(args["--eval-temp"]),
      dropout            = float(args["--dropout"]),
      monteCarloSamples  = int(args["--monte-carlo-samples"]),
      cPuct              = float(args["--c-puct"]),
      accumulateData     = bool(args["--accumulate-data"]),
      binary             = args["--binary"],
   )


def main ():
   global _command_log
   cfg = configFromArgs(docopt(__doc__))

   if not os.path.isdir(cfg.initialModel):
      print(f"Error: initial model not found at {cfg.initialModel}", file=sys.stderr)
      print(f"Run scripts/bootstrap.sh first, or pass --initial-model PATH.", file=sys.stderr)
      sys.exit(1)

   os.makedirs(cfg.runDir,                            exist_ok=True)
   os.makedirs(os.path.join(cfg.runDir, "trainingdata"), exist_ok=True)
   os.makedirs(os.path.join(cfg.runDir, "models"),       exist_ok=True)

   _command_log = f"{cfg.runDir}/commands.log"
   with open(_command_log, "w") as f:
      f.write(f"# Orion training run — {datetime.datetime.now().isoformat()}\n")

   master = {
      "schemaVersion":  1,
      "type":           "iterativeRun",
      "runDir":         cfg.runDir,
      "startedAt":      datetime.datetime.now().isoformat(timespec="seconds"),
      "completedAt":    None,
      "config":         asdict(cfg),
      "cycles":         [],
   }
   writeJsonAtomic(masterFile(cfg), master)

   currentModel = cfg.initialModel
   for cycle in range(1, cfg.maxCycles + 1):
      currentModel, cycleEntry = runCycle(cfg, cycle, currentModel)
      master["cycles"].append(cycleEntry)
      master["completedAt"] = datetime.datetime.now().isoformat(timespec="seconds")
      writeJsonAtomic(masterFile(cfg), master)

   print(f"\n=== Training complete! Final model: {currentModel} ===")
   print(f"Run report: {masterFile(cfg)}")


if __name__ == "__main__":
   main()
