#!/usr/bin/env python3
"""Iterative self-play training loop for Orion.

Usage:
   iterative_gameplay.py [options]

Options:
   --initial-games N       Games to generate in the first cycle                    [default: 5000]
   --games-per-cycle N     Games to generate in subsequent cycles                  [default: 5000]
   --epochs N              Training epochs per cycle                               [default: 100]
   --training-batch-size N Training batch size                                     [default: 128]
   --generate-batch-size N Games to run in parallel during MCTS generation         [default: 128]
   --cycles N              Total number of cycles to run                           [default: 15]
   --eval-games N          Games to play when evaluating                           [default: 500]
   --champion-threshold N  Min win rate vs previous to accept new model (0=off)    [default: 0.52]
   --early-stopping N      Stop training after N epochs w/out improvement (0=off)  [default: 10]
   --initial-temp TEMP     Sampling temperature for cycle 1                        [default: 1.5]
   --final-temp TEMP       Sampling temperature for the last cycle                 [default: 0.5]
   --learning-rate R       Learning rate for cycle 1                               [default: 0.0003]
   --lr-decay R            Multiplicative LR decay per cycle (1.0 = no decay)      [default: 0.95]
   --weight-decay N        Weight decay rate                                       [default: 0.0]
   --eval-temp TEMP        Sampling temperature during evaluation (0=greedy)       [default: 0]
   --min-policy-weight N   Min policy weight for losers (0=ignore, 1=equal)        [default: 0.5]
   --dropout N             Dropout rate for trunk layers (0=disabled)              [default: 0.1]
   --monte-carlo-samples N MCTS monteCarloSamples per move (0=disabled)            [default: 25]
   --c-puct N              MCTS exploration constant                               [default: 1.5]
   --accumulate-data       Train on all previous cycles' data, not just the latest
   --data-dir DIR          Directory for generated training data                   [default: trainingdata]
   --model-dir DIR         Directory for saved models                              [default: models]
   --eval-dir DIR          Directory for evaluation results                        [default: evaluations]
   --binary PATH           Path to the orion binary                                [default: .build/release/orion]
   -h --help               Show this help message
"""

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

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

   # 66  ./orion generate -g 1000 -o trainingdata/onehot_data -s 42
   # 67  ./orion train -i trainingdata/onehot_data.gz -o models/onehot_model -e 40 --learning-rate 0.0003 --weight-decay 0.01 --early-stopping 8
   # 68  ./orion play -n 100 -a models/onehot_model random

@dataclass
class Config:
   initialGames:      int
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
   minPolicyWeight:   float
   dropout:           float
   monteCarloSamples: int
   cPuct:             float
   accumulateData:    bool
   dataDir:           str
   modelDir:          str
   evalDir:           str
   binary:            str


# ── Shell helpers ──────────────────────────────────────────────────────────────

_command_log: str | None = None   # set in main()

BOLD_CYAN  = "\033[1;36m"
RESET      = "\033[0m"

def _logCommand (args: list[str], suffix: str = ""):
   """Print command prominently and append to the command log."""
   cmd = " ".join(args) + (f"  {suffix}" if suffix else "")
   print(f"\n{BOLD_CYAN}▶ {cmd}{RESET}")
   if _command_log:
      with open(_command_log, "a") as f:
         f.write(cmd + "\n")

def run (args: list[str], label: str) -> int:
   """Run a subprocess, streaming its output. Returns exit code."""
   _logCommand(args)
   result = subprocess.run(args)
   if result.returncode != 0:
      print(f"[ERROR] '{label}' exited with code {result.returncode}", file=sys.stderr)
   return result.returncode


# ── Step functions ─────────────────────────────────────────────────────────────

def generateData (cfg: Config, outputPath: str, agent: str, temperature: float) -> bool:
   """Play games using the given agent and write training data to outputPath."""
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
   return run(args, "generate") == 0


def generateInitialData (cfg: Config, outputPath: str) -> bool:
   """Play the first batch of games using the random agent.
   MCTS is not used for cycle 1 — the random agent has no value head."""
   args = [
      cfg.binary, "generate",
      "-o", outputPath,
      "-n", str(cfg.initialGames),
      "-a", "random",
      "--monte-carlo-samples", str(cfg.monteCarloSamples),
      "-t", f"{cfg.initialTemp:.2f}",
   ]
   return run(args, "generate-initial") == 0


def trainModel (cfg: Config, inputPath: str, outputPath: str, learningRate: float, prevModelPath: str | None = None, minPolicyWeight: float = 0.0) -> bool:
   """Train (or fine-tune) a model on inputPath, saving weights to outputPath."""
   args = [
      cfg.binary, "train",
      "-i", inputPath,
      "-e", str(cfg.epochs),
      "-b", str(cfg.trainingBatchSize),
      "-o", outputPath,
      "--learning-rate", str(learningRate),
      "--weight-decay", str(cfg.weightDecay),
      "--early-stopping", str(cfg.earlyStopping),
      "--min-policy-weight", str(minPolicyWeight),
      "--dropout", str(cfg.dropout),
   ]
   if prevModelPath is not None:
      args += ["-m", prevModelPath]
   return run(args, "train") == 0


def cycleLearningRate (cfg: Config, cycle: int) -> float:
   """Geometric decay: LR halves (or scales by lrDecay) each cycle."""
   return cfg.learningRate * (cfg.lrDecay ** (cycle - 1))


def evaluateVsRandom (cfg: Config, modelPath: str, outputFile: str) -> bool:
   """Evaluate modelPath against the random agent, writing results to outputFile."""
   args = [
      cfg.binary, "play",
      "-n", str(cfg.evalGames),
      "-a", modelPath, "random",
      "-t", f"{cfg.evalTemp:.2f}",
   ]
   _logCommand(args, f"> {outputFile}")
   with open(outputFile, "w") as f:
      result = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
   print(f"Evaluation results saved to {outputFile}")
   return result.returncode == 0


def evaluateVsPrevious (cfg: Config, modelPath: str, prevModelPath: str, outputFile: str) -> bool:
   """Evaluate modelPath against prevModelPath, writing results to outputFile."""
   args = [
      cfg.binary, "play",
      "-n", str(cfg.evalGames),
      "-a", modelPath, prevModelPath,
      "-t", f"{cfg.evalTemp:.2f}",
   ]
   _logCommand(args, f"> {outputFile}")
   with open(outputFile, "w") as f:
      result = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
   print(f"Evaluation results saved to {outputFile}")
   return result.returncode == 0


# ── Evaluation parsing ─────────────────────────────────────────────────────────

def parseWinRate (evalFile: str, playerIndex: int = 0) -> float | None:
   """Parse win rate for a player from an orion play output file. Returns None if parsing fails."""
   try:
      with open(evalFile) as f:
         for line in f:
            if f"Player {playerIndex}" in line and "won" in line:
               match = re.search(r'\((\d+\.?\d*)%\)', line)
               if match:
                  return float(match.group(1)) / 100.0
   except (FileNotFoundError, ValueError):
      pass
   return None


# ── Temperature schedule ───────────────────────────────────────────────────────

def computeTemperature (cfg: Config, cycle: int) -> float:
   """Linear decay from initialTemp (cycle 1) to finalTemp (cycle maxCycles)."""
   if cfg.maxCycles <= 1:
      return cfg.finalTemp
   progress = (cycle - 1) / (cfg.maxCycles - 1)
   return cfg.initialTemp - (cfg.initialTemp - cfg.finalTemp) * progress


# ── Path helpers ───────────────────────────────────────────────────────────────

def cycleStr (cfg: Config, cycle: int) -> str:
   """Zero-padded cycle number wide enough for cfg.maxCycles."""
   width = len(str(cfg.maxCycles))
   return str(cycle).zfill(width)

def modelPath (cfg: Config, cycle: int) -> str:
   return f"{cfg.modelDir}/model_c{cycleStr(cfg, cycle)}_e{cfg.epochs}_b{cfg.trainingBatchSize}"

def dataPath (cfg: Config, cycle: int) -> str:
   if cycle == 1:
      return f"{cfg.dataDir}/data_r1_{cfg.initialGames}"
   return f"{cfg.dataDir}/data_c{cycleStr(cfg, cycle - 1)}_{cfg.gamesPerCycle}"

def evalPath (cfg: Config, cycle: int, suffix: str = "") -> str:
   return f"{cfg.evalDir}/eval_cycle{cycleStr(cfg, cycle)}{suffix}.txt"


# ── Main loop ──────────────────────────────────────────────────────────────────

def runFirstCycle (cfg: Config) -> str:
   """Bootstrap: generate data with random agent, train first model, evaluate."""
   print(f"\n=== Cycle 1: Generating {cfg.initialGames} games with random agent ===")
   data = dataPath(cfg, 1)
   if not generateInitialData(cfg, data):
      sys.exit(1)

   print(f"\n=== Cycle 1: Training model ===")
   model = modelPath(cfg, 1)
   lr = cycleLearningRate(cfg, 1)
   trainingInput = cfg.dataDir if cfg.accumulateData else f"{data}.bin.lz4"
   if not trainModel(cfg, trainingInput, model, learningRate=lr, minPolicyWeight=0.0):
      sys.exit(1)

   print(f"\n=== Cycle 1: Evaluating model vs random ===")
   evaluateVsRandom(cfg, f"{model}/", evalPath(cfg, 1))

   return model


def runCycle (cfg: Config, cycle: int, prevModel: str) -> str:
   """One self-play cycle: generate → train → evaluate."""
   temp = computeTemperature(cfg, cycle)
   print(f"\n=== Cycle {cycle} (temperature: {temp:.2f}) ===")

   data = dataPath(cfg, cycle)
   print(f"Generating {cfg.gamesPerCycle} games with model from cycle {cycle - 1}...")
   if not generateData(cfg, data, f"{prevModel}/", temp):
      sys.exit(1)

   currentModel = modelPath(cfg, cycle)
   lr = cycleLearningRate(cfg, cycle)
   print(f"Training model (continuing from previous cycle, LR={lr:.6f})...")
   trainingInput = cfg.dataDir if cfg.accumulateData else f"{data}.bin.lz4"
   if not trainModel(cfg, trainingInput, currentModel, learningRate=lr, prevModelPath=f"{prevModel}/", minPolicyWeight=cfg.minPolicyWeight):
      sys.exit(1)

   print("Evaluating model vs random...")
   evaluateVsRandom(cfg, f"{currentModel}/", evalPath(cfg, cycle))

   vsPrevFile = evalPath(cfg, cycle, "_vs_prev")
   if os.path.isdir(f"{prevModel}/") or os.path.isdir(prevModel):
      print("Evaluating model vs previous cycle...")
      evaluateVsPrevious(cfg, f"{currentModel}/", f"{prevModel}/", vsPrevFile)

   # Champion gating: only accept the new model if it clears the win-rate threshold
   if cfg.championThreshold > 0:
      winRate = parseWinRate(vsPrevFile)
      if winRate is None:
         print("Warning: could not parse vs-prev win rate, accepting new model by default")
      elif winRate < cfg.championThreshold:
         print(f"New model win rate {winRate:.1%} < threshold {cfg.championThreshold:.1%} — keeping previous champion")
         return prevModel
      else:
         print(f"New model win rate {winRate:.1%} >= threshold {cfg.championThreshold:.1%} — accepting new champion")

   return currentModel


def configFromArgs (args: dict) -> Config:
   return Config(
      initialGames       = int(args["--initial-games"]),
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
      minPolicyWeight    = float(args["--min-policy-weight"]),
      dropout            = float(args["--dropout"]),
      monteCarloSamples  = int(args["--monte-carlo-samples"]),
      cPuct              = float(args["--c-puct"]),
      accumulateData     = bool(args["--accumulate-data"]),
      dataDir            = args["--data-dir"],
      modelDir           = args["--model-dir"],
      evalDir            = args["--eval-dir"],
      binary             = args["--binary"],
   )


def main ():
   global _command_log
   cfg = configFromArgs(docopt(__doc__))

   os.makedirs(cfg.dataDir,  exist_ok=True)
   os.makedirs(cfg.modelDir, exist_ok=True)
   os.makedirs(cfg.evalDir,  exist_ok=True)

   _command_log = "log.txt"
   with open(_command_log, "w") as f:
      f.write(f"# Orion training run — {__import__('datetime').datetime.now().isoformat()}\n")

   currentModel = runFirstCycle(cfg)

   for cycle in range(2, cfg.maxCycles + 1):
      currentModel = runCycle(cfg, cycle, currentModel)

   print(f"\n=== Training complete! Final model: {currentModel} ===")
   print(f"Evaluation results in: {cfg.evalDir}/")


if __name__ == "__main__":
   main()
