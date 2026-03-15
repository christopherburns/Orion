#!/usr/bin/env python3
"""Iterative self-play training loop for Orion.

Usage:
   iterative_gameplay.py [options]

Options:
   --initial-games N       Games to generate in the first cycle                    [default: 1000]
   --games-per-cycle N     Games to generate in subsequent cycles                  [default: 1000]
   --epochs N              Training epochs per cycle                               [default: 100]
   --batch-size N          Training batch size                                     [default: 128]
   --cycles N              Total number of cycles to run                           [default: 10]
   --eval-games N          Games to play when evaluating                           [default: 500]
   --early-stopping N      Stop training after N epochs w/out improvement (0=off)  [default: 10]
   --initial-temp TEMP     Sampling temperature for cycle 1                        [default: 1.5]
   --final-temp TEMP       Sampling temperature for the last cycle                 [default: 0.5]
   --learning-rate R       Learning rate                                           [default: 0.0003]
   --weight-decay N        Weight decay rate                                       [default: 0.01]
   --eval-temp TEMP        Sampling temperature during evaluation (0=greedy)       [default: 0]
   --accumulate-data       Train on all previous cycles' data, not just the latest
   --data-dir DIR          Directory for generated training data                   [default: trainingdata]
   --model-dir DIR         Directory for saved models                              [default: models]
   --eval-dir DIR          Directory for evaluation results                        [default: evaluations]
   --binary PATH           Path to the orion binary                                [default: .build/release/orion]
   -h --help               Show this help message
"""

import os
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
   initialGames:    int   = 1000
   gamesPerCycle:   int   = 50
   epochs:          int   = 100
   batchSize:       int   = 128
   maxCycles:       int   = 10
   evalGames:       int   = 50
   earlyStopping:   int   = 10   # epochs without improvement before stopping (0 = disabled)
   initialTemp:     float = 1.5
   finalTemp:       float = 0.5
   learningRate:    float = 0.0003
   weightDecay:     float = 0.01
   evalTemp:        float = 0
   accumulateData:  bool  = False
   dataDir:         str   = "trainingdata"
   modelDir:        str   = "models"
   evalDir:         str   = "evaluations"
   binary:          str   = ".build/release/orion"


# ── Shell helpers ──────────────────────────────────────────────────────────────

def run (args: list[str], label: str) -> int:
   """Run a subprocess, streaming its output. Returns exit code."""
   print(f"\n$ {' '.join(args)}")
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
      "-g", str(cfg.gamesPerCycle),
      "-a", agent,
      "-t", f"{temperature:.2f}",
   ]
   return run(args, "generate") == 0


def generateInitialData (cfg: Config, outputPath: str) -> bool:
   """Play the first batch of games using the random agent."""
   args = [
      cfg.binary, "generate",
      "-o", outputPath,
      "-g", str(cfg.initialGames),
      "-a", "random",
      "-t", f"{cfg.initialTemp:.2f}",
   ]
   return run(args, "generate-initial") == 0


def trainModel (cfg: Config, inputPath: str, outputPath: str, prevModelPath: str | None = None) -> bool:
   """Train (or fine-tune) a model on inputPath, saving weights to outputPath."""
   args = [
      cfg.binary, "train",
      "-i", inputPath,
      "-e", str(cfg.epochs),
      "-b", str(cfg.batchSize),
      "-o", outputPath,
      "--learning-rate", str(cfg.learningRate),
      "--weight-decay", str(cfg.weightDecay),
      "--early-stopping", str(cfg.earlyStopping),
   ]
   if prevModelPath is not None:
      args += ["-m", prevModelPath]
   return run(args, "train") == 0


def evaluateVsRandom (cfg: Config, modelPath: str, outputFile: str) -> bool:
   """Evaluate modelPath against the random agent, writing results to outputFile."""
   args = [
      cfg.binary, "play",
      "-n", str(cfg.evalGames),
      "-a", modelPath, "random",
      "-t", f"{cfg.evalTemp:.2f}",
   ]
   print(f"\n$ {' '.join(args)}  > {outputFile}")
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
   print(f"\n$ {' '.join(args)}  > {outputFile}")
   with open(outputFile, "w") as f:
      result = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
   print(f"Evaluation results saved to {outputFile}")
   return result.returncode == 0


# ── Temperature schedule ───────────────────────────────────────────────────────

def computeTemperature (cfg: Config, cycle: int) -> float:
   """Linear decay from initialTemp (cycle 1) to finalTemp (cycle maxCycles)."""
   if cfg.maxCycles <= 1:
      return cfg.finalTemp
   progress = (cycle - 1) / (cfg.maxCycles - 1)
   return cfg.initialTemp - (cfg.initialTemp - cfg.finalTemp) * progress


# ── Path helpers ───────────────────────────────────────────────────────────────

def modelPath (cfg: Config, cycle: int) -> str:
   return f"{cfg.modelDir}/model_c{cycle}_e{cfg.epochs}_b{cfg.batchSize}"

def dataPath (cfg: Config, cycle: int) -> str:
   if cycle == 1:
      return f"{cfg.dataDir}/data_r1_{cfg.initialGames}"
   return f"{cfg.dataDir}/data_c{cycle - 1}_{cfg.gamesPerCycle}"

def evalPath (cfg: Config, cycle: int, suffix: str = "") -> str:
   return f"{cfg.evalDir}/eval_cycle{cycle}{suffix}.txt"


# ── Main loop ──────────────────────────────────────────────────────────────────

def runFirstCycle (cfg: Config) -> str:
   """Bootstrap: generate data with random agent, train first model, evaluate."""
   print(f"\n=== Cycle 1: Generating {cfg.initialGames} games with random agent ===")
   data = dataPath(cfg, 1)
   if not generateInitialData(cfg, data):
      sys.exit(1)

   print(f"\n=== Cycle 1: Training model ===")
   model = modelPath(cfg, 1)
   trainingInput = cfg.dataDir if cfg.accumulateData else f"{data}.gz"
   if not trainModel(cfg, trainingInput, model):
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
   print("Training model (continuing from previous cycle)...")
   trainingInput = cfg.dataDir if cfg.accumulateData else f"{data}.gz"
   if not trainModel(cfg, trainingInput, currentModel, prevModelPath=f"{prevModel}/"):
      sys.exit(1)

   print("Evaluating model vs random...")
   evaluateVsRandom(cfg, f"{currentModel}/", evalPath(cfg, cycle))

   if os.path.isdir(f"{prevModel}/") or os.path.isdir(prevModel):
      print("Evaluating model vs previous cycle...")
      evaluateVsPrevious(cfg, f"{currentModel}/", f"{prevModel}/", evalPath(cfg, cycle, "_vs_prev"))

   return currentModel


def configFromArgs (args: dict) -> Config:
   return Config(
      initialGames  = int(args["--initial-games"]),
      gamesPerCycle = int(args["--games-per-cycle"]),
      epochs        = int(args["--epochs"]),
      batchSize     = int(args["--batch-size"]),
      maxCycles     = int(args["--cycles"]),
      evalGames     = int(args["--eval-games"]),
      earlyStopping = int(args["--early-stopping"]),
      initialTemp   = float(args["--initial-temp"]),
      finalTemp     = float(args["--final-temp"]),
      learningRate  = float(args["--learning-rate"]),
      weightDecay   = float(args["--weight-decay"]),
      evalTemp      = float(args["--eval-temp"]),
      accumulateData = bool(args["--accumulate-data"]),
      dataDir       = args["--data-dir"],
      modelDir      = args["--model-dir"],
      evalDir       = args["--eval-dir"],
      binary        = args["--binary"],
   )


def main ():
   cfg = configFromArgs(docopt(__doc__))

   os.makedirs(cfg.dataDir,  exist_ok=True)
   os.makedirs(cfg.modelDir, exist_ok=True)
   os.makedirs(cfg.evalDir,  exist_ok=True)

   currentModel = runFirstCycle(cfg)

   for cycle in range(2, cfg.maxCycles + 1):
      currentModel = runCycle(cfg, cycle, currentModel)

   print(f"\n=== Training complete! Final model: {currentModel} ===")
   print(f"Evaluation results in: {cfg.evalDir}/")


if __name__ == "__main__":
   main()
