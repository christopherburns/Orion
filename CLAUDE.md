# Orion - Neural Network Splendor AI

## Project Overview

Orion is a Swift-based system for training neural networks to play the card game Splendor. It uses Apple's MLX framework for on-device machine learning and implements a complete Splendor game engine with a neural network agent architecture.

**Current Status:** Full training pipeline operational. Iterative self-play loop with champion gating, hyperparameter sweeps, interactive human-vs-AI play mode. Active work on improving training convergence.

## Project Structure

```
Orion/
├── src/
│   ├── core/
│   │   ├── GameProtocol.swift         # Generic game interface + GameTerminalCondition enum
│   │   ├── AgentProtocol.swift        # Generic agent interface (isHuman property)
│   │   └── RandomAgent.swift          # Random baseline agent (DumbAgent)
│   │
│   ├── splendor/
│   │   ├── Game.swift                 # Complete Splendor game logic
│   │   ├── Card.swift                 # Card definitions and gem types
│   │   ├── GamePrinter.swift          # Console output, interactive UI, probability bars
│   │   ├── SplendorNeuralAgent.swift  # PolicyValueNetwork + SplendorNeuralAgent
│   │   └── SplendorTrainingData.swift # Training data structs + binary serialization
│   │
│   ├── utility/
│   │   ├── Utility.swift              # PRNG and misc utilities
│   │   └── OptionParser.swift         # CLI argument parsing
│   │
│   ├── Orion.swift                    # Main entry point (dispatches to subcommands)
│   ├── Common.swift                   # Shared helpers (initializeAgents, sampleMoveWithTemperature)
│   ├── DataGenerator.swift            # `orion generate` — self-play data collection
│   ├── NetworkTrainer.swift           # `orion train` — training loop with early stopping
│   ├── GameplayTester.swift           # `orion play` — evaluation and interactive play
│   └── HumanAgent.swift              # Human player agent for interactive mode
│
├── scripts/
│   ├── iterative_gameplay.py          # Multi-cycle self-play training orchestrator
│   └── sweep.sh                       # Hyperparameter sweep harness
│
├── Package.swift
├── build.sh                           # xcodebuild wrapper (required for Metal shaders)
└── CLAUDE.md
```

## CLI Tools

Orion provides three subcommands:

### `orion generate` — Generate training data
Plays self-play games and saves training examples.
- `-n` game count, `-a` agent (path or "random"), `-o` output path
- `-t` temperature, `-s` seed, `-p` player count, `--max-turns`
- Output: `.bin.lz4` binary format (LZ4-compressed packed floats)

### `orion train` — Train neural network
Trains on generated data with Adam optimizer.
- `-i` input data (file or directory), `-o` output model path
- `-m` existing model to continue from, `-e` epochs, `-b` batch size
- `-r` learning rate, `-w` weight decay, `-d` dropout rate
- `-E` early stopping patience, `--min-policy-weight` (loser weight floor)
- Policy loss: cross-entropy, value-weighted with configurable min weight
- Value loss: MSE on [-1, 1] targets

### `orion play` — Play and evaluate
Plays games for evaluation or interactive human play.
- `-n` game count, `-a` agent specs (one per player or broadcast)
- `-t` temperature, `--max-turns`, `-v` verbose, `--show-probabilities`
- Interactive mode: `orion play -a human models/model/` (colored probability bars, card descriptions)

## Neural Network Architecture

**PolicyValueNetwork** (`SplendorNeuralAgent.swift`):
- Input: **357** floats (encoded game state)
- Shared trunk: 357 → 512 → 512 → 512 (ReLU + dropout, configurable rate, default 0.1)
- Policy head: 512 → 48 (raw logits, no activation)
- Value head: 512 → 128 → 1 (ReLU, then tanh for [-1, 1])
- He initialization, architecture version **4**
- ~622k parameters
- Dropout rate stored in architecture.json, configurable via `--dropout`

## Game State Encoding (357 Float16 values)

```
Game State Encoding (357 Float16 values)
═══════════════════════════════════════════════════════════════════════════

 Index    Field                          Size   Normalization
───────────────────────────────────────────────────────────────────────────
 0-46     Current player state           47     (see player encoding)
 47-93    Next player state              47     (see player encoding)
 94-140   Player +2 state (or zeros)     47     zero-padded if <3 players
 141-187  Player +3 state (or zeros)     47     zero-padded if <4 players

          Players rotated so slot 0 is always the player to move next.
          No explicit current-player encoding needed.
───────────────────────────────────────────────────────────────────────────
 188      supply[red]                    ┐
 189      supply[green]                  │  /6
 190      supply[blue]                   │
 191      supply[white]                  │
 192      supply[brown]                  ┘
 193      goldGemSupply                  /5
───────────────────────────────────────────────────────────────────────────
          ┌─ Noble 0 (or zeros) ───────────────────────────────────┐
 194      │  points                                          /3    │
 195-199  │  price [red, green, blue, white, brown]          /4    │
          └────────────────────────────────────────────────────────┘
          ┌─ Noble 1 (or zeros) ───────────────────────────────────┐
 200      │  points                                          /3    │
 201-205  │  price [red, green, blue, white, brown]          /4    │
          └────────────────────────────────────────────────────────┘
          ... Nobles 2-4 follow same pattern (6 values each) ...
───────────────────────────────────────────────────────────────────────────
 224-355  Visible cards: 3 tiers × 4 positions × 11         132
          (each card: 1 point/10 + 5 price/10 + 5 color one-hot)
          Zero-padded for empty positions.
───────────────────────────────────────────────────────────────────────────
 356      turnNumber                     /100
───────────────────────────────────────────────────────────────────────────
```

```
Player Encoding (47 Float16 values)
═══════════════════════════════════════════════════════════════════════════

 Index  Field              Normalization
─────────────────────────────────────────────────────────────────────────
  0     gems[red]          ┐
  1     gems[green]        │  /10
  2     gems[blue]         │
  3     gems[white]        │
  4     gems[brown]        ┘
  5     goldGems           /10
─────────────────────────────────────────────────────────────────────────
  6     cardPower[red]     ┐
  7     cardPower[green]   │  /7
  8     cardPower[blue]    │
  9     cardPower[white]   │
 10     cardPower[brown]   ┘
─────────────────────────────────────────────────────────────────────────
 11     reservedCount      /3
─────────────────────────────────────────────────────────────────────────
        ┌─ Reserved Card 0 (or zeros) ─────────────────────────────┐
 12     │  points                                            /10   │
 13-17  │  price [red, green, blue, white, brown]            /10   │
 18-22  │  color one-hot [red, green, blue, white, brown]          │
        └──────────────────────────────────────────────────────────┘
        ┌─ Reserved Card 1 (or zeros) ─────────────────────────────┐
 23     │  points                                            /10   │
 24-28  │  price [red, green, blue, white, brown]            /10   │
 29-33  │  color one-hot [red, green, blue, white, brown]          │
        └──────────────────────────────────────────────────────────┘
        ┌─ Reserved Card 2 (or zeros) ─────────────────────────────┐
 34     │  points                                            /10   │
 35-39  │  price [red, green, blue, white, brown]            /10   │
 40-44  │  color one-hot [red, green, blue, white, brown]          │
        └──────────────────────────────────────────────────────────┘
─────────────────────────────────────────────────────────────────────────
 45     nobleCount         /5
 46     score              /15
─────────────────────────────────────────────────────────────────────────
```

## Training Pipeline

### Iterative Self-Play (`scripts/iterative_gameplay.py`)
Orchestrates the generate → train → evaluate cycle:
- Cycle 1: Random agent generates initial data
- Cycles 2+: Current champion generates self-play data
- Champion gating: new model must beat previous by >52% to be accepted
- Temperature schedule: linear decay from initial (1.5) to final (0.5)
- LR decay: geometric per cycle (default 0.95×)
- Commands logged to `evaluations/commands.log`

Current defaults: 5000 initial games, 5000 games/cycle, 15 cycles, BS=128, LR=3e-4, WD=0.0, dropout=0.1, min-policy-weight=0.5 for self-play cycles (0.0 for cycle 1)

### Training Data Format
Binary `.bin.lz4` files (LZ4-compressed):
- 24-byte header: magic "ORIN", version, dimensions, counts
- Packed examples: [357×f32 state][48×f32 policy][1×f32 value] = 1624 bytes each
- Legacy `.gz` JSON format still loadable

### Loss Function
- **Policy loss**: Cross-entropy with value weighting. Winner weight=1.0, loser weight=`minPolicyWeight` (0.0 for random data, 0.5 for self-play). Without this, self-play training causes catastrophic forgetting.
- **Value loss**: MSE between predicted and actual outcome (±1 for win/loss, 0 for tie)
- Combined: `policyWeight * policyLoss + valueWeight * valueLoss`

**Note:** When minPolicyWeight=0.0, the reported loss is artificially halved because loser examples contribute 0 to the mean. A reported loss of ~1.3 actually represents ~2.6 cross-entropy on winner examples.

## Splendor Game Rules

- **48 canonical moves**: 12 purchase + 3 purchase-reserved + 10 take-three-gems + 5 take-two-gems + 12 reserve + 6 discard
- **Two-phase turns**: Normal action → Discard phase (if >10 gems) → Next player
- **Terminal conditions**: playerWon (15+ points), tied, timedOut, inProgress
- **Gold gems**: Wildcards for purchasing; correctly deducted from player inventory when spent
- 2-4 players, 5 gem types + gold, 90 cards across 3 tiers, nobles

## Building

**Important**: MLX requires Metal shader compilation. Use xcodebuild via `build.sh`:

```bash
./build.sh release    # Required for Metal shaders
./build.sh debug      # Debug with better error messages
```

`swift build` compiles but the binary will crash at runtime with "Failed to load metallib" unless a metallib from a prior xcodebuild is present.

## Dependencies

- **mlx-swift** (v0.10.0+): MLX, MLXNN, MLXOptimizers
- **Swift 5.9+**, **macOS 14+**
- **Python 3** with `docopt` (for iterative_gameplay.py)
