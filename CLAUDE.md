# Orion - Neural Network Splendor AI

## Project Overview

Orion is a Swift-based system for training neural networks to play the card game Splendor. It uses Apple's MLX framework for on-device machine learning and implements a complete Splendor game engine, an AlphaZero-style MCTS, and a neural network agent architecture.

**Current Status:** Full training pipeline operational. Iterative self-play loop with champion gating, virtual-loss batched MCTS for fast data generation, determinized MCTS for hidden-information-fair evaluation, example-targeted data generation, sliding-window data accumulation, and interactive human-vs-AI play. Active work on improving training convergence and play strength.

## Project Structure

```
Orion/
├── src/
│   ├── core/
│   │   ├── GameProtocol.swift         # Generic game interface + GameTerminalCondition enum
│   │   ├── AgentProtocol.swift        # Generic agent interface (predict, batchPredict, isHuman)
│   │   └── RandomAgent.swift          # Random baseline agent (DumbAgent)
│   │
│   ├── splendor/
│   │   ├── Game.swift                 # Splendor game logic + state encoding + determinized(seed:)
│   │   ├── Card.swift                 # Card definitions and gem types
│   │   ├── MCTS.swift                 # AlphaZero PUCT search: virtual-loss batching, aggregation
│   │   ├── GamePrinter.swift          # Console output, interactive UI, probability bars
│   │   ├── SplendorNeuralAgent.swift  # PolicyValueNetwork + SplendorNeuralAgent
│   │   ├── SplendorHeuristicAgent.swift # Hand-coded heuristic agent (evaluation opponent)
│   │   └── SplendorTrainingData.swift # Training data structs + binary serialization
│   │
│   ├── utility/
│   │   ├── Utility.swift              # PRNG (SeededRandomNumberGenerator) and misc
│   │   ├── OptionParser.swift         # CLI argument parsing
│   │   └── Report.swift               # Structured JSON report helpers
│   │
│   ├── Orion.swift                    # Main entry point (dispatches to subcommands)
│   ├── Common.swift                   # Shared helpers (initializeAgents, sampling helpers)
│   ├── DataGenerator.swift            # `orion generate` — self-play data collection
│   ├── NetworkTrainer.swift           # `orion train` — training loop with early stopping
│   ├── GameplayTester.swift           # `orion play` — evaluation and interactive play
│   └── HumanAgent.swift               # Human player agent for interactive mode
│
├── scripts/
│   ├── iterative_gameplay.py          # Multi-generation self-play training orchestrator
│   ├── summarize_run.py               # Reads a run.json, prints a per-generation summary
│   ├── sweep.sh                       # Hyperparameter sweep harness
│   └── docopt.py                      # Vendored docopt dependency
│
├── Package.swift
├── build.sh                           # xcodebuild wrapper (required for Metal shaders)
└── CLAUDE.md
```

## CLI Tools

Orion provides three subcommands (dispatched by `Orion.swift`). All three accept `--report-json PATH` to emit a structured report, `--precision fp32|bf16|fp16`, `-b/--batch-size` (parallel lanes), and `--serial`.

### `orion generate` — Generate training data
Plays self-play games via batched MCTS and saves training examples.
- `-n` game count, `-a` agent (model path or "random"/"heuristic"), `-o` output path
- `-t` temperature, `-s` seed, `-p` player count, `--max-turns`
- `--monte-carlo-samples` MCTS sims/move (default 1), `--c-puct` (default 1.5)
- `--mcts-leaf-batch` leaves selected per MCTS round via virtual loss (default 8)
- Output: `.bin.lz4` binary format (LZ4-compressed packed floats)

### `orion train` — Train neural network
Trains on generated data with Adam/AdamW.
- `-i` input data (file or directory — a directory loads and merges all `.lz4/.gz/.json` files, non-recursively)
- `-o` output model path, `-m` existing model to continue from
- `-e` epochs, `-b` batch size, `-r` learning rate, `-w` weight decay, `-d` dropout
- `-f` validation split (default 0.1), `-E` early-stopping patience, `-O` optimizer (adam/adamw/sgd)
- `-P`/`-V` policy/value loss weights, `--precision` (default bf16 for training)
- Passing nonzero `-w` auto-selects AdamW. Note: `-l/--learning-rate-decay` is parsed but currently NOT applied in the training loop.
- Saves the best-validation-loss checkpoint (early stopping on combined val loss).

### `orion play` — Play and evaluate
Plays games for evaluation or interactive human play.
- `-n` game count, `-a` agent specs (one per player, or one broadcast to all)
- `-t` temperature (0 = greedy), `--max-turns`, `-v` verbose, `--show-probabilities`
- `--monte-carlo-samples` (default 0 = raw single-ply policy; >0 enables determinized MCTS)
- `--determinizations` hidden-deck reshuffles aggregated per move (default 8)
- `--mcts-leaf-batch` (default 8), `--c-puct` (default 1.5)
- Interactive mode: `orion play -a human models/model/` (colored probability bars, card descriptions)

## Neural Network Architecture

**PolicyValueNetwork** (`SplendorNeuralAgent.swift`):
- Input: **496** floats (encoded game state)
- Shared trunk: 496 → 512 → 512 → 512 (ReLU + dropout, configurable rate, default 0.1)
- Policy head: 512 → 48 (raw logits, no activation)
- Value head: 512 → 128 → 1 (ReLU, then tanh for [-1, 1])
- He initialization, architecture version **7**
- ~870k parameters
- Default precision **bfloat16** for weights/training (fp32 loss math). Inference paths (`generate`, `play`) default to fp32 via `--precision`.
- Dropout rate stored in `architecture.json`, configurable via `--dropout`
- `batchPredict` evaluates many game states in one dispatch (used by MCTS batching).

## Game State Encoding (496 Float16 values)

```
Game State Encoding — Game.GAME_STATE_ENCODING_SIZE = 496
═══════════════════════════════════════════════════════════════════════════

 Index    Field                                     Size   Normalization
───────────────────────────────────────────────────────────────────────────
 0-49     Current player state                       50    (see player encoding)
 50-99    Next player state                          50
 100-149  Player +2 state (or zeros)                 50    zero-padded if <3 players
 150-199  Player +3 state (or zeros)                 50    zero-padded if <4 players

          Players rotated so slot 0 is always the player to move next.
───────────────────────────────────────────────────────────────────────────
 200-204  supply[red,green,blue,white,brown]          5    /6
 205      goldGemSupply                               1    /5
───────────────────────────────────────────────────────────────────────────
 206-215  take-three yields (10 combinations)        10    /3   how many gems the
 216-220  take-two yields (5 colors)                  5    /2   move would deliver
          Ordering matches generateAllCanonicalMoves (lex 3-combos, then colors);
          flag i maps 1:1 to canonical move 15+i (take-three) / 25+i (take-two).
───────────────────────────────────────────────────────────────────────────
 221-350  5 nobles × 26 each                        130    (see noble encoding)
───────────────────────────────────────────────────────────────────────────
 351-494  Visible cards: 3 tiers × 4 positions × 12  144   (see card encoding)
          Zero-padded for empty positions.
───────────────────────────────────────────────────────────────────────────
 495      turnNumber                                  1    tanh(currentTurn)
───────────────────────────────────────────────────────────────────────────

Noble encoding (26 floats each, zero-padded if fewer than 5 nobles):
  1  points                                    /3
  5  price [red, green, blue, white, brown]    /4
 20  per-player card-color deficit             /4   4 players × 5 colors,
                                                     rotated to match player slots;
                                                     deficit = max(0, price - cardCount)

Card encoding (Card.ENCODED_SIZE = 12):
  1  points                        /10
  5  price [r,g,b,w,k]             /10
  5  color one-hot
  1  affordability flag            1.0 if current player can afford, else 0.0
```

```
Player Encoding — PlayerState.ENCODED_SIZE = 50
═══════════════════════════════════════════════════════════════════════════

 Index  Field                        Normalization
─────────────────────────────────────────────────────────────────────────
  0-4   gems[red,green,blue,white,brown]     /10
  5     goldGems                             /10
  6-10  cardPower[r,g,b,w,k] (owned cards)   /7
 11     reservedCount                        /3
 12-23  Reserved Card 0 (or zeros)           12 floats (card encoding, incl. affordability)
 24-35  Reserved Card 1 (or zeros)           12 floats
 36-47  Reserved Card 2 (or zeros)           12 floats
 48     nobleCount                           /5
 49     score                                /15
─────────────────────────────────────────────────────────────────────────
```

Note: `turnNumber` is encoded as `tanh(currentTurn)`, which saturates toward 1.0 within a few turns — it carries little signal past the opening. `cardPower` and `score` are maintained incrementally on `PlayerState` (updated in `addCard`/`addNobles`) rather than recomputed, so encoding and legality checks stay cheap.

## MCTS Search (`MCTS.swift`)

AlphaZero-style PUCT search over `Game` states, using any `AgentProtocol` for priors and leaf value. Core primitives are agent-agnostic and split for batching: `selectLeaf`, `selectLeavesWithVirtualLoss`, `completeEvaluation`, `backpropagate`, `visitCountPolicy`, `aggregatedPolicy`.

### Data generation (`orion generate`)
- Many games run as parallel "lanes"; each move runs `monteCarloSamples` simulations.
- **Virtual-loss multi-leaf batching**: each round selects up to `mctsLeafBatch` distinct leaves per tree (temporary pessimistic loss along each selected path forces divergence), then evaluates all pending leaves across all lanes in one batched network dispatch. Cuts GPU round trips ~K× at fixed simulation count.
- **Subtree reuse**: the chosen child becomes the next move's root, so search starts warm.
- Full-information search here is fair — self-play sees a representative spread of deck orders across many games.

### Evaluation (`orion play`, when `--monte-carlo-samples > 0`)
- **Determinization** handles hidden information. Splendor's only hidden state is the face-down deck order; `Game.determinized(seed:)` reshuffles the unseen deck tail (visible cards untouched) from an independent RNG, so search never exploits deck order no real player can see.
- Each move samples `determinizations` independent reshuffles, runs a full search on each, and picks the move by visit counts summed across all of them (`aggregatedPolicy`).
- `DumbAgent` (random) opponents skip search entirely and fall back to raw sampling — their coin-flip value estimate would otherwise corrupt the search tree.

## Training Pipeline

### Iterative Self-Play (`scripts/iterative_gameplay.py`)
Orchestrates generate → train → evaluate over many generations. Starts from EITHER `--initial-model` (gen 1 generates from it) OR `--initial-data` (gen 1 trains on supplied data, skipping generation). Exactly one is required.

- **Champion gating**: a new model must beat the previous champion by ≥ `--champion-threshold` (default 0.52) in head-to-head play to be accepted; otherwise the previous champion is retained.
- **Evaluation** uses determinized MCTS (`--eval-monte-carlo-samples`, `--eval-determinizations`, `--eval-mcts-leaf-batch`, `--eval-c-puct`) — kept deliberately separate from generation-side MCTS settings. Each generation evaluates vs random, vs heuristic, and vs the previous champion.
- **Example-targeted generation**: `--target-examples` (default 120000) holds training-signal volume roughly constant as games shorten. The game count for each generation is computed from the previous generation's realized examples-per-game; `--games-per-generation` is only a fallback when no prior report exists.
- **Sliding-window accumulation**: `--accumulate-window N` (default 10) trains on the last N generations of data. Older data files are physically moved into `trainingdata/stale/` (the trainer's directory glob is shallow, so shelved files are excluded with no other machinery). `N=1` reproduces "newest generation only".
- **Temperature schedule**: linear decay from `--initial-temp` (1.5) to `--final-temp` (0.5).
- **LR decay**: geometric per generation (`--lr-decay`, default 0.95×).
- Per-step reports written as `RUN_DIR/gen_NN_*.json`; master aggregate as `RUN_DIR/run.json` (rewritten atomically each generation). Commands logged to `RUN_DIR/commands.log`.
- `scripts/summarize_run.py RUN_DIR/run.json` prints a per-generation summary table.

Orchestrator defaults of note: 15 generations, target 120k examples/gen, epochs ≤ 100 (early-stop patience 10), training batch 256, LR 3e-4, weight decay 0.0, dropout 0.1, generation MCTS 25 sims (users often raise this).

### Training Data Format
Binary `.bin.lz4` files (LZ4-compressed):
- 24-byte header: magic "ORIN" (0x4F52494E), version, state dim, policy dim, example count, reserved
- Packed examples: [496×f32 state][48×f32 policy][1×f32 value] = 2180 bytes each
- Loaded from a single file or a directory (directory merges all data files, non-recursively)
- Legacy `.gz` JSON format still loadable

### Loss Function
- **Policy loss**: cross-entropy between predicted distribution and the MCTS visit distribution at each position. All examples weighted equally regardless of game outcome.
- **Value loss**: MSE between predicted and actual outcome (+1 win / −1 loss / 0 tie), from the perspective of the player to move.
- Combined: `policyWeight * policyLoss + valueWeight * valueLoss`.
- Forward pass runs in bf16; softmax/MSE upcast to fp32 for numerical stability.

## Splendor Game Rules

- **48 canonical moves**: 12 purchase + 3 purchase-reserved + 10 take-three-gems + 5 take-two-gems + 12 reserve + 6 discard.
- **Two-phase turns**: normal action → discard phase (if > 10 gems) → next player. `currentTurn` advances only when play passes to the next player, so discard sub-moves do not inflate the turn count.
- **Terminal conditions**: playerWon (15+ points), tied, timedOut, inProgress.
- **Gold gems**: wildcards for purchasing; deducted from the player and returned to supply when spent.
- **Hidden information**: only the face-down deck order below the 4 visible cards per tier (see MCTS determinization). Reserved cards in this implementation are fully public in the encoding.
- 2-4 players, 5 gem types + gold, 90 cards across 3 tiers, nobles.

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
- **Python 3** with `docopt` (vendored at `scripts/docopt.py`) for the orchestrator scripts
