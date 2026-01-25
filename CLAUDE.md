# Orion - Neural Network Splendor AI

## Project Overview

Orion is a Swift-based system for training neural networks to play the card game Splendor. The project uses Apple's MLX framework for on-device machine learning and implements a complete Splendor game engine with a neural network agent architecture.

**Current Status:** Basic infrastructure complete. Game logic fully implemented with a random agent. Neural network architecture defined but not yet integrated into training pipeline.

## Project Structure

```
Orion/
├── src/
│   ├── core/              # Core abstractions and protocols
│   │   ├── GameProtocol.swift      # Generic game interface
│   │   ├── AgentProtocol.swift     # Generic agent interface
│   │   └── RandomAgent.swift       # Random baseline agent (DumbAgent)
│   │
│   ├── splendor/          # Splendor-specific implementation
│   │   ├── Game.swift              # Complete Splendor game logic
│   │   ├── Card.swift              # Card definitions and gem types
│   │   ├── GamePrinter.swift       # Console output formatting
│   │   └── SplendorNeuralAgent.swift # Neural network agent
│   │
│   ├── utility/           # Helper utilities
│   │   ├── Utility.swift           # PRNG and misc utilities
│   │   └── OptionParser.swift      # CLI argument parsing
│   │
│   └── Orion.swift        # Main executable entry point
│
├── Package.swift          # Swift package definition
└── CLAUDE.md             # This file
```

## Key Components

### 1. Core Protocols (`src/core/`)

**GameProtocol** - Generic interface for turn-based games:
- `canonicalMoveCount`: Total number of possible moves
- `currentTurn`, `currentPlayer`: Game state tracking
- `legalMoveMaskForCurrentPlayer()`: Returns boolean array of legal moves
- `applyMove(canonicalMoveIndex:)`: Applies a move to the game state
- `terminalCondition`: Returns game end state (won/tied/in-progress)

**AgentProtocol** - Generic interface for game-playing agents:
- `predict(game:currentPlayerIndex:)`: Returns tuple of (policyLogits, valueEstimate) for all canonical moves

**DumbAgent** (RandomAgent) - Baseline random agent that assigns random preferences to all moves. Uses seeded PRNG for reproducibility.

### 2. Splendor Game Implementation (`src/splendor/`)

**Game Logic** (Game.swift):
- Full implementation of Splendor card game rules
- **48 canonical moves**: 12 purchase + 3 purchase-reserved + 10 take-three-gems + 5 take-two-gems + 12 reserve + 6 discard
- **Two-phase turn system**:
  - Normal action phase: take gems, buy cards, or reserve cards
  - Discard phase (if needed): discard gems to get down to 10
- Victory condition: First to 15 points
- Supports 2-4 players
- Tracks: card decks (3 tiers), player states, gem supply, available nobles, current phase
- Memoized legal move masks for efficiency
- Game state encoding: 360 Float16 values for neural network input

**Card System** (Card.swift):
- 5 gem types: red, green, blue, white, brown
- 90 total cards across 3 tiers (40/30/20 distribution)
- Each card has: points, price (5-element array), and color
- Card encoding: 11 Float16 values (1 point + 5 price + 5 color one-hot)

**Player State**:
- Gems inventory (5 colors + gold)
- Owned cards and reserved cards (max 3)
- Acquired nobles (max 3)
- Score calculation (card points + noble points)
- Purchase power (permanent gems from cards + temporary gems)
- Player encoding: 47 Float16 values

**Game Rules Implemented**:
- Gem taking: 3 different colors OR 2 of same color (requires 4+ in supply)
- Card purchase: from table or from reserved cards
- Card reservation: take card + gold gem (max 3 reserved)
- Noble acquisition: automatic when requirements met (card-based only)
- **Gem limit enforcement**: Players can take gems that put them over 10, then must discard down to exactly 10
- **Two-phase turns**: Normal action → Discard phase (if >10 gems) → Next player
- Win condition: 15+ points

### 3. Neural Network Architecture (`src/splendor/SplendorNeuralAgent.swift`)

**PolicyValueNetwork**:
- Input: 360 Float16 (encoded game state)
- Architecture:
  - Shared trunk: 360 → 512 → 512 → 256 (ReLU activations)
  - Policy head: 256 → 48 (raw logits, no activation)
  - Value head: 256 → 128 → 1 (tanh activation for [-1, 1] range)
- He initialization for weights
- Architecture version: 2 (updated for 48-move system)

**SplendorNeuralAgent**:
- Implements AgentProtocol
- Uses PolicyValueNetwork for inference
- `calculateMovePreferences()`: Returns policy logits
- `predict()`: Returns both policy logits and value estimate

**Model Serialization**:
- Save/load functionality defined but not fully implemented
- Stores weights, metadata, and architecture info in JSON format
- Version checking to ensure compatibility

### 4. Main Executable (`src/Orion.swift`)

Current functionality:
- Plays multiple games with random agent (DumbAgent)
- Command-line options:
  - `-s, --seed`: PRNG seed (default: 42)
  - `-p, --player-count`: Number of players 2-4 (default: 2)
  - `-n, --game-count`: Number of games to simulate (default: 1)
- Outputs: total turns, win counts per player, tie count, win percentages
- `sampleMove()`: Selects highest-preference legal move (argmax over valid moves)

## Technical Details

### Game State Encoding (360 Float16 values)

1. **Players (188)**: 4 × 47 (zero-padded for missing players)
   - Per player: gems (5) + gold (1) + card-power (5) + reserved-count (1) + reserved-cards (33) + nobles (1) + score (1)

2. **Supply (5)**: Gem counts for each color

3. **Nobles (30)**: 5 × 6 (points + 5-element price array)

4. **Visible Cards (132)**: 3 tiers × 4 positions × 11 (card encoding)

5. **Current Player (4)**: One-hot encoding

6. **Turn Number (1)**: Normalized by 100

### Dependencies

- **mlx-swift** (v0.10.0+): Apple's MLX framework
  - MLX: Core array operations
  - MLXNN: Neural network layers
  - MLXOptimizers: Training optimizers
- **Swift 5.9+**
- **macOS 14+**

## Building and Running

**Important**: MLX Swift requires Metal shader compilation, which `swift build` cannot do. Use the provided build script instead.

```bash
# Build release version (recommended)
./build.sh release

# Build debug version
./build.sh debug

# Run with default settings (1 game, 2 players)
.build/release/orion

# Run multiple games
.build/release/orion --game-count 100 --player-count 3 --seed 12345

# Debug build (slower but better error messages)
.build/debug/orion -n 10 -p 4
```

**Build Output**:
- Binary: `.build/release/orion` or `.build/debug/orion`
- MetalLib bundle: `.build/release/mlx-swift_Cmlx.bundle/` (required at runtime)

**Symlinking**: You can symlink the binary anywhere, and it will find the metallib because they're built to the same directory:
```bash
ln -s $(pwd)/.build/release/orion ~/bin/orion
```

## Current Limitations / TODOs

### Implemented
- ✅ Complete Splendor game logic with all rules
- ✅ Generic game/agent protocol abstractions
- ✅ Random baseline agent (DumbAgent)
- ✅ Neural network architecture definition
- ✅ Game state encoding for neural input
- ✅ Command-line interface for game simulation
- ✅ Game statistics collection
- ✅ Neural network inference working with MLX
- ✅ Build system with Metal shader compilation
- ✅ Move type tracking and statistics

### Not Yet Implemented
- ❌ Training pipeline for neural network
- ❌ Self-play data collection
- ❌ Model checkpoint saving/loading (partially implemented)
- ❌ Gradient computation and backpropagation
- ❌ Experience replay buffer
- ❌ Policy and value loss calculation
- ❌ Integration of SplendorNeuralAgent into main game loop
- ❌ Evaluation framework (neural vs random, neural vs neural)
- ❌ Hyperparameter configuration system
- ❌ Training metrics and logging
- ❌ Model versioning and experiment tracking

## Architecture Notes

### Design Decisions

1. **Generic Protocols**: GameProtocol and AgentProtocol allow for multiple game implementations and agent types. Future work could add other games (Chess, Go, etc.).

2. **Canonical Move Ordering**: All possible moves are pre-enumerated in a fixed order. Agents output preferences for all moves, and illegal moves are masked out. This simplifies neural network output.

3. **Float16 Encoding**: Game state uses Float16 for memory efficiency and potential hardware acceleration benefits.

4. **Move Sampling**: Currently uses argmax selection. Could be extended with temperature-based sampling or epsilon-greedy exploration.

5. **Memoized Legal Moves**: Legal move masks are precomputed and updated only when game state changes, avoiding repeated computation.

6. **Immutable Game Logic**: Game state modifications are explicit through `mutating func applyMove()`, making state transitions clear.

### Neural Network Design Rationale

- **Two-headed architecture**: Policy head for move selection, value head for position evaluation (standard in AlphaZero-style agents)
- **Shared trunk**: Features useful for both move selection and position evaluation are learned together
- **No softmax on policy**: Outputs raw logits; softmax applied during training with legal move masking
- **Tanh on value**: Constrains value estimates to [-1, 1] range (loss = -1, tie = 0, win = 1)

## Next Steps for Training

1. Implement self-play loop:
   - Play games using current neural network
   - Store (state, policy, outcome) tuples

2. Create training pipeline:
   - Sample batches from replay buffer
   - Compute policy loss (cross-entropy) and value loss (MSE)
   - Backpropagate and update weights

3. Add evaluation:
   - Periodically test new model vs old model
   - Only keep new model if it wins >55% of games

4. Add monitoring:
   - Track loss curves
   - Monitor move diversity
   - Log win rates over time

## Game Rules Reference

For full Splendor rules, see: https://www.spacecowboys.fr/splendor

Key simplifications in this implementation:
- Simplified noble acquisition (automatic)
- Standard card distribution (no expansions)
- No explicit discard phase (gem limit enforced during taking)

## Development Notes

- Use `--seed` for reproducible random agent behavior during debugging
- Enable silence mode (automatic when game-count > 1) to avoid verbose output
- DumbAgent is useful for baseline performance measurement (~80 turns per game)
- All game state mutations go through `applyMove()` with precondition checks

### Untrained Neural Network Behavior

The neural network currently uses random He-initialized weights (untrained). This causes:
- **Highly variable game lengths**: 20-400+ turns depending on random seed
- **Pass move bias**: Some game states cause the network to heavily prefer "pass" moves (move index 42)
- **Normal for untrained networks**: This behavior will disappear once training begins

Example statistics from untrained network:
- Seed 1: 403 turns (85.9% passes)
- Seed 2: 62 turns (6.5% passes)
- Average: ~60-80 turns when not pass-heavy

Once trained, expect:
- Consistent game lengths (similar to DumbAgent: ~80 turns)
- Strategic move selection
- Minimal pass moves
