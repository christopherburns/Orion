import Swift
import Foundation
import Core
import Splendor
import Utility

/// Move type categories for histogram tracking
enum MoveCategory: String, CaseIterable {
   case purchaseTier1 = "Purchase Tier 1"
   case purchaseTier2 = "Purchase Tier 2"
   case purchaseTier3 = "Purchase Tier 3"
   case purchaseReserved = "Purchase Reserved"
   case takeThreeGems = "Take Three Gems"
   case takeTwoGems = "Take Two Gems"
   case reserveCard = "Reserve Card"
   case discardGem = "Discard Gem"

   /// Categorize a move index into its type
   /// - Parameter moveIndex: Canonical move index (0-47)
   /// - Returns: The category this move belongs to
   static func categorize (_ moveIndex: Int) -> MoveCategory {
      switch moveIndex {
      case 0..<4:
         return .purchaseTier1
      case 4..<8:
         return .purchaseTier2
      case 8..<12:
         return .purchaseTier3
      case 12..<15:
         return .purchaseReserved
      case 15..<25:
         return .takeThreeGems
      case 25..<30:
         return .takeTwoGems
      case 30..<42:
         return .reserveCard
      case 42..<48:
         return .discardGem
      default:
         preconditionFailure("Invalid move index: \(moveIndex)")
      }
   }
}

/// Statistics about move types in the generated training data
struct MoveStatistics {
   var winnerMoves: [MoveCategory: Int] = [:]
   var loserMoves: [MoveCategory: Int] = [:]
   var tiedMoves: [MoveCategory: Int] = [:]

   init () {
      // Initialize all buckets to 0
      for category in MoveCategory.allCases {
         winnerMoves[category] = 0
         loserMoves[category] = 0
         tiedMoves[category] = 0
      }
   }

   /// Record a move made by a player
   /// - Parameters:
   ///   - moveIndex: The canonical move index
   ///   - playerIndex: Which player made the move
   ///   - winner: The winner of the game (nil if tied)
   mutating func recordMove (moveIndex: Int, playerIndex: Int, winner: Int?) {
      let category = MoveCategory.categorize(moveIndex)

      if let winnerIndex = winner {
         if playerIndex == winnerIndex {
            winnerMoves[category]! += 1
         } else {
            loserMoves[category]! += 1
         }
      } else {
         // Tied game
         tiedMoves[category]! += 1
      }
   }

   /// Print formatted statistics
   func printSummary () {
      let totalWinner = winnerMoves.values.reduce(0, +)
      let totalLoser = loserMoves.values.reduce(0, +)
      let totalTied = tiedMoves.values.reduce(0, +)
      let grandTotal = totalWinner + totalLoser + totalTied

      // Derive count column width from the largest number that will appear
      let maxCount = max(totalWinner, totalLoser, totalTied, 1)
      let countWidth = String(maxCount).count

      // Each data cell is: countWidth digits + " (XX.X%)" = countWidth + 8 chars
      let cellWidth = countWidth + 8
      let nameWidth = 25
      let tableWidth = nameWidth + 1 + cellWidth + 2 + cellWidth + 2 + cellWidth

      func cell (_ count: Int, _ pct: Float) -> String {
         String(format: "%\(countWidth)d (%4.1f%%)", count, pct)
      }

      print("\n" + String(repeating: "=", count: tableWidth))
      print("MOVE STATISTICS")
      print(String(repeating: "=", count: tableWidth))

      print("\nOverall Totals:")
      print("  Winner moves: \(totalWinner)")
      print("  Loser moves:  \(totalLoser)")
      print("  Tied moves:   \(totalTied)")
      print("  Grand total:  \(grandTotal)")

      let header = "Move Type".padding(toLength: nameWidth, withPad: " ", startingAt: 0)
         + " " + "Winners".padding(toLength: cellWidth, withPad: " ", startingAt: 0)
         + "  " + "Losers".padding(toLength: cellWidth, withPad: " ", startingAt: 0)
         + "  Tied"
      print("\n" + String(repeating: "-", count: tableWidth))
      print(header)
      print(String(repeating: "-", count: tableWidth))

      for category in MoveCategory.allCases {
         let winnerCount = winnerMoves[category]!
         let loserCount = loserMoves[category]!
         let tiedCount = tiedMoves[category]!

         let winnerPct = totalWinner > 0 ? Float(winnerCount) / Float(totalWinner) * 100.0 : 0.0
         let loserPct = totalLoser > 0 ? Float(loserCount) / Float(totalLoser) * 100.0 : 0.0
         let tiedPct = totalTied > 0 ? Float(tiedCount) / Float(totalTied) * 100.0 : 0.0

         let name = category.rawValue.padding(toLength: nameWidth, withPad: " ", startingAt: 0)
         print("\(name) \(cell(winnerCount, winnerPct))  \(cell(loserCount, loserPct))  \(cell(tiedCount, tiedPct))")
      }
      print(String(repeating: "=", count: tableWidth))
   }
}

/// State for one concurrent game lane during batched MCTS data generation.
private struct GameLane {
   var game: Splendor.Game
   var mctsRoot: MCTSNode
   var rng: SeededRandomNumberGenerator
   var examples: [TrainingExample]
   var moves: [(playerIndex: Int, moveIndex: Int)]
   var turnCount: Int
   var gameIndex: Int
   var active: Bool
}

public struct DataGenerator {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Data Generator", "n", "game-count", "Number of self-play games to generate")
      opts.addOption("Data Generator", "a", "agent", "Path to model file, or name of non-model-based agent (optional, default is 'random')",
         longDoc:
            "Specifies which agent(s) to use for gameplay. " +
            "If a path to a model file is provided (e.g., 'models/best.mlx'), " +
            "a neural network agent will be loaded from that file. " +
            "Alternatively, you can specify 'random' to use a random agent " +
            "that makes valid moves uniformly at random (default). " +
            "The model file path can be relative to the current working " +
            "directory or an absolute path.")
      opts.addOption("Data Generator", "o", "output", "Output file path for training data (base filename, .bin extension added automatically, default: trainingdata/data_TIMESTAMP)")
      opts.addOption("Data Generator", "p", "player-count", "Number of players (default: 2)")
      opts.addOption("Data Generator", "s", "seed", "Random seed for game generation (default: random)")
      opts.addOption("Data Generator", "t", "temperature", "Sampling temperature for move selection (default: 1.0, higher = more exploration)")
      opts.addOption("Data Generator", "", "max-turns", "Maximum turns per game before timeout (default: 1000)")
      opts.addOption("Data Generator", "", "monte-carlo-samples", "MCTS simulations per move (default: 1, minimum: 1)")
      opts.addOption("Data Generator", "", "c-puct", "MCTS exploration constant (default: 1.5)")
      opts.addOption("Data Generator", "b", "batch-size", "Number of games to run in parallel (default: 128)")
      opts.addOption("Data Generator", "", "mcts-debug", "Print MCTS search tree and π after every move (very verbose, for debugging)", requireArgument: false)
   }


   /// Sample a move index from a policy distribution using CDF sampling.
   private static func sampleMove (from policy: [Float], rng: inout SeededRandomNumberGenerator) -> Int {
      let threshold = Float.random(in: 0.0..<1.0, using: &rng)
      var cumulative: Float = 0.0
      for i in 0..<policy.count {
         cumulative += policy[i]
         if cumulative > threshold { return i }
      }
      return policy.indices.max(by: { policy[$0] < policy[$1] }) ?? 0
   }

   /// Run batched MCTS data generation with `laneCount` games in parallel.
   /// Each simulation round issues one batched network call for all active lanes.
   private static func batchedGenerateGames (
      mctsSearch: MCTSSearch,
      gameCount: Int,
      playerCount: Int,
      temperature: Float,
      maxTurns: Int,
      baseSeed: UInt64,
      laneCount: Int
   ) -> (games: [GameData], statistics: MoveStatistics) {

      let actualLanes = min(laneCount, gameCount)
      var nextGameIndex = actualLanes
      var completedGames: [GameData] = []
      var statistics = MoveStatistics()

      func initLane (_ gameIndex: Int) -> GameLane? {
         let seed = baseSeed + UInt64(gameIndex)
         guard let game = Splendor.Game(playerCount: playerCount, seed: seed) else { return nil }
         return GameLane(
            game: game,
            mctsRoot: MCTSNode(),
            rng: SeededRandomNumberGenerator(seed: seed),
            examples: [],
            moves: [],
            turnCount: 0,
            gameIndex: gameIndex,
            active: true)
      }

      func finalizeGame (_ lane: GameLane) -> GameData? {
         let winner: Int?
         if case .playerWon(let idx) = lane.game.terminalCondition { winner = idx }
         else { winner = nil }

         let examplesWithValues = lane.examples.map { ex in
            let value: Float
            if let w = winner { value = (ex.playerIndex == w) ? 1.0 : -1.0 }
            else { value = 0.0 }
            return TrainingExample(
               turnNumber: ex.turnNumber, playerIndex: ex.playerIndex,
               state: ex.state, policy: ex.policy, value: value)
         }
         return GameData(
            gameIndex: lane.gameIndex, seed: baseSeed + UInt64(lane.gameIndex),
            playerCount: playerCount, winner: winner, turnCount: lane.turnCount,
            examples: examplesWithValues, moves: lane.moves)
      }

      // Initialize lanes
      var lanes = (0..<actualLanes).compactMap { initLane($0) }
      var completedCount = 0

      // Ensure agent is in inference mode (e.g. disable dropout for neural agents)
      mctsSearch.agent.prepareForInference()

      while lanes.contains(where: { $0.active }) {
         // --- Simulation phase: monteCarloSamples rounds of batched selection + eval ---
         for _ in 0..<mctsSearch.monteCarloSamples {
            var pending: [(laneIdx: Int, result: SelectionResult)] = []

            for i in lanes.indices {
               guard lanes[i].active else { continue }
               let result = mctsSearch.selectLeaf(root: lanes[i].mctsRoot, game: lanes[i].game)
               if let terminalValue = result.terminalValue {
                  mctsSearch.backpropagate(result: result, leafValue: terminalValue)
               } else {
                  pending.append((i, result))
               }
            }

            if !pending.isEmpty {
               let leafGames = pending.map { $0.result.leafGame }
               let (allLogits, allValues) = mctsSearch.batchEvaluate(leafGames: leafGames)
               for (j, (_, result)) in pending.enumerated() {
                  let mask = result.leafGame.legalMoveMaskForCurrentPlayer()
                  mctsSearch.expandLeaf(node: result.leafNode, logits: allLogits[j], legalMask: mask)
                  mctsSearch.backpropagate(result: result, leafValue: allValues[j])
               }
            }
         }

         // --- Move application phase ---
         for i in lanes.indices {
            guard lanes[i].active else { continue }

            // Check timeout
            if lanes[i].turnCount >= maxTurns {
               print("Warning: Game \(lanes[i].gameIndex) reached maximum turn limit (\(maxTurns))")
               if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                  lanes[i] = lane; nextGameIndex += 1
               } else {
                  lanes[i].active = false
               }
               continue
            }

            let policy = mctsSearch.visitCountPolicy(root: lanes[i].mctsRoot, temperature: temperature)
            if mctsSearch.debug && i == 0 {
               mctsSearch.printSearchResults(root: lanes[i].mctsRoot, policy: policy, turn: lanes[i].turnCount)
            }
            let currentPlayer = lanes[i].game.currentPlayer
            let stateEncoding = lanes[i].game.encoding().map { Float($0) }
            let moveIndex = sampleMove(from: policy, rng: &lanes[i].rng)

            lanes[i].examples.append(TrainingExample(
               turnNumber: lanes[i].turnCount, playerIndex: currentPlayer,
               state: stateEncoding, policy: policy, value: 0.0))
            lanes[i].moves.append((playerIndex: currentPlayer, moveIndex: moveIndex))

            lanes[i].game.applyMove(canonicalMoveIndex: moveIndex)
            lanes[i].mctsRoot = MCTSNode()
            lanes[i].turnCount += 1

            // Check if game is complete
            if case .inProgress = lanes[i].game.terminalCondition { } else {
               if let gameData = finalizeGame(lanes[i]) {
                  completedGames.append(gameData)
                  for (pIdx, mIdx) in gameData.moves {
                     statistics.recordMove(moveIndex: mIdx, playerIndex: pIdx, winner: gameData.winner)
                  }
               }
               completedCount += 1
               if completedCount % 100 == 0 {
                  print("Completed \(completedCount)/\(gameCount) games...")
               }
               if nextGameIndex < gameCount, let lane = initLane(nextGameIndex) {
                  lanes[i] = lane; nextGameIndex += 1
               } else {
                  lanes[i].active = false
               }
            }
         }
      }

      return (completedGames, statistics)
   }

   /// Generate training data programmatically (without parsing command-line args)
   public static func generateTrainingData (
      gameCount: Int,
      playerCount: Int,
      agentSpec: String,
      temperature: Float,
      seed: UInt64,
      maxTurns: Int,
      outputPath: String,
      monteCarloSamples: Int = 1,
      cPuct: Float = 1.5,
      mctsDebug: Bool = false,
      batchSize: Int = 128) throws {

      precondition(monteCarloSamples >= 1, "monteCarloSamples must be at least 1")

      print("Configuration:")
      print("  Games:            \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Agent:            \(agentSpec.isEmpty ? "random" : agentSpec)")
      print("  Temperature:      \(String(format: "%.2f", temperature))")
      print("  Max turns:        \(maxTurns)")
      print("  Seed:             \(seed)")
      print("  MCTS sims/move:   \(monteCarloSamples)  (c_puct=\(cPuct))")
      print("  Batch size:       \(batchSize)")
      print("  Output:           \(outputPath)")

      // Initialize agent for self-play
      let agents = initializeAgents(playerCount: 1, agentSpecs: [agentSpec], seed: seed)
      let agent = agents[0]

      let mctsSearch = MCTSSearch(agent: agent, monteCarloSamples: monteCarloSamples, cPuct: cPuct, debug: mctsDebug)

      let (allGameData, statistics) = batchedGenerateGames(
         mctsSearch: mctsSearch,
         gameCount: gameCount,
         playerCount: playerCount,
         temperature: temperature,
         maxTurns: maxTurns,
         baseSeed: seed,
         laneCount: batchSize)
      let successfulGames = allGameData.count

      // Create dataset
      let totalExamples = allGameData.reduce(0) { $0 + $1.examples.count }
      print("\nCreating dataset with \(totalExamples) examples from \(successfulGames) games...")

      let dataset = TrainingDataset(
         generatedAt: ISO8601DateFormatter().string(from: Date()),
         modelPath: agentSpec.isEmpty ? nil : agentSpec,
         temperature: temperature,
         totalGames: successfulGames,
         totalExamples: totalExamples,
         games: allGameData
      )

      // Save to compressed JSON
      print("Encoding to compressed JSON...")
      try dataset.save(to: outputPath, compress: true)

      print("\nTraining data generation complete!")
      print("  Successful games: \(successfulGames)/\(gameCount)")
      print("  Total training examples: \(totalExamples)")
      print("  Average examples per game: \(totalExamples / max(successfulGames, 1))")
      print("  Saved to: \(outputPath).bin.lz4")

      // Print move statistics
      print("\nComputing move statistics...")
      statistics.printSummary()
      print("\nDone!")
   }

   public static func main () throws {
      let opts = OptionParser(help: "Generate training data via self-play games")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      // Parse options
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let playerCount = opts.get(option: "player-count", orElse: 2)
      let temperature = opts.get(option: "temperature", orElse: Float(1.0))
      let maxTurns = opts.get(option: "max-turns", orElse: 1000)
      let baseSeed = opts.get(option: "seed", orElse: UInt64.random(in: 0...UInt64.max))

      // Generate default output path with timestamp
      let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
      let defaultOutputBase = "trainingdata/data_\(timestamp)"
      let outputBase = opts.get(option: "output", orElse: defaultOutputBase)

      // Strip any existing extension
      let outputURL = URL(fileURLWithPath: outputBase)
      let outputPath = outputURL.deletingPathExtension().path

      let agentSpec = opts.get(option: "agent", orElse: "random")
      let monteCarloSamples = opts.get(option: "monte-carlo-samples", orElse: 1)
      let cPuct = opts.get(option: "c-puct", orElse: Float(1.5))
      let mctsDebug = opts.wasProvided(option: "mcts-debug")
      let batchSize = opts.get(option: "batch-size", orElse: 128)

      try generateTrainingData(
         gameCount: gameCount,
         playerCount: playerCount,
         agentSpec: agentSpec,
         temperature: temperature,
         seed: baseSeed,
         maxTurns: maxTurns,
         outputPath: outputPath,
         monteCarloSamples: monteCarloSamples,
         cPuct: cPuct,
         mctsDebug: mctsDebug,
         batchSize: batchSize
      )
   }
}
