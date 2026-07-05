import Swift
import Foundation
import Core
import MLX
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

   /// camelCase identifier for structured output (the rawValue holds the display label).
   var jsonKey: String {
      switch self {
      case .purchaseTier1:    return "purchaseTier1"
      case .purchaseTier2:    return "purchaseTier2"
      case .purchaseTier3:    return "purchaseTier3"
      case .purchaseReserved: return "purchaseReserved"
      case .takeThreeGems:    return "takeThreeGems"
      case .takeTwoGems:      return "takeTwoGems"
      case .reserveCard:      return "reserveCard"
      case .discardGem:       return "discardGem"
      }
   }

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

   /// Build a serializable structured representation of the statistics.
   func toReport () -> DataGenerator.MoveStatisticsReport {
      let totalWinner = winnerMoves.values.reduce(0, +)
      let totalLoser  = loserMoves.values.reduce(0, +)
      let totalTied   = tiedMoves.values.reduce(0, +)
      var byMoveType: [String: DataGenerator.MoveCountsReport] = [:]
      for category in MoveCategory.allCases {
         byMoveType[category.jsonKey] = DataGenerator.MoveCountsReport(
            winner: winnerMoves[category]!,
            loser:  loserMoves[category]!,
            tied:   tiedMoves[category]!)
      }
      return DataGenerator.MoveStatisticsReport(
         totals: DataGenerator.MoveCountsReport(winner: totalWinner, loser: totalLoser, tied: totalTied),
         byMoveType: byMoveType)
   }
}

/// State for one concurrent game lane during batched MCTS data generation.
private struct GameLane {
   var game: Splendor.Game
   var mctsRoot: MCTSNode
   var rng: SeededRandomNumberGenerator
   var examples: [TrainingExample]
   var moves: [(playerIndex: Int, moveIndex: Int)]
   var gameIndex: Int
   var active: Bool
   // Whether Dirichlet root noise has been applied to mctsRoot for the current
   // move yet. Reset each time a new root is set (new move / new game).
   var rootNoiseApplied: Bool
   // Turn count is not tracked here — game.currentTurn is authoritative and
   // counts player turns correctly (a discard sub-move does not advance it).
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
      opts.addOption("Data Generator", "", "mcts-leaf-batch", "Leaves selected per MCTS round per game via virtual loss (default: 8, 1 = classic one-at-a-time)")
      opts.addOption("Data Generator", "", "c-puct", "MCTS exploration constant (default: 1.5)")
      opts.addOption("Data Generator", "", "dirichlet-alpha", "Symmetric Dirichlet concentration for root exploration noise (default: 0.3)")
      opts.addOption("Data Generator", "", "dirichlet-epsilon", "Root-prior weight given to Dirichlet noise during self-play (default: 0.25, 0 = disable)")
      opts.addOption("Data Generator", "b", "batch-size", "Number of games per batch / parallel lanes (default: 128)")
      opts.addOption("Data Generator", "", "mcts-debug", "Print MCTS search tree and π after every move (very verbose, for debugging)", requireArgument: false)
      opts.addOption("Data Generator", "", "serial", "Force single-threaded generation (default: concurrent)", requireArgument: false)
      opts.addOption("Data Generator", "", "precision", "Numeric precision for the neural agent: fp32, bf16, fp16 (default: fp32)")
      opts.addOption("Data Generator", "", "report-json", "Write a structured JSON report of inputs and results to this path")
   }


   // MARK: - Structured report

   struct GenerateReport: Encodable {
      let schemaVersion: Int
      let command: String
      let startedAt: String
      let completedAt: String
      let elapsedSeconds: Double
      let parameters: Parameters
      let results: Results

      struct Parameters: Encodable {
         let agent: String
         let agentKind: String
         let agentLabel: String
         let gameCount: Int
         let playerCount: Int
         let temperature: Float
         let maxTurns: Int
         let seed: UInt64
         let monteCarloSamples: Int
         let mctsLeafBatch: Int
         let cPuct: Float
         let dirichletAlpha: Float
         let dirichletEpsilon: Float
         let batchSize: Int
         let output: String
      }

      struct Results: Encodable {
         let successfulGames: Int
         let timedOutGames: Int
         let totalExamples: Int
         // Examples (decisions) include discard sub-moves; turns count only
         // completed player turns. They differ whenever a player over-takes
         // gems and must discard, so both are reported distinctly.
         let avgExamplesPerGame: Double
         let avgTurnsPerGame: Double
         let outputFile: String
         let outputBytesUncompressed: Int
         let outputBytesCompressed: Int
         let moveStatistics: MoveStatisticsReport
      }
   }

   struct MoveCountsReport: Encodable {
      let winner: Int
      let loser: Int
      let tied: Int
   }

   struct MoveStatisticsReport: Encodable {
      let totals: MoveCountsReport
      let byMoveType: [String: MoveCountsReport]
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
      laneCount: Int,
      mctsLeafBatch: Int,
      dirichletAlpha: Float,
      dirichletEpsilon: Float,
      baseGameIndex: Int = 0) -> (games: [GameData], statistics: MoveStatistics) {

      let actualLanes = min(laneCount, gameCount)
      var nextGameIndex = actualLanes
      var completedGames: [GameData] = []
      var statistics = MoveStatistics()

      func initLane (_ gameIndex: Int) -> GameLane? {
         let globalIndex = baseGameIndex + gameIndex
         let seed = baseSeed + UInt64(gameIndex)
         guard let game = Splendor.Game(playerCount: playerCount, seed: seed) else { return nil }
         return GameLane(
            game: game,
            mctsRoot: MCTSNode(),
            rng: SeededRandomNumberGenerator(seed: seed),
            examples: [],
            moves: [],
            gameIndex: globalIndex,
            active: true,
            rootNoiseApplied: false)
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
         let localIndex = lane.gameIndex - baseGameIndex
         return GameData(
            gameIndex: lane.gameIndex, seed: baseSeed + UInt64(localIndex),
            playerCount: playerCount, winner: winner, turnCount: lane.game.currentTurn,
            examples: examplesWithValues, moves: lane.moves)
      }

      // Initialize lanes
      var lanes = (0..<actualLanes).compactMap { initLane($0) }

      // Ensure agent is in inference mode (e.g. disable dropout for neural agents)
      mctsSearch.agent.prepareForInference()

      while lanes.contains(where: { $0.active }) {
         // --- Simulation phase: batched selection + eval rounds ---
         // Each round selects up to mctsLeafBatch leaves per lane under virtual
         // loss, so one network dispatch serves up to laneCount × mctsLeafBatch
         // evaluations. Total selections per move stays ≈ monteCarloSamples.
         let leafBatch = max(1, mctsLeafBatch)
         let roundCount = (mctsSearch.monteCarloSamples + leafBatch - 1) / leafBatch
         for _ in 0..<roundCount {
            // Apply Dirichlet root noise once per move, as soon as the root is
            // expanded and before any PUCT descent uses its priors. A reused
            // (already-expanded) root is noised at the top of round 1; a fresh
            // root is expanded during round 1's eval and noised atop round 2
            // (round 1 only selected the root itself as a leaf — no prior use).
            if dirichletEpsilon > 0 {
               for i in lanes.indices where lanes[i].active {
                  if !lanes[i].rootNoiseApplied && lanes[i].mctsRoot.isExpanded {
                     mctsSearch.applyDirichletNoise(
                        root: lanes[i].mctsRoot, alpha: dirichletAlpha,
                        epsilon: dirichletEpsilon, rng: &lanes[i].rng)
                     lanes[i].rootNoiseApplied = true
                  }
               }
            }

            var pending: [(laneIdx: Int, result: SelectionResult)] = []

            for i in lanes.indices {
               guard lanes[i].active else { continue }
               let results = mctsSearch.selectLeavesWithVirtualLoss(
                  root: lanes[i].mctsRoot, game: lanes[i].game, count: leafBatch)
               for result in results {
                  pending.append((i, result))
               }
            }

            if !pending.isEmpty {
               let leafGames = pending.map { $0.result.leafGame }
               let (allLogits, allValues) = mctsSearch.batchEvaluate(leafGames: leafGames)
               for (j, (_, result)) in pending.enumerated() {
                  mctsSearch.completeEvaluation(result: result, logits: allLogits[j], value: allValues[j])
               }
            }
         }

         // --- Move application phase ---
         for i in lanes.indices {
            guard lanes[i].active else { continue }

            // Check timeout
            if lanes[i].game.currentTurn >= maxTurns {
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
               mctsSearch.printSearchResults(root: lanes[i].mctsRoot, policy: policy, turn: lanes[i].game.currentTurn)
            }
            let currentPlayer = lanes[i].game.currentPlayer
            let stateEncoding = lanes[i].game.encoding().map { Float($0) }
            let moveIndex = sampleMoveFromPolicy(policy, rng: &lanes[i].rng)

            lanes[i].examples.append(TrainingExample(
               turnNumber: lanes[i].game.currentTurn, playerIndex: currentPlayer,
               state: stateEncoding, policy: policy, value: 0.0))
            lanes[i].moves.append((playerIndex: currentPlayer, moveIndex: moveIndex))

            lanes[i].game.applyMove(canonicalMoveIndex: moveIndex)

            // Reuse the chosen child's subtree as the new search root — its visits and
            // priors carry over, so the next move's simulations start warm instead of
            // rebuilding the tree from nothing. Falls back to a fresh node when the
            // child was never expanded (e.g. move sampled from the prior fallback).
            lanes[i].mctsRoot = lanes[i].mctsRoot.child(action: moveIndex) ?? MCTSNode()
            lanes[i].rootNoiseApplied = false

            // Check if game is complete
            if case .inProgress = lanes[i].game.terminalCondition { } else {
               if let gameData = finalizeGame(lanes[i]) {
                  completedGames.append(gameData)
                  for (pIdx, mIdx) in gameData.moves {
                     statistics.recordMove(moveIndex: mIdx, playerIndex: pIdx, winner: gameData.winner)
                  }
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

   /// Statistics returned from a generation run, suitable for building a structured report.
   public struct GenerateStats {
      public let successfulGames: Int
      public let timedOutGames: Int
      public let totalExamples: Int
      public let totalTurns: Int
      public let outputFile: String
      public let outputBytesUncompressed: Int
      public let outputBytesCompressed: Int
      let moveStatistics: MoveStatistics
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
      mctsLeafBatch: Int = 8,
      cPuct: Float = 1.5,
      dirichletAlpha: Float = 0.3,
      dirichletEpsilon: Float = 0.25,
      mctsDebug: Bool = false,
      batchSize: Int = 128,
      serial: Bool = false,
      precision: DType = PolicyValueNetwork.DEFAULT_PRECISION) throws -> GenerateStats {

      precondition(monteCarloSamples >= 1, "monteCarloSamples must be at least 1")

      let taskCount = (gameCount + batchSize - 1) / batchSize

      print("Configuration:")
      print("  Games:            \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Agent:            \(agentSpec.isEmpty ? "random" : agentSpec)")
      print("  Temperature:      \(String(format: "%.2f", temperature))")
      print("  Max turns:        \(maxTurns)")
      print("  Seed:             \(seed)")
      print("  MCTS sims/move:   \(monteCarloSamples)  (c_puct=\(cPuct), leaf batch=\(mctsLeafBatch))")
      print("  Dirichlet noise:  \(dirichletEpsilon > 0 ? "alpha=\(dirichletAlpha), epsilon=\(dirichletEpsilon)" : "disabled")")
      print("  Batch size:       \(batchSize)")
      print("  Tasks:            \(taskCount)\(serial ? " (serial)" : " (concurrent)")")
      print("  Output:           \(outputPath)")

      let workQueue: DispatchQueue
      if serial {
         workQueue = DispatchQueue(label: "orion.generate.work")
      } else {
         workQueue = DispatchQueue(label: "orion.generate.work", attributes: .concurrent)
      }
      let resultQueue = DispatchQueue(label: "orion.generate.results")
      let group = DispatchGroup()

      // Thread-safe result storage — each task writes to its own slot
      let taskResults = UnsafeMutableBufferPointer<(games: [GameData], statistics: MoveStatistics)?>.allocate(capacity: taskCount)
      taskResults.initialize(repeating: nil)
      defer { taskResults.deallocate() }

      for taskIndex in 0..<taskCount {
         let taskOffset = taskIndex * batchSize
         let taskGameCount = min(batchSize, gameCount - taskOffset)
         let taskBaseSeed = seed + UInt64(taskOffset)

         group.enter()
         workQueue.async {
            let agent = initializeAgents(playerCount: 1, agentSpecs: [agentSpec], seed: taskBaseSeed, precision: precision)[0]
            let mctsSearch = MCTSSearch(agent: agent, monteCarloSamples: monteCarloSamples, cPuct: cPuct, debug: mctsDebug)

            let result = batchedGenerateGames(
               mctsSearch: mctsSearch,
               gameCount: taskGameCount,
               playerCount: playerCount,
               temperature: temperature,
               maxTurns: maxTurns,
               baseSeed: taskBaseSeed,
               laneCount: taskGameCount,
               mctsLeafBatch: mctsLeafBatch,
               dirichletAlpha: dirichletAlpha,
               dirichletEpsilon: dirichletEpsilon,
               baseGameIndex: taskOffset)

            taskResults[taskIndex] = result
            resultQueue.async {
               let completed = taskResults.compactMap({ $0 }).reduce(0, { $0 + $1.games.count })
               print("Completed \(completed)/\(gameCount) games...")
               group.leave()
            }
         }
      }

      group.wait()

      // Merge results from all tasks
      var allGameData: [GameData] = []
      var statistics = MoveStatistics()
      for taskIndex in 0..<taskCount {
         guard let result = taskResults[taskIndex] else { continue }
         allGameData.append(contentsOf: result.games)
         for game in result.games {
            for (pIdx, mIdx) in game.moves {
               statistics.recordMove(moveIndex: mIdx, playerIndex: pIdx, winner: game.winner)
            }
         }
      }

      let successfulGames = allGameData.count
      let totalExamples = allGameData.reduce(0) { $0 + $1.examples.count }
      let totalTurns = allGameData.reduce(0) { $0 + $1.turnCount }
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
      let (uncompressedBytes, compressedBytes) = try dataset.save(to: outputPath, compress: true)

      print("\nTraining data generation complete!")
      print("  Successful games: \(successfulGames)/\(gameCount)")
      print("  Total training examples: \(totalExamples)")
      print("  Average examples per game: \(totalExamples / max(successfulGames, 1))  (decisions, includes discards)")
      print("  Average turns per game:    \(totalTurns / max(successfulGames, 1))  (completed player turns)")
      print("  Saved to: \(outputPath).bin.lz4")

      // Print move statistics
      print("\nComputing move statistics...")
      statistics.printSummary()
      print("\nDone!")

      return GenerateStats(
         successfulGames: successfulGames,
         timedOutGames: gameCount - successfulGames,
         totalExamples: totalExamples,
         totalTurns: totalTurns,
         outputFile: outputPath + ".bin.lz4",
         outputBytesUncompressed: uncompressedBytes,
         outputBytesCompressed: compressedBytes,
         moveStatistics: statistics)
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
      let mctsLeafBatch = opts.get(option: "mcts-leaf-batch", orElse: 8)
      let cPuct = opts.get(option: "c-puct", orElse: Float(1.5))
      let dirichletAlpha = opts.get(option: "dirichlet-alpha", orElse: Float(0.3))
      let dirichletEpsilon = opts.get(option: "dirichlet-epsilon", orElse: Float(0.25))
      let mctsDebug = opts.wasProvided(option: "mcts-debug")
      let batchSize = opts.get(option: "batch-size", orElse: 128)
      let serial = opts.wasProvided(option: "serial")
      let precisionStr = opts.get(option: "precision", orElse: "fp32")
      let precision = parsePrecision(precisionStr)
      let reportPath: String? = opts.get(option: "report-json")

      let startDate = Date()
      let stats = try generateTrainingData(
         gameCount: gameCount,
         playerCount: playerCount,
         agentSpec: agentSpec,
         temperature: temperature,
         seed: baseSeed,
         maxTurns: maxTurns,
         outputPath: outputPath,
         monteCarloSamples: monteCarloSamples,
         mctsLeafBatch: mctsLeafBatch,
         cPuct: cPuct,
         dirichletAlpha: dirichletAlpha,
         dirichletEpsilon: dirichletEpsilon,
         mctsDebug: mctsDebug,
         batchSize: batchSize,
         serial: serial,
         precision: precision
      )
      let endDate = Date()

      if let reportPath = reportPath {
         let report = GenerateReport(
            schemaVersion:  Report.SCHEMA_VERSION,
            command:        "generate",
            startedAt:      Report.timestamp(startDate),
            completedAt:    Report.timestamp(endDate),
            elapsedSeconds: endDate.timeIntervalSince(startDate),
            parameters: GenerateReport.Parameters(
               agent:             agentSpec,
               agentKind:         Report.agentKind(spec: agentSpec),
               agentLabel:        Report.agentLabel(spec: agentSpec),
               gameCount:         gameCount,
               playerCount:       playerCount,
               temperature:       temperature,
               maxTurns:          maxTurns,
               seed:              baseSeed,
               monteCarloSamples: monteCarloSamples,
               mctsLeafBatch:     mctsLeafBatch,
               cPuct:             cPuct,
               dirichletAlpha:    dirichletAlpha,
               dirichletEpsilon:  dirichletEpsilon,
               batchSize:         batchSize,
               output:            outputPath),
            results: GenerateReport.Results(
               successfulGames:        stats.successfulGames,
               timedOutGames:          stats.timedOutGames,
               totalExamples:          stats.totalExamples,
               avgExamplesPerGame:     Double(stats.totalExamples) / Double(max(stats.successfulGames, 1)),
               avgTurnsPerGame:        Double(stats.totalTurns) / Double(max(stats.successfulGames, 1)),
               outputFile:             stats.outputFile,
               outputBytesUncompressed: stats.outputBytesUncompressed,
               outputBytesCompressed:   stats.outputBytesCompressed,
               moveStatistics:         stats.moveStatistics.toReport()))
         try Report.write(report, to: reportPath)
         print("Report written to: \(reportPath)")
      }
   }
}
