import Swift
import Foundation
import Core
import Splendor
import Utility

/// A single training example collected during a game
public struct TrainingExample: Codable {
   let turnNumber: Int
   let playerIndex: Int
   let state: [Float]  // 361-dimensional game state encoding
   let policy: [Float]  // 48-dimensional policy target (probability distribution over moves)
   let value: Float  // Value target from this player's perspective (-1, 0, or 1)
}

/// Data from a single self-play game
public struct GameData: Codable {
   let gameIndex: Int
   let seed: UInt64
   let playerCount: Int
   let winner: Int?  // Player index who won, or nil if tied
   let turnCount: Int
   let examples: [TrainingExample]
   let moves: [(playerIndex: Int, moveIndex: Int)]  // Sequence of (player, move) for statistics

   enum CodingKeys: String, CodingKey {
      case gameIndex, seed, playerCount, winner, turnCount, examples, moves
   }

   public func encode (to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(gameIndex, forKey: .gameIndex)
      try container.encode(seed, forKey: .seed)
      try container.encode(playerCount, forKey: .playerCount)
      try container.encode(winner, forKey: .winner)
      try container.encode(turnCount, forKey: .turnCount)
      try container.encode(examples, forKey: .examples)

      // Encode tuples as array of dicts
      let movesArray = moves.map { ["player": $0.playerIndex, "move": $0.moveIndex] }
      try container.encode(movesArray, forKey: .moves)
   }

   public init (from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      gameIndex = try container.decode(Int.self, forKey: .gameIndex)
      seed = try container.decode(UInt64.self, forKey: .seed)
      playerCount = try container.decode(Int.self, forKey: .playerCount)
      winner = try container.decode(Int?.self, forKey: .winner)
      turnCount = try container.decode(Int.self, forKey: .turnCount)
      examples = try container.decode([TrainingExample].self, forKey: .examples)

      let movesArray = try container.decode([[String: Int]].self, forKey: .moves)
      moves = movesArray.map { (playerIndex: $0["player"]!, moveIndex: $0["move"]!) }
   }

   public init (gameIndex: Int, seed: UInt64, playerCount: Int, winner: Int?, turnCount: Int, examples: [TrainingExample], moves: [(playerIndex: Int, moveIndex: Int)]) {
      self.gameIndex = gameIndex
      self.seed = seed
      self.playerCount = playerCount
      self.winner = winner
      self.turnCount = turnCount
      self.examples = examples
      self.moves = moves
   }
}

/// Container for all training data
public struct TrainingDataset: Codable {
   let generatedAt: String
   let modelPath: String?
   let temperature: Float
   let totalGames: Int
   let totalExamples: Int
   let games: [GameData]
}

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
      print("\n" + String(repeating: "=", count: 80))
      print("MOVE STATISTICS")
      print(String(repeating: "=", count: 80))

      let totalWinner = winnerMoves.values.reduce(0, +)
      let totalLoser = loserMoves.values.reduce(0, +)
      let totalTied = tiedMoves.values.reduce(0, +)
      let grandTotal = totalWinner + totalLoser + totalTied

      print("\nOverall Totals:")
      print("  Winner moves: \(totalWinner)")
      print("  Loser moves:  \(totalLoser)")
      print("  Tied moves:   \(totalTied)")
      print("  Grand total:  \(grandTotal)")

      print("\n" + String(repeating: "-", count: 80))
      print("Move Type                 Winners        Losers         Tied")
      print(String(repeating: "-", count: 80))

      for category in MoveCategory.allCases {
         let winnerCount = winnerMoves[category]!
         let loserCount = loserMoves[category]!
         let tiedCount = tiedMoves[category]!

         let winnerPct = totalWinner > 0 ? Float(winnerCount) / Float(totalWinner) * 100.0 : 0.0
         let loserPct = totalLoser > 0 ? Float(loserCount) / Float(totalLoser) * 100.0 : 0.0
         let tiedPct = totalTied > 0 ? Float(tiedCount) / Float(totalTied) * 100.0 : 0.0

         // Completely avoid String(format:...) - just use string interpolation
         let name = category.rawValue.padding(toLength: 25, withPad: " ", startingAt: 0)

         let winnerPctRounded = Int(winnerPct * 10) // For one decimal place
         let winnerLine = "\(winnerCount) (\(winnerPctRounded / 10).\(winnerPctRounded % 10)%)"

         let loserPctRounded = Int(loserPct * 10)
         let loserLine = "\(loserCount) (\(loserPctRounded / 10).\(loserPctRounded % 10)%)"

         let tiedPctRounded = Int(tiedPct * 10)
         let tiedLine = "\(tiedCount) (\(tiedPctRounded / 10).\(tiedPctRounded % 10)%)"

         print("\(name) \(winnerLine.padding(toLength: 13, withPad: " ", startingAt: 0)) \(loserLine.padding(toLength: 13, withPad: " ", startingAt: 0)) \(tiedLine)")
      }
      print(String(repeating: "=", count: 80))
   }
}

public struct DataGenerator {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Data Generator", "g", "game-count", "Number of self-play games to generate")
      opts.addOption("Gameplay Tester", "a", "agent", "Path to model file, or name of non-model-based agent (optional, default is 'random')",
         longDoc:
            "Specifies which agent(s) to use for gameplay. " +
            "If a path to a model file is provided (e.g., 'models/best.mlx'), " +
            "a neural network agent will be loaded from that file. " +
            "Alternatively, you can specify 'random' to use a random agent " +
            "that makes valid moves uniformly at random (default). " +
            "The model file path can be relative to the current working " +
            "directory or an absolute path.")
      opts.addOption("Data Generator", "o", "output", "Output file path for training data (default: training_data/data_TIMESTAMP.json)")
      opts.addOption("Data Generator", "p", "player-count", "Number of players (default: 2)")
      opts.addOption("Data Generator", "s", "seed", "Random seed for game generation (default: random)")
      opts.addOption("Data Generator", "t", "temperature", "Sampling temperature for move selection (default: 1.0, higher = more exploration)")
      opts.addOption("Data Generator", "", "max-turns", "Maximum turns per game before timeout (default: 1000)")
   }


   /// Play a self-play game and collect training examples
   /// - Parameters:
   ///   - playerCount: Number of players in the game
   ///   - agent: The agent to use for all players (self-play)
   ///   - temperature: Temperature for move sampling
   ///   - seed: Random seed for game initialization
   ///   - maxTurns: Maximum turns before timeout
   ///   - gameIndex: Index of this game in the generation batch
   /// - Returns: GameData containing all training examples, or nil if game failed
   static func playGameAndCollectData (
      playerCount: Int,
      agent: any AgentProtocol,
      temperature: Float,
      seed: UInt64,
      maxTurns: Int,
      gameIndex: Int) -> GameData? {

      guard var game = Splendor.Game(playerCount: playerCount, seed: seed) else {
         print("Error: Failed to create game state")
         return nil
      }

      var rng = SeededRandomNumberGenerator(seed: seed)
      var examples: [TrainingExample] = []
      var moves: [(playerIndex: Int, moveIndex: Int)] = []
      var turnCount = 0

      // Play the game and collect examples
      while case .inProgress = game.terminalCondition {
         if turnCount >= maxTurns {
            print("Warning: Game \(gameIndex) reached maximum turn limit (\(maxTurns))")
            return nil  // Discard games that timeout
         }

         let validMoveMask = game.legalMoveMaskForCurrentPlayer()
         let currentPlayer = game.currentPlayer
         let (policyLogits, _) = agent.predict(game: game, currentPlayerIndex: currentPlayer)

         // Sample move with temperature
         guard let (moveIndex, probabilities) = sampleMoveWithTemperature(
            logits: policyLogits,
            validMoveMask: validMoveMask,
            temperature: temperature,
            rng: &rng
         ) else {
            print("Error: No valid moves available for player \(currentPlayer) in game \(gameIndex)")
            return nil
         }

         // Collect training example (value will be assigned after game ends)
         let stateEncoding = game.encoding().map { Float($0) }
         let example = TrainingExample(
            turnNumber: turnCount,
            playerIndex: currentPlayer,
            state: stateEncoding,
            policy: probabilities,
            value: 0.0  // Placeholder, will be updated after game ends
         )
         examples.append(example)

         // Record move for statistics
         moves.append((playerIndex: currentPlayer, moveIndex: moveIndex))

         // Apply the move
         game.applyMove(canonicalMoveIndex: moveIndex)
         turnCount += 1
      }
      precondition(game.terminalCondition != .inProgress, "Game should be terminal")

      // Determine winner and assign value targets
      let winner: Int?
      if case .playerWon(let playerIndex) = game.terminalCondition {
         winner = playerIndex
      } else {
         winner = nil  // Tied game
      }


      // Assign value targets based on outcome
      let examplesWithValues = examples.map { example in
         let value: Float
         if let winnerIndex = winner {
            value = (example.playerIndex == winnerIndex) ? 1.0 : -1.0
         } else {
            value = 0.0  // Tied game
         }
         return TrainingExample(
            turnNumber: example.turnNumber,
            playerIndex: example.playerIndex,
            state: example.state,
            policy: example.policy,
            value: value)
      }

      return GameData(
         gameIndex: gameIndex,
         seed: seed,
         playerCount: playerCount,
         winner: winner,
         turnCount: turnCount,
         examples: examplesWithValues,
         moves: moves)
   }

   public static func main () throws {
      let opts = OptionParser(help: "Generate training data via self-play games")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      // Parse options
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let playerCount = opts.get(option: "player-count", orElse: 2)
      let temperature = opts.get(option: "temperature", orElse: 1.0)
      let maxTurns = opts.get(option: "max-turns", orElse: 1000)
      let baseSeed = opts.get(option: "seed", orElse: UInt64.random(in: 0...UInt64.max))
      let modelPathString = opts.get(option: "model", orElse: "")
      let modelPath = modelPathString.isEmpty ? nil : modelPathString

      // Generate default output path with timestamp
      let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
      let defaultOutputPath = "trainingdata/data_\(timestamp).json"
      let outputPath = opts.get(option: "output", orElse: defaultOutputPath)

      print("Generating training data:")
      print("  Games: \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Temperature: \(temperature)")
      print("  Max turns: \(maxTurns)")
      print("  Base seed: \(baseSeed)")
      print("  Output: \(outputPath)")

      // Initialize agent for self-play using shared function
      // Pass the model path or "uninitialized" as a single-element array
      let agentSpec = modelPath ?? "uninitialized"
      let agents = initializeAgents(playerCount: 1, agentSpecs: [agentSpec], seed: baseSeed)
      let agent = agents[0]  // Extract the single agent for self-play

      // Generate games and collect training data
      var allGameData: [GameData] = []
      var successfulGames = 0
      var statistics = MoveStatistics()

      for gameIndex in 0..<gameCount {
         let gameSeed = baseSeed + UInt64(gameIndex)
         print("Generating game \(gameIndex + 1)/\(gameCount) (seed: \(gameSeed))...")

         if let gameData = playGameAndCollectData(
            playerCount: playerCount,
            agent: agent,
            temperature: temperature,
            seed: gameSeed,
            maxTurns: maxTurns,
            gameIndex: gameIndex) {

            allGameData.append(gameData)
            successfulGames += 1
            print("  Completed: \(gameData.turnCount) turns, \(gameData.examples.count) examples, winner: \(gameData.winner?.description ?? "tied")")

            // Collect statistics from this game
            for (playerIndex, moveIndex) in gameData.moves {
               // Validate move index before categorizing
               guard moveIndex >= 0 && moveIndex < 48 else {
                  print("ERROR: Invalid move index \(moveIndex) from player \(playerIndex) in game \(gameIndex)")
                  continue
               }
               statistics.recordMove(moveIndex: moveIndex, playerIndex: playerIndex, winner: gameData.winner)
            }
         }
         else {
            print("  Failed or timed out, skipping...")
         }
      }

      // Create dataset
      let totalExamples = allGameData.reduce(0) { $0 + $1.examples.count }
      print("\nCreating dataset with \(totalExamples) examples from \(successfulGames) games...")

      let dataset = TrainingDataset(
         generatedAt: ISO8601DateFormatter().string(from: Date()),
         modelPath: modelPath,
         temperature: temperature,
         totalGames: successfulGames,
         totalExamples: totalExamples,
         games: allGameData
      )

      // Save to JSON
      print("Encoding to JSON...")
      let encoder = JSONEncoder()
      encoder.outputFormatting = .prettyPrinted
      let jsonData = try encoder.encode(dataset)

      // Create output directory if it doesn't exist
      print("Writing to \(outputPath)...")
      let outputURL = URL(fileURLWithPath: outputPath)
      let outputDir = outputURL.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true, attributes: nil)

      // Write to file
      try jsonData.write(to: outputURL)
      print("File written successfully.")

      print("\nTraining data generation complete!")
      print("  Successful games: \(successfulGames)/\(gameCount)")
      print("  Total training examples: \(totalExamples)")
      print("  Average examples per game: \(totalExamples / max(successfulGames, 1))")
      print("  Saved to: \(outputPath)")

      // Print move statistics
      print("\nComputing move statistics...")
      statistics.printSummary()
      print("\nDone!")
   }
}
