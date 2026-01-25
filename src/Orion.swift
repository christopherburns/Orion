
import Swift
import Foundation
import Core
import Splendor
import Utility

// MARK: - Orion CLI Tools
//
// Orion provides three main tools for training and testing neural networks on Splendor:
//
// 1. GENERATE - Generate training data via self-play
//    Usage: orion generate [options]
//    Example: orion generate -g 100 -m models/best.mlx -o training_data/run1.json -t 1.2
//
// 2. TRAIN - Train network on existing training data
//    Usage: orion train [options]
//    Example: orion train -i training_data/ -m models/checkpoint.mlx -o models/improved.mlx -b 256 -e 20
//
// 3. PLAY - Play games for testing and evaluation
//    Usage: orion play [options]
//    Example: orion play -n 100 -m models/best.mlx -m2 models/random
//
// Use --help with any tool to see available options.


struct DataGenerator {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Data Generator", "g", "game-count", "Number of self-play games to generate")
      opts.addOption("Data Generator", "m", "model", "Path to model file in models/ directory (default: untrained)")
      opts.addOption("Data Generator", "o", "output", "Output file path for training data (default: training_data/data_TIMESTAMP.json)")
      opts.addOption("Data Generator", "p", "player-count", "Number of players (default: 2)")
      opts.addOption("Data Generator", "s", "seed", "Random seed for game generation (default: random)")
      opts.addOption("Data Generator", "t", "temperature", "Sampling temperature for move selection (default: 1.0, higher = more exploration)")
      opts.addOption("Data Generator", "max-turns", "max-turns", "Maximum turns per game before timeout (default: 1000)")
   }

   static func main () throws {
      let opts = OptionParser(help: "Generate training data via self-play games")
      self.registerOptions(opts: opts)
      // parse the command line arguments, now that all options are registered
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let gameCount = opts.get(option: "game-count", orElse: 1)
      print("Generating training data for \(gameCount) games")
      for gameIndex in 0..<gameCount {
         print("Generating training data for game \(gameIndex+1) of \(gameCount)")
         /// ...
      }
   }
}

struct NetworkTrainer {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Network Trainer", "i", "input", "Input training data file or directory (required)")
      opts.addOption("Network Trainer", "m", "model", "Path to input model file to continue training (default: create new untrained model)")
      opts.addOption("Network Trainer", "o", "output", "Output path for trained model (default: models/model_TIMESTAMP.mlx)")
      opts.addOption("Network Trainer", "s", "seed", "Random seed for reproducibility (default: random)")
      opts.addOption("Network Trainer", "b", "batch-size", "Batch size for training (default: 256)")
      opts.addOption("Network Trainer", "e", "epochs", "Number of training epochs (default: 10)")
      opts.addOption("Network Trainer", "lr", "learning-rate", "Learning rate (default: 0.001)")
      opts.addOption("Network Trainer", "v", "validation-split", "Fraction of data to use for validation (default: 0.1)")
      opts.addOption("Network Trainer", "save-interval", "save-interval", "Save checkpoint every N epochs (default: 5)")
      opts.addOption("Network Trainer", "opt", "optimizer", "Optimizer: adam, sgd (default: adam)")
      opts.addOption("Network Trainer", "ploss", "policy-loss-weight", "Weight for policy loss (default: 1.0)")
      opts.addOption("Network Trainer", "vloss", "value-loss-weight", "Weight for value loss (default: 1.0)")
   }

   static func main () throws {
      let opts = OptionParser(help: "Train a neural network model on training data")
      self.registerOptions(opts: opts)
      // parse the command line arguments, now that all options are registered
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let gameCount = opts.get(option: "game-count", orElse: 1)
      print("Training network for \(gameCount) games")
   }
}



struct GameplayTester {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Gameplay Tester", "s", "seed", "Seed for random number generator (default: 42)")
      opts.addOption("Gameplay Tester", "n", "game-count", "Number of games to play (default: 1)")
      opts.addOption("Gameplay Tester", "p", "player-count", "Number of players [2-4] (default: 2)")

      opts.addOption("Gameplay Tester", "a", "agents", "Path to model file, or name of non-model-based agent (optional, default is 'random')",
         longDoc:
            "Specifies which agent(s) to use for gameplay. " +
            "If a path to a model file is provided (e.g., 'models/best.mlx'), " +
            "a neural network agent will be loaded from that file. " +
            "Alternatively, you can specify 'random' to use a random agent " +
            "that makes valid moves uniformly at random (default). " +
            "The model file path can be relative to the current working " +
            "directory or an absolute path. The number of arguments must " +
            "match the number of players, or if one is supplied it will be " +
            "instanced for all players.")

      opts.addOption("Gameplay Tester", "v", "verbose", "Show detailed game output even for multiple games (default: auto)")
      opts.addOption("Gameplay Tester", "", "show-probabilities", "Show move probability distribution for each turn (default: false)")
      opts.addOption("Gameplay Tester", "", "max-turns", "Maximum turns per game before timeout (default: 1000)")
   }

   /// Convert logits to probabilities by masking illegal moves and applying softmax
   /// - Parameters:
   ///   - logits: Raw network output logits for all moves
   ///   - validMoveMask: Boolean mask indicating which moves are legal
   /// - Returns: Probability distribution over all moves (illegal moves have probability 0), or nil if no legal moves
   static func computeMoveProbabilities (logits: [Float], validMoveMask: [Bool]) -> [Float]? {
      precondition(validMoveMask.count == logits.count, "Move mask and logits must have same length")

      // Mask illegal moves by setting their logits to -infinity
      var maskedLogits = logits
      for (index, isValid) in validMoveMask.enumerated() {
         if !isValid {
            maskedLogits[index] = -Float.infinity
         }
      }

      // Apply softmax to get probabilities
      // First find max for numerical stability
      let maxLogit = maskedLogits.max() ?? -Float.infinity
      guard maxLogit.isFinite else {
         // All moves are illegal
         return nil
      }

      // Compute exp(logit - maxLogit) for numerical stability
      let expScores = maskedLogits.map { exp($0 - maxLogit) }
      let sumExp = expScores.reduce(0.0, +)

      // Return normalized probabilities
      return expScores.map { $0 / sumExp }
   }

   /// Sample a move using greedy selection (argmax)
   /// - Parameters:
   ///   - validMoveMask: Boolean mask indicating which moves are legal
   ///   - movePreferences: Raw logits from the neural network
   /// - Returns: Index of the best move, or nil if no legal moves
   static func sampleMove (validMoveMask: [Bool], movePreferences: [Float]) -> Int? {
      guard let probabilities = computeMoveProbabilities(logits: movePreferences, validMoveMask: validMoveMask) else {
         return nil
      }

      // For now, use greedy selection (argmax)
      // TODO: Add temperature-based sampling for training
      return probabilities.enumerated().max(by: { $0.1 < $1.1 })?.0
   }

   static func showGameState (game: Splendor.Game) {
      GamePrinter.present(game)
      for (index, player) in game.players.enumerated() {
         GamePrinter.presentPlayer(player, playerIndex: index)
      }
   }

   /// Initialize agents based on command-line specifications
   /// - Parameters:
   ///   - playerCount: Number of players in the game
   ///   - agentSpecs: Array of agent specifications (model paths or "random")
   ///   - seed: Random seed for DumbAgents
   /// - Returns: Array of agents, one per player
   static func initializeAgents (playerCount: Int, agentSpecs: [String], seed: UInt64) -> [any AgentProtocol] {
      var specs: [String] = []

      if agentSpecs.isEmpty {
         // No specs provided, default to random for all players
         specs = Array(repeating: "random", count: playerCount)
      } else if agentSpecs.count == 1 {
         // One spec provided, use it for all players
         specs = Array(repeating: agentSpecs[0], count: playerCount)
      } else if agentSpecs.count == playerCount {
         // Exact match, use each spec for corresponding player
         specs = agentSpecs
      } else {
         print("Error: Number of agent specifications (\(agentSpecs.count)) must be 1 or match player count (\(playerCount))")
         print("Falling back to random agents for all players")
         specs = Array(repeating: "random", count: playerCount)
      }

      var agents: [any AgentProtocol] = []
      for (index, spec) in specs.enumerated() {
         let agent: any AgentProtocol

         if spec.lowercased() == "random" {
            // Use random agent
            agent = DumbAgent(prngSeed: seed + UInt64(index))
            print("Using random agent for player \(index)")
         }
         else if spec.lowercased() == "uninitialized" {
            // Use uninitialized neural agent with deterministic seed
            agent = SplendorNeuralAgent(seed: seed + UInt64(index))
            print("Using uninitialized neural agent for player \(index) (seed: \(seed + UInt64(index)))")
         }
         else {
            // Treat as model path
            let modelURL = URL(fileURLWithPath: spec)
            do {
               agent = try SplendorNeuralAgent(url: modelURL)
               print("Loaded neural agent from \(spec) for player \(index)")
            } catch {
               print("Error: Failed to load model from \(spec) for player \(index): \(error)")
               print("Falling back to random agent for player \(index)")
               agent = DumbAgent(prngSeed: seed + UInt64(index))
            }
         }

         agents.append(agent)
      }

      return agents
   }

   static func playGame (playerCount: Int, silence: Bool, seed: UInt64, agents: [any AgentProtocol]) -> (GameTerminalCondition, Int) {

      precondition(agents.count == playerCount, "Number of agents must match player count")

      guard var g = Splendor.Game(playerCount: playerCount, seed: seed) else {
         print("Error: Failed to create game state")
         return (.inProgress, 0)
      }

      if !silence {
         showGameState(game: g)
      }

      // Track move statistics
      var moveTypeCounts: [String: Int] = [
         "purchase": 0,
         "purchaseReserved": 0,
         "takeThreeGems": 0,
         "takeTwoGems": 0,
         "reserve": 0,
         "discard": 0
      ]

      // Game loop
      var turnCount = 0
      let maxTurns = 1000 // Prevent infinite loops with untrained networks

      while case .inProgress = g.terminalCondition {
         if turnCount >= maxTurns {
            print("Warning: Game reached maximum turn limit (\(maxTurns))")
            break
         }

         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let currentAgent = agents[g.currentPlayer]
         let (policyLogits, _) = currentAgent.predict(game: g, currentPlayerIndex: g.currentPlayer)

         if !silence {
            if let probabilities = computeMoveProbabilities(logits: policyLogits, validMoveMask: validMoveMask) {
               GamePrinter.presentMoveProbabilities(probabilities, game: g, topN: 10)
            }
         }

         guard let moveIndex = self.sampleMove(validMoveMask: validMoveMask, movePreferences: policyLogits) else {
            print("Error: No valid moves available for player \(g.currentPlayer)")
            print("Valid move mask: \(validMoveMask)")
            print("All false? \(validMoveMask.allSatisfy { !$0 })")
            print("   Game phase: \(g.phase)")
            print("   Game state:")
            print("      Players:")
            GamePrinter.present(g)
            GamePrinter.presentPlayer(g.players[g.currentPlayer], playerIndex: g.currentPlayer)
            preconditionFailure("No valid moves available for player \(g.currentPlayer)")
            break
         }

         // Track move type
         if moveIndex < 12 {
            moveTypeCounts["purchase"]! += 1
         } else if moveIndex < 15 {
            moveTypeCounts["purchaseReserved"]! += 1
         } else if moveIndex < 25 {
            moveTypeCounts["takeThreeGems"]! += 1
         } else if moveIndex < 30 {
            moveTypeCounts["takeTwoGems"]! += 1
         } else if moveIndex < 42 {
            moveTypeCounts["reserve"]! += 1
         } else {
            moveTypeCounts["discard"]! += 1
         }

         g.applyMove(canonicalMoveIndex: moveIndex)
         turnCount += 1

         if !silence {
            GamePrinter.presentMove(moveIndex: moveIndex, game: g)
            showGameState(game: g)
         }
      }

      // Print move statistics
      if !silence {
         print("\n\u{001B}[1mMove Statistics:\u{001B}[0m")
         for (moveType, count) in moveTypeCounts.sorted(by: { $0.key < $1.key }) {
            let percentage = Float(count) / Float(turnCount) * 100.0
            print("  \(moveType): \(count) (\(String(format: "%.1f", percentage))%)")
         }
      }

      return (g.terminalCondition, turnCount)
   }


   static func main () throws {
      print("Hello Orion!")

      ////////////////////////////////////////
      // Set up Command Line Option Parsing //
      ////////////////////////////////////////

      let opts = OptionParser(help: "Play Splendor games using neural network or random agents")
      self.registerOptions(opts: opts)
      // parse the command line arguments, now that all options are registered
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)


      let playerCount = opts.get(option: "player-count", orElse: 2)
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let seed = opts.get(option: "seed", orElse: UInt64(42))
      let agentSpecs = opts.getAll(option: "agents", as: String.self)

      let silence = gameCount > 1

      // Initialize agents based on command-line specifications
      let agents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: seed)

      // Print summary of agent types
      print("Playing \(gameCount) games with \(playerCount) players")
      if agentSpecs.isEmpty || agentSpecs.count == 1 {
         // All players use the same agent
         let spec = agentSpecs.isEmpty ? "random" : agentSpecs[0]
         print("  All players: \(spec)")
      } else {
         // Different agents per player
         for (index, spec) in agentSpecs.enumerated() {
            print("  Player \(index): \(spec)")
         }
      }

      var gameResults: [(GameTerminalCondition, Int)] = []
      var totalTurnCount = 0
      var playerWinCounts: [Int: Int] = [:]
      var tiedCount = 0
      for gameIndex in 0..<gameCount {
         let (terminalCondition, turnCount) = self.playGame(playerCount: playerCount, silence: silence, seed: seed+UInt64(gameIndex), agents: agents)
         print("Completed game \(gameIndex) in \(turnCount) turns, \(terminalCondition)")
         gameResults.append((terminalCondition, turnCount))
         totalTurnCount += turnCount
         switch terminalCondition {
         case .playerWon(let playerIndex):
            playerWinCounts[playerIndex, default: 0] += 1
         case .tied:
            tiedCount += 1
         case .inProgress:
            break
         }
      }

      print("Total turns: \(totalTurnCount)")
      print("Player win counts: \(playerWinCounts)")
      print("Tied count: \(tiedCount)")

      for (index, count) in playerWinCounts {
         print("Player \(index) won \(count) games (\(Float(count) / Float(gameCount) * 100.0)%)")
      }

      print("Tied games: \(tiedCount)")
   }
}


@main
struct Orion {
   static func main () throws {

      if CommandLine.arguments.count == 1 {
         print("Usage: orion <command> [options]")
         print("Commands:")
         print("  train - Train a neural network model")
         print("  play - Play games using a neural network model")
         print("  generate - Generate training data for a neural network model")
         return
      }

      if CommandLine.arguments[1] == "train" {
         try NetworkTrainer.main()
      }
      else if CommandLine.arguments[1] == "play" {
         try GameplayTester.main()
      }
      else if CommandLine.arguments[1] == "generate" {
         try DataGenerator.main()
      }
      else {
         print("Unknown command: \(CommandLine.arguments[1])")
      }
   }
}
