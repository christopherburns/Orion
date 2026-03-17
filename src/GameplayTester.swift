import Swift
import Foundation
import Core
import Splendor
import Utility

public struct GameplayTester {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Gameplay Tester", "s", "seed", "Seed for random number generator (default: 42)")
      opts.addOption("Gameplay Tester", "n", "game-count", "Number of games to play (default: 1)")
      opts.addOption("Gameplay Tester", "p", "player-count", "Number of players [2-4] (default: 2)")

      opts.addOption("Gameplay Tester", "a", "agent", "Path to model file, or name of non-model-based agent (optional, default is 'random')",
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

      opts.addOption("Gameplay Tester", "t", "temperature", "Sampling temperature for move selection (0 = greedy, default: 0)")
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

   static func playGame (playerCount: Int, silence: Bool, seed: UInt64, agents: [any AgentProtocol], temperature: Float = 0, maxTurns: Int = 1000) -> (GameTerminalCondition, Int) {

      precondition(agents.count == playerCount, "Number of agents must match player count")

      var rng = SeededRandomNumberGenerator(seed: seed)

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
      var timedOut = false
      while case .inProgress = g.terminalCondition {
         if g.currentTurn >= maxTurns {
            timedOut = true
            break
         }

         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let currentAgent = agents[g.currentPlayer]
         let (policyLogits, _) = currentAgent.predict(game: g, currentPlayerIndex: g.currentPlayer)

         let moveResult = temperature > 0
            ? sampleMoveWithTemperature(logits: policyLogits, validMoveMask: validMoveMask, temperature: temperature, rng: &rng)
            : sampleMove(validMoveMask: validMoveMask, movePreferences: policyLogits).map { ($0, [Float]()) }
         guard let (moveIndex, _) = moveResult else {
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

         if !silence {
            GamePrinter.presentMove(moveIndex: moveIndex, game: g)
            showGameState(game: g)
         }
      }

      // Print move statistics
      if !silence {
         print("\n\u{001B}[1mMove Statistics:\u{001B}[0m")
         for (moveType, count) in moveTypeCounts.sorted(by: { $0.key < $1.key }) {
            let percentage = Float(count) / Float(g.currentTurn) * 100.0
            print("  \(moveType): \(count) (\(String(format: "%.1f", percentage))%)")
         }
      }

      return (timedOut ? .timedOut : g.terminalCondition, g.currentTurn)
   }


   public static func main () throws {
      let opts = OptionParser(help: "Play Splendor games using neural network or random agents")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let playerCount = opts.get(option: "player-count", orElse: 2)
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let seed = opts.get(option: "seed", orElse: UInt64(42))
      let agentSpecs = opts.getAll(option: "agent", as: String.self)
      let temperature = opts.get(option: "temperature", orElse: Float(0))
      let maxTurns = opts.get(option: "max-turns", orElse: 1000)

      // Print configuration
      let agentDesc = agentSpecs.isEmpty ? "random" : agentSpecs.joined(separator: ", ")
      print("Configuration:")
      print("  Games:            \(gameCount)")
      print("  Players per game: \(playerCount)")
      print("  Agent:            \(agentDesc)")
      print("  Temperature:      \(String(format: "%.2f", temperature))")
      print("  Max turns:        \(maxTurns)")
      print("  Seed:             \(seed)")

      let silence = gameCount > 1

      // Initialize agents based on command-line specifications
      let agents = initializeAgents(playerCount: playerCount, agentSpecs: agentSpecs, seed: seed)

      var gameResults: [(GameTerminalCondition, Int)] = []
      var totalTurnCount = 0
      var playerWinCounts: [Int: Int] = [:]
      var tiedCount = 0
      var timedOutCount = 0
      for gameIndex in 0..<gameCount {

         if gameIndex % 100 == 0 {
            print("Playing game \(gameIndex + 1)/\(gameCount) (seed: \(seed+UInt64(gameIndex)))...")
         }
         
         let (terminalCondition, turnCount) = self.playGame(playerCount: playerCount, silence: silence, seed: seed+UInt64(gameIndex), agents: agents, temperature: temperature, maxTurns: maxTurns)
         gameResults.append((terminalCondition, turnCount))
         totalTurnCount += turnCount
         switch terminalCondition {
         case .playerWon(let playerIndex):
            playerWinCounts[playerIndex, default: 0] += 1
         case .tied:
            tiedCount += 1
         case .timedOut:
            timedOutCount += 1
         case .inProgress:
            break
         }
      }

      let concludedCount = gameCount - timedOutCount
      print("Total turns: \(totalTurnCount)")
      print("Average turns/game: \(String(format: "%.1f", Float(totalTurnCount)/Float(gameCount)))")

      for (index, count) in playerWinCounts.sorted(by: { $0.key < $1.key }) {
         let spec = index < agentSpecs.count ? agentSpecs[index] : "random"
         print("Player \(index) (\(spec)) won \(count) games (\(String(format: "%.1f", Float(count) / Float(concludedCount) * 100.0))%)")
      }
      print("Tied games: \(tiedCount)")
      if timedOutCount > 0 {
         print("Timed out:  \(timedOutCount) games (exceeded \(maxTurns) turns)")
      }
   }
}
