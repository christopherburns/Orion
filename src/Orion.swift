
import Swift
import Foundation
import Core
import Splendor
import Utility

@main
struct Orion {

   static func registerOptions (opts: OptionParser) {

      opts.addOption("General", "s", "seed", "Seed for random number generator")
      opts.addOption("General", "p", "player-count", "Number of players")
      opts.addOption("General", "n", "game-count", "Number of games to play")
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

   static func playGame (playerCount: Int, silence: Bool, seed: UInt64) -> (GameTerminalCondition, Int) {

      guard var g = Splendor.Game(playerCount: playerCount) else {
         print("Error: Failed to create game state")
         return (.inProgress, 0)
      }

      if !silence {
         showGameState(game: g)
      }

      // Instantiate agent
      //let agent = DumbAgent(prngSeed: seed)
      let agent = SplendorNeuralAgent()

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
         let (policyLogits, _) = agent.predict(game: g, currentPlayerIndex: g.currentPlayer)

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

      let opts = OptionParser()
      self.registerOptions(opts: opts)
      // parse the command line arguments, now that all options are registered
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)


      let playerCount = opts.get(option: "player-count", orElse: 2)
      let gameCount = opts.get(option: "game-count", orElse: 1)
      let seed = opts.get(option: "seed", orElse: UInt64(42))

      let silence = gameCount > 1

      print("Playing \(gameCount) games with \(playerCount) players")
      var gameResults: [(GameTerminalCondition, Int)] = []
      var totalTurnCount = 0
      var playerWinCounts: [Int: Int] = [:]
      var tiedCount = 0
      for gameIndex in 0..<gameCount {
         print("Playing game \(gameIndex+1) of \(gameCount)")
         let (terminalCondition, turnCount) = self.playGame(playerCount: playerCount, silence: silence, seed: seed+UInt64(gameIndex))
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

