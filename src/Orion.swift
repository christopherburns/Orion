
import Swift
import Foundation
import Core
import Splendor
import Utility


class DumbAgent: AgentProtocol {

   private var prng: SeededRandomNumberGenerator

   init (prngSeed: UInt64 = 0) {
      self.prng = SeededRandomNumberGenerator(seed: prngSeed)
   }

   public func calculateMovePreferences (game: any GameProtocol, currentPlayerIndex: Int) -> [Float] {
      let moveCount = game.canonicalMoveCount
      return (0..<moveCount).map { _ in
         let randomUInt = prng.next()
         return Float(randomUInt) / Float(UInt64.max)
      }
   }
}



@main
struct Orion {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("General", "s", "seed", "Seed for random number generator")
      opts.addOption("General", "p", "player-count", "Number of players")
      opts.addOption("General", "n", "game-count", "Number of games to play")
   }

   static func sampleMove (validMoveMask: [Bool], movePreferences: [Float]) -> Int? {
      precondition(validMoveMask.count == movePreferences.count, "Move mask and preferences must have same length")

      var bestIndex: Int? = nil
      var bestScore: Float = Float.leastNormalMagnitude

      for (index, isValid) in validMoveMask.enumerated() {
         if isValid {
            let score = movePreferences[index]
            if score > bestScore {
               bestScore = score
               bestIndex = index
            }
         }
      }

      return bestIndex
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
      let agent = DumbAgent(prngSeed: seed)

      // Game loop
      var turnCount = 0
      while case .inProgress = g.terminalCondition {
         let validMoveMask = g.legalMoveMaskForCurrentPlayer()
         let movePreferences = agent.calculateMovePreferences(game: g, currentPlayerIndex: g.currentPlayer)

         guard let moveIndex = self.sampleMove(validMoveMask: validMoveMask, movePreferences: movePreferences) else {
            print("Error: No valid moves available for player \(g.currentPlayer)")
            break
         }

         g.applyMove(canonicalMoveIndex: moveIndex)
         turnCount += 1

         if !silence {
            GamePrinter.presentMove(moveIndex: moveIndex, game: g)
            showGameState(game: g)
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

