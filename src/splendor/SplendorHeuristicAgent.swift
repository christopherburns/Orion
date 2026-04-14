import Utility
import Core
import Foundation

public class SplendorHeuristicAgent: AgentProtocol {

   private var prng: SeededRandomNumberGenerator

   public init (prngSeed: UInt64 = 0) {
      self.prng = SeededRandomNumberGenerator(seed: prngSeed)
   }

   public func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      
      let splendorGame = game as? Splendor.Game
      precondition(splendorGame != nil, "SplendorHeuristicAgent can only be used with Splendor.Game")

      let policyLogits = heuristicPolicy(splendorGame: splendorGame!, currentPlayerIndex: currentPlayerIndex)
      let valueEstimate = heuristicValue(splendorGame: splendorGame!, currentPlayerIndex: currentPlayerIndex)

      return (policyLogits, valueEstimate)
   }

   private func heuristicPolicy (splendorGame: Splendor.Game, currentPlayerIndex: Int) -> [Float] {
      let moveCount = splendorGame.canonicalMoveCount
      let policyLogits = (0..<moveCount).map { _ in
         let randomUInt = prng.next()
         return Float(randomUInt) / Float(UInt64.max)
      }

      return policyLogits
   }

   private func heuristicValue (splendorGame: Splendor.Game, currentPlayerIndex: Int) -> Float {

      // This function should return a value between -1 and 1
      // based on the current state of the game
      // 1 indicates confidence the current player will win
      // 0 indicates ambiguity
      // -1 indicates confidence the current player will lose
 
      var scores = [Int](repeating: 0, count: splendorGame.players.count)
      for i in 0..<splendorGame.players.count {
         scores[i] = splendorGame.players[(currentPlayerIndex + i) % splendorGame.players.count].score
      }

      var cardPurchasePowers = [Int](repeating: 0, count: splendorGame.players.count)
      for i in 0..<splendorGame.players.count {
         let index = (currentPlayerIndex+i) % splendorGame.players.count
         cardPurchasePowers[i] = splendorGame.players[index].cardBasedPurchasePower().reduce(0, +)
      }

      // Sum the differences in purchase powers between the current player and all other players
      var scoreLead = 0
      for i in 1..<splendorGame.players.count {
         scoreLead += scores[0] - scores[i]
      }
      
      var purchasePowerLead = 0
      for i in 1..<splendorGame.players.count {
         purchasePowerLead += cardPurchasePowers[0] - cardPurchasePowers[i]
      }

      return tanh(Float(scoreLead * 4 + purchasePowerLead))
   }
}