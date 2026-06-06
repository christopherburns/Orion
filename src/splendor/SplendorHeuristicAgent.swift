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
      var logits = [Float](repeating: 0.0, count: moveCount)
      for i in 0..<moveCount {
         let noise = Float(prng.next()) / Float(UInt64.max) * 0.3
         // Move-type prior bias. Purchases drive scoring; gem-taking is preparation;
         // reserves are situational; discards are forced (illegal unless >10 gems).
         let bias: Float
         switch i {
         case 0..<12:  bias = 2.0   // purchase visible card
         case 12..<15: bias = 2.0   // purchase reserved card
         case 15..<25: bias = 1.0   // take three gems
         case 25..<30: bias = 0.5   // take two gems
         case 30..<42: bias = 0.3   // reserve card
         case 42..<48: bias = 0.0   // discard gem
         default:      bias = 0.0
         }
         logits[i] = bias + noise
      }
      return logits
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