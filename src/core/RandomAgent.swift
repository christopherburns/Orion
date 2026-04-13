import Utility

public class DumbAgent: AgentProtocol {

   private var prng: SeededRandomNumberGenerator

   public init (prngSeed: UInt64 = 0) {
      self.prng = SeededRandomNumberGenerator(seed: prngSeed)
   }

   public func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      let moveCount = game.canonicalMoveCount
      let policyLogits = (0..<moveCount).map { _ in
         let randomUInt = prng.next()
         return Float(randomUInt) / Float(UInt64.max)
      }

      // Flip a coin to decide win/loss
      let valueEstimate: Float = Float(prng.next()) / Float(UInt64.max) > 0.5 ? 1.0 : 0.0 
      
      return (policyLogits, valueEstimate)
   }
}
