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

      let valueEstimate: Float = 0.0 // No opinion on the value of this board position
      return (policyLogits, valueEstimate)
   }
}

