import Utility

public class DumbAgent: AgentProtocol {

   private var prng: SeededRandomNumberGenerator

   public init (prngSeed: UInt64 = 0) {
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

