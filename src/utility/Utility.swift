
// Simple seeded random number generator
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
   private var state: UInt64

   public init (seed: UInt64) {
      self.state = seed
   }

   public mutating func next () -> UInt64 {
      // Linear congruential generator
      state = state &* 1103515245 &+ 12345
      return state
   }
}
