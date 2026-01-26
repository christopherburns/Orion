import Foundation
import Utility

/// Apply temperature scaling and masked softmax to logits, then sample a move
/// - Parameters:
///   - logits: Raw network output logits for all moves
///   - validMoveMask: Boolean mask indicating which moves are legal
///   - temperature: Temperature parameter for exploration (higher = more random)
///   - rng: Random number generator for sampling
/// - Returns: Tuple of (sampled move index, probability distribution), or nil if no legal moves
func sampleMoveWithTemperature (
   logits: [Float],
   validMoveMask: [Bool],
   temperature: Float,
   rng: inout SeededRandomNumberGenerator
) -> (moveIndex: Int, probabilities: [Float])? {

   precondition(validMoveMask.count == logits.count, "Move mask and logits must have same length")
   precondition(temperature > 0, "Temperature must be positive")

   // Apply temperature scaling
   let scaledLogits = logits.map { $0 / temperature }

   // Mask illegal moves by setting their logits to -infinity
   var maskedLogits = scaledLogits
   for (index, isValid) in validMoveMask.enumerated() {
      if !isValid {
         maskedLogits[index] = -Float.infinity
      }
   }

   // Apply softmax to get probabilities
   let maxLogit = maskedLogits.max() ?? -Float.infinity
   guard maxLogit.isFinite else {
      return nil  // All moves are illegal
   }

   let expScores = maskedLogits.map { exp($0 - maxLogit) }
   let sumExp = expScores.reduce(0.0, +)
   let probabilities = expScores.map { $0 / sumExp }

   // Sample from the probability distribution
   let randomValue = Float(rng.next()) / Float(UInt64.max)
   var cumulativeProbability: Float = 0.0
   for (index, probability) in probabilities.enumerated() {
      cumulativeProbability += probability
      if randomValue <= cumulativeProbability {
         return (index, probabilities)
      }
   }

   // Fallback: return highest probability move (shouldn't happen with proper sampling)
   let maxIndex = probabilities.enumerated().max(by: { $0.1 < $1.1 })!.0
   return (maxIndex, probabilities)
}
