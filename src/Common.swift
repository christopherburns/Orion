import Foundation
import Core
import Splendor
import Utility

/// Initialize agents based on command-line specifications
/// - Parameters:
///   - playerCount: Number of players in the game
///   - agentSpecs: Array of agent specifications (model paths or "random")
///   - seed: Random seed for DumbAgents
/// - Returns: Array of agents, one per player
func initializeAgents (playerCount: Int, agentSpecs: [String], seed: UInt64) -> [any AgentProtocol] {
   var specs: [String] = []

   if agentSpecs.isEmpty {
      // No specs provided, default to random for all players
      specs = Array(repeating: "random", count: playerCount)
   } else if agentSpecs.count == 1 {
      // One spec provided, use it for all players
      specs = Array(repeating: agentSpecs[0], count: playerCount)
   } else if agentSpecs.count == playerCount {
      // Exact match, use each spec for corresponding player
      specs = agentSpecs
   } else {
      print("Error: Number of agent specifications (\(agentSpecs.count)) must be 1 or match player count (\(playerCount))")
      print("Falling back to random agents for all players")
      specs = Array(repeating: "random", count: playerCount)
   }

   var agents: [any AgentProtocol] = []
   for (index, spec) in specs.enumerated() {
      let agent: any AgentProtocol

      if spec.lowercased() == "random" {
         // Use random agent
         agent = DumbAgent(prngSeed: seed + UInt64(index))
         print("Using random agent for player \(index)")
      }
      else if spec.lowercased() == "human" {
         // Use interactive human agent
         agent = HumanAgent()
         print("Using human player for player \(index)")
      }
      else if spec.lowercased() == "heuristic" {
         agent = SplendorHeuristicAgent(prngSeed: seed + UInt64(index))
         print("Using heuristic agent for player \(index)")
      }
      else if spec.lowercased() == "uninitialized" {
         // Use uninitialized neural agent with deterministic seed
         agent = SplendorNeuralAgent(seed: seed + UInt64(index))
         print("Using uninitialized neural agent for player \(index) (seed: \(seed + UInt64(index)))")
      }
      else {
         // Treat as model path
         let modelURL = URL(fileURLWithPath: spec)
         do {
            agent = try SplendorNeuralAgent(url: modelURL)
            print("Loaded neural agent from \(spec) for player \(index)")
         } catch {
            print("Error: Failed to load model from \(spec) for player \(index): \(error)")
            print("Falling back to random agent for player \(index)")
            agent = DumbAgent(prngSeed: seed + UInt64(index))
         }
      }

      agents.append(agent)
   }

   return agents
}

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
   rng: inout SeededRandomNumberGenerator) -> (moveIndex: Int, probabilities: [Float])? {

   precondition(validMoveMask.count == logits.count, "Move mask and logits must have same length")
   precondition(temperature > 0, "Temperature must be positive")

   // Apply temperature scaling
   let scaledLogits = logits.map { $0 / temperature }

   // Mask illegal moves by setting their logits to -infinity
   let maskedLogits = zip(scaledLogits, validMoveMask).map { logit, isValid in
      isValid ? logit : -Float.infinity
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
