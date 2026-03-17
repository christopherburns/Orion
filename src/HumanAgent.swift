import Foundation
import Core
import Splendor
import Utility

/// An agent that represents a human player. Move selection is handled externally
/// (by the interactive game loop in GameplayTester), so predict() is never called
/// during interactive play. isHuman signals the game loop to take over input.
public struct HumanAgent: AgentProtocol {
   public init () {}

   public var isHuman: Bool { true }

   public func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      // Returns uniform logits — not used in interactive mode
      return (Array(repeating: 0.0, count: game.canonicalMoveCount), 0.0)
   }
}
