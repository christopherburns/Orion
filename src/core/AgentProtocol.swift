
public protocol AgentProtocol {
   /// Given a game state and player index, returns a logits vector over all canonical moves.
   /// The returned array must have length equal to game.canonicalMoveCount.
   /// The logits do not need to sum to one, or be limited to legal moves
   ///
   /// - Parameters:
   ///   - game: The current game state
   ///   - playerIndex: The index of the player whose turn it is
   /// - Returns: Tuple of (policyLogits, valueEstimate) where policyLogits is an
   ///   array of probabilities, one for each canonical move, and valueEstimate is a float between -1 and 1
   func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float)

   /// True for human players that read moves interactively from stdin. Default is false.
   var isHuman: Bool { get }
}

extension AgentProtocol {
   public var isHuman: Bool { false }
}

