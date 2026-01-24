
public protocol AgentProtocol {
   /// Given a game state and player index, returns a logits vector over all canonical moves.
   /// The returned array must have length equal to game.canonicalMoveCount.
   /// The logits do not need to sum to one, or be limited to legal moves
   ///
   /// - Parameters:
   ///   - game: The current game state
   ///   - playerIndex: The index of the player whose turn it is
   /// - Returns: Array of probabilities, one for each canonical move
   func calculateMovePreferences (game: any GameProtocol, currentPlayerIndex: Int) -> [Float]
}

