
public enum GameTerminalCondition {
   case playerWon(playerIndex: Int)
   case tied
   case inProgress
}


public protocol GameProtocol {

   associatedtype Move

   // A canonical move is a move that is guaranteed to be valid for at least one valid
   // game state, but not necessarily any particular game state. Every game defines a
   // canonical ordering of these moves.
   var canonicalMoveCount: Int { get }

   var currentTurn: Int { get }
   var currentPlayer: Int { get }

   // Given the present game state, compute a boolean array of length canonicalMoveCount
   // where each element is true if the move is legal for the given player
   func legalMoveMaskForCurrentPlayer () -> [Bool]
   func legalMoveMask (forPlayer playerIndex: Int) -> [Bool]

   // Apply a move to the game state. Move must be a legal and valid move for the
   // current game state. It is defined by the index in the game's canonical move ordering.
   mutating func applyMove (canonicalMoveIndex: Int)

   // Return the terminal condition of the game
   var terminalCondition: GameTerminalCondition { get }
}

