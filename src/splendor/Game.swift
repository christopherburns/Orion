import Core
import Utility

// Nobles are acquired only with cards, not with gems
public struct Noble {
   public var points: Int
   public var price: [Int] = [0, 0, 0, 0, 0] // Indexed by GemType.rawValue

   public init (points: Int, price: [Int]) {
      self.points = points
      self.price = price
   }

   public static func allNobles () -> [Noble] {
      return [
         Noble(points: 3, price: [0, 0, 0, 4, 4]), // 4 White, 4 Brown
         Noble(points: 3, price: [3, 3, 3, 0, 0]), // 3 Red, 3 Green, 3 Blue
         Noble(points: 3, price: [0, 0, 4, 4, 0]), // 4 Blue, 4 White
         Noble(points: 3, price: [0, 3, 3, 3, 0]), // 3 Green, 3 Blue, 3 White
         Noble(points: 3, price: [4, 0, 0, 0, 4]), // 4 Red, 4 Brown
         Noble(points: 3, price: [4, 4, 0, 0, 0]), // 4 Red, 4 Green
         Noble(points: 3, price: [0, 4, 4, 0, 0]), // 4 Green, 4 Blue
         Noble(points: 3, price: [3, 0, 0, 3, 3]), // 3 Red, 3 White, 3 Brown
         Noble(points: 3, price: [0, 0, 3, 3, 3]), // 3 Blue, 3 White, 3 Brown
         Noble(points: 3, price: [3, 3, 0, 0, 3])  // 3 Red, 3 Green, 3 Brown
      ]
   }
}

public struct PlayerState {
   public var gems: [Int] = [0, 0, 0, 0, 0] // Indexed by GemType.rawValue
   public var goldGems: Int = 0
   public var reservedCards: [Card] = [] // Indexed by tier
   public var cards: [Card] = []
   public var nobles: [Noble] = []

   public init () {}

   public func validate () -> Bool {
      precondition(self.gems.count == GemType.allCases.count, "Player gems must be indexed by GemType")
      precondition(self.reservedCards.count <= 3, "Player can only reserve up to 3 cards")
      precondition(self.cards.count <= 10, "Player can only have up to 10 cards")
      precondition(self.nobles.count <= 3, "Player can only have up to 3 nobles")
      return true
   }

   public var score: Int {
      return self.cards.reduce(0, { $0 + $1.points }) + self.nobles.reduce(0, { $0 + $1.points })
   }

   public var gemCount: Int {
      return self.gems.reduce(0, +) + self.goldGems
   }

   public func purchasePower () -> [Int] {
      var purchasePower = self.gems
      for ownedCard in self.cards {
         purchasePower[ownedCard.color.rawValue] += 1
      }
      return purchasePower
   }

   public func canAfford (cost: [Int]) -> Bool {
      // Check if player can afford the given cost using gems, permanent gems from cards, and gold gems as wildcards
      precondition(cost.count == GemType.allCases.count, "Cost array must match GemType count")

      var totalShortfall = 0
      let purchasePower = self.purchasePower()
      for (gemIndex, price) in cost.enumerated() {
         if purchasePower[gemIndex] < price {
            totalShortfall += price - purchasePower[gemIndex]
         }
      }
      // Gold gems can be used as wildcards to cover any shortfall
      return self.goldGems >= totalShortfall
   }

   public func cardBasedPurchasePower () -> [Int] {
      // Returns card-based purchasing power (cards only, no gems)
      // Counts how many cards the player owns of each color
      var power = Array(repeating: 0, count: GemType.allCases.count)
      for card in self.cards {
         power[card.color.rawValue] += 1
      }
      return power
   }

   // Encode player state as a fixed-size array of Float16
   // Size: 5 (gems) + 1 (goldGems) + 5 (card color counts) + 1 (reserved card count) + 33 (3 reserved cards × 11) + 1 (noble count) + 1 (score) = 47
   public static let ENCODED_SIZE = 47

   public func encoding () -> [Float16] {
      var encoded: [Float16] = []
      encoded.reserveCapacity(PlayerState.ENCODED_SIZE)

      // 5 gem counts (one per GemType), and a count of gold gems
      encoded.append(contentsOf: self.gems.map { Float16($0) / 10.0 })
      encoded.append(Float16(self.goldGems) / 10.0)

      // Record the number of cards owned of each color - this is 5 move values
      var power: [Float16] = Array(repeating: Float16(0), count: GemType.allCases.count)
      for card in self.cards {
         power[card.color.rawValue] += 1.0
      }
      encoded.append(contentsOf: power.map { Float16($0) })

      // reserved card count + 3 reserved cards × 11 floats each (1 point + 5 price + 5 color one-hot)
      encoded.append(Float16(self.reservedCards.count) / 3.0)
      for i in 0..<3 {
         if i < self.reservedCards.count {
            encoded.append(contentsOf: self.reservedCards[i].encoding())
         } else { // Zero-padding for missing reserved cards
            encoded.append(contentsOf: Array(repeating: Float16(0), count: Card.ENCODED_SIZE))
         }
      }

      // Just record the number of nobles owned, up to 5
      encoded.append(Float16(self.nobles.count) / 5.0)

      // record the number of points, total
      encoded.append(Float16(self.score) / Float16(Game.VICTORY_POINTS_THRESHOLD))

      precondition(encoded.count == PlayerState.ENCODED_SIZE, "Encoded size mismatch: expected \(PlayerState.ENCODED_SIZE), got \(encoded.count)")
      return encoded
   }
}

public struct Game: GameProtocol {
   public enum Move {
      case purchaseCard(tier: Int, position: Int) // tier: 0-2, position: 0-3 (top 4 cards)
      case purchaseReservedCard(position: Int) // 0, 1, or 2 (3 reserved cards)
      case takeThreeGems([GemType]) // Three different colored gems
      case takeTwoGems(GemType) // Two of one color
      case reserveCard(tier: Int, position: Int) // Reserve one of the available cards
      case pass // Do nothing, pass turn to next player
   }

   public static let GEMS_PER_PLAYER_LIMIT = 10
   public static let GEM_SUPPLY_LIMIT = 6
   public static let RESERVED_CARDS_PER_PLAYER_LIMIT = 3
   public static let VICTORY_POINTS_THRESHOLD = 15

   // Number of canonical moves:
   //    12 purchase moves (3 tiers × 4 positions)
   //    3 purchase reserved card moves (3 reserved card positions)
   //    10 take three gems moves (combinations of 3 from 5 gem types)
   //    5 take two gems moves (5 possible gem types)
   //    12 reserve moves (3 tiers × 4 positions)
   //    1 pass move
   private static let CANONICAL_MOVE_COUNT = 12 + 3 + 10 + 5 + 12 + 1

   // Three card decks, one for each tier, top four are face up and available to be bought
   public var cardDecks: [[Card]] = [[], [], []]
   public var players: [PlayerState]
   public var supply: [GemType: Int] = [:]
   public var nobles: [Noble] = []
   public var currentPlayer: Int = 0
   public var currentTurn: Int = 0

   // Memoized canonical moves and legal move masks - should always be current and valid
   private var _allMoves: [Move] = []
   private var _legalMoveMasks: [[Bool]] = []

   public init? (playerCount: Int, seed: UInt64 = 0) {
      var allCards = Card.allCards()
      var rng = SeededRandomNumberGenerator(seed: seed)

      // Shuffle each deck
      for tier in 0..<allCards.count {
         allCards[tier].shuffle(using: &rng)
      }

      self.cardDecks = allCards
      self.players = (0..<playerCount).map { _ in PlayerState() }

      if playerCount < 2 || playerCount > 4 {
         return nil
      }

      self.supply = Dictionary(uniqueKeysWithValues: GemType.allCases.map { ($0, Game.GEM_SUPPLY_LIMIT) })

      // Initialize nobles: select playerCount + 1 nobles randomly
      var allNobles = Noble.allNobles()
      allNobles.shuffle(using: &rng)
      self.nobles = Array(allNobles.prefix(playerCount + 1))

      guard self.validate() else { return nil }

      // Initialize memoized values
      self._allMoves = self.generateAllCanonicalMoves()
      self._legalMoveMasks = (0..<playerCount).map { playerIndex in
         self._allMoves.map { move in
            self.isMoveLegal(move, forPlayer: playerIndex)
         }
      }
   }

   private func validate () -> Bool {
      for deck in self.cardDecks {
         for card in deck {
            precondition(card.price.count == GemType.allCases.count, "Card price must be indexed by GemType")
         }
      }
      return true
   }

   // Generate all possible moves in canonical order
   private func generateAllCanonicalMoves () -> [Move] {
      var moves: [Move] = []

      // Purchase moves: 12 possible (3 tiers × 4 positions)
      moves.append(contentsOf: (0..<3).flatMap { tier in
         (0..<4).map { position in Move.purchaseCard(tier: tier, position: position) }
      })

      // Purchase reserved card moves: 3 possible (3 reserved cards)
      moves.append(contentsOf: (0..<3).map { position in Move.purchaseReservedCard(position: position) })

      // Take three different gems: combinations of 3 from 5 gem types
      let gemTypes = Array(GemType.allCases)
      for i in 0..<gemTypes.count {
         for j in (i+1)..<gemTypes.count {
            for k in (j+1)..<gemTypes.count {
               moves.append(Move.takeThreeGems([gemTypes[i], gemTypes[j], gemTypes[k]]))
            }
         }
      }

      // Take two of one color: 5 possible gem types
      moves.append(contentsOf: GemType.allCases.map { Move.takeTwoGems($0) })

      // Reserve moves: 12 possible (3 tiers × 4 positions)
      moves.append(contentsOf: (0..<3).flatMap { tier in
         (0..<4).map { position in Move.reserveCard(tier: tier, position: position) }
      })

      // Pass move: 1 possible (always legal)
      moves.append(Move.pass)

      precondition(moves.count == Game.CANONICAL_MOVE_COUNT, "Canonical move count must be \(Game.CANONICAL_MOVE_COUNT)")
      return moves
   }

   private func isMoveLegal (_ move: Move, forPlayer playerIndex: Int) -> Bool {
      guard playerIndex < players.count else { return false }
      let player = players[playerIndex]

      switch move {
      case .purchaseCard(let tier, let position):
         // Check if card exists at this position and the player can afford it
         guard tier < cardDecks.count && position < cardDecks[tier].count && position < 4 && position >= 0 else { return false }
         return player.canAfford(cost: cardDecks[tier][position].price)

      case .purchaseReservedCard(let position):
         // Check if player has a reserved card at this position
         guard position < player.reservedCards.count else { return false }
         return player.canAfford(cost: player.reservedCards[position].price)

      case .takeThreeGems(let gems):
         // Must be exactly 3 different gems
         guard gems.count == 3 && Set(gems).count == 3 else { return false }
         // Check supply has at least 1 of each
         for gem in gems {
            if supply[gem, default: 0] < 1 {
               return false
            }
         }
         // Check player won't exceed gem limit (typically 10 total)
         return player.gemCount + 3 <= Game.GEMS_PER_PLAYER_LIMIT

      case .takeTwoGems(let gem):
         // Check supply has at least 4 of this gem type, and player won't exceed gem limit
         guard supply[gem, default: 0] >= 4 else { return false }
         return player.gemCount + 2 <= Game.GEMS_PER_PLAYER_LIMIT

      case .reserveCard(let tier, let position):
         // Check if card exists at this position and player has fewer than the limit
         guard tier < cardDecks.count && position < cardDecks[tier].count else { return false }
         return player.reservedCards.count < Game.RESERVED_CARDS_PER_PLAYER_LIMIT

      case .pass:
         // Pass is always legal
         return true
      }
   }


   // Game Protocol Functions

   public var canonicalMoveCount: Int {
      return Game.CANONICAL_MOVE_COUNT
   }

   public func move (atIndex index: Int) -> Move {
      precondition(index >= 0 && index < canonicalMoveCount, "Move index out of bounds")
      return _allMoves[index]
   }

   public func legalMoveMaskForCurrentPlayer () -> [Bool] {
      return self.legalMoveMask(forPlayer: self.currentPlayer)
   }

   public func legalMoveMask (forPlayer playerIndex: Int) -> [Bool] {
      guard playerIndex < _legalMoveMasks.count else {
         return []
      }
      return _legalMoveMasks[playerIndex]
   }

   public var terminalCondition: GameTerminalCondition {
      // Calculate points for all players
      for playerIndex in 0..<players.count {
         if players[playerIndex].score >= Game.VICTORY_POINTS_THRESHOLD {
            return .playerWon(playerIndex: playerIndex)
         }
      }
      return .inProgress
   }

   private mutating func payForCard (card: Card, playerIndex: Int) {
      // Pay for the card using gems and permanent gems from owned cards
      for (gemIndex, price) in card.price.enumerated() {
         if price > 0 {
            let gemType = GemType(rawValue: gemIndex)!
            let permanentGems = players[playerIndex].cards.filter { $0.color == gemType }.count
            let gemsToPay = max(0, price - permanentGems)
            players[playerIndex].gems[gemIndex] -= gemsToPay
            supply[gemType, default: 0] += gemsToPay
         }
      }
   }

   private mutating func awardAvailableNobles (toPlayer playerIndex: Int) {
      // Check if player can afford any noble using only card-based purchasing power
      let cardPower = players[playerIndex].cardBasedPurchasePower()

      // Helper to check if a noble is affordable
      let isAffordable = { (noble: Noble) -> Bool in
         noble.price.enumerated().allSatisfy { (gemIndex, price) in
            cardPower[gemIndex] >= price
         }
      }

      // Filter nobles to find those the player can afford, transfer to the player
      let affordableNobles = nobles.filter(isAffordable)
      players[playerIndex].nobles.append(contentsOf: affordableNobles)
      nobles = nobles.filter { !isAffordable($0) }
   }

   public mutating func applyMove (canonicalMoveIndex: Int) {

      // This function should never be called with an invalidate game state or invalid move,
      // we can check all these conditions with asserts that crash the program if they are violated

      precondition(self.validate(), "Game state is invalid")
      precondition(canonicalMoveIndex < self.canonicalMoveCount, "Canonical move index is out of bounds")
      precondition(canonicalMoveIndex >= 0, "Canonical move index is negative")
      precondition(self.legalMoveMask(forPlayer: currentPlayer)[canonicalMoveIndex], "Move is not legal")

      currentTurn += 1

      let playerIndex = currentPlayer
      guard playerIndex < players.count else {
         preconditionFailure("Invalid player index")
      }

      let move = _allMoves[canonicalMoveIndex]

      switch move {
      case .purchaseCard(let tier, let position):
         let card = cardDecks[tier][position]

         // Pay for the card
         payForCard(card: card, playerIndex: playerIndex)

         // Remove card from deck and add to player
         cardDecks[tier].remove(at: position)
         players[playerIndex].cards.append(card)

         // Check and award any available nobles
         awardAvailableNobles(toPlayer: playerIndex)

      case .purchaseReservedCard(let position):
         precondition(position < players[playerIndex].reservedCards.count, "Invalid reserved card position")
         let card = players[playerIndex].reservedCards[position]

         // Pay for the card
         payForCard(card: card, playerIndex: playerIndex)

         // Move card from reserved cards to owned cards
         players[playerIndex].cards.append(card)
         players[playerIndex].reservedCards.remove(at: position)

         // Check and award any available nobles
         awardAvailableNobles(toPlayer: playerIndex)

      case .takeThreeGems(let gems):
         precondition(players[playerIndex].gemCount + 3 <= Game.GEMS_PER_PLAYER_LIMIT, "Player would exceed gem limit")

         // Take gems from supply
         for gem in gems {
            precondition(supply[gem, default: 0] >= 1, "Insufficient gems in supply")
            supply[gem]! -= 1
            players[playerIndex].gems[gem.rawValue] += 1
         }

      case .takeTwoGems(let gem):
         precondition(supply[gem, default: 0] >= 4, "Insufficient gems in supply")
         precondition(players[playerIndex].gems[gem.rawValue] + 2 <= Game.GEMS_PER_PLAYER_LIMIT, "Player would exceed gem limit")

         // Take 2 gems from supply
         supply[gem]! -= 2
         players[playerIndex].gems[gem.rawValue] += 2

      case .reserveCard(let tier, let position):
         precondition(players[playerIndex].reservedCards.count < Game.RESERVED_CARDS_PER_PLAYER_LIMIT, "Too many reserved cards")

         // Remove card from deck and add to reserved cards
         let card = cardDecks[tier].remove(at: position)
         players[playerIndex].reservedCards.append(card)

         // Give gold gem (if available - simplified, assumes unlimited gold)
         players[playerIndex].goldGems += 1

      case .pass:
         // Do nothing, turn will advance automatically
         break
      }

      // Advance to next player (circular)
      currentPlayer = (currentPlayer + 1) % players.count

      // Recompute legal move masks for all players
      _legalMoveMasks = (0..<players.count).map { playerIndex in
         _allMoves.map { move in
            isMoveLegal(move, forPlayer: playerIndex)
         }
      }
   }

   // Encode game state as a fixed-size array of Float16
   // Size: 188 (4 players × 47) + 5 (supply) + 30 (5 nobles × 6) + 132 (3 tiers × 4 cards × 11) + 4 (current player one-hot) + 1 (turn) = 360
   public static let ENCODED_SIZE = 360

   public func encoding () -> [Float16] {
      var encoded: [Float16] = []

      // 4 players × 167 floats each (zero-padded if player not present)
      for i in 0..<4 {
         if i < self.players.count {
            encoded.append(contentsOf: self.players[i].encoding())
         } else {
            // Zero-padding for missing players
            encoded.append(contentsOf: Array(repeating: Float16(0), count: PlayerState.ENCODED_SIZE))
         }
      }

      // 5 supply gem counts (one per GemType)
      for gemType in GemType.allCases {
         encoded.append(Float16(self.supply[gemType] ?? 0) / 6.0)
      }

      // 5 available nobles × 6 floats each (1 point + 5 price)
      for i in 0..<5 {
         if i < self.nobles.count {
            let noble = self.nobles[i]
            encoded.append(Float16(noble.points)/3.0)
            encoded.append(contentsOf: noble.price.map { Float16($0)/4.0 })
         } else {
            // Zero-padding for missing nobles
            encoded.append(contentsOf: Array(repeating: Float16(0), count: 6))
         }
      }

      // 3 tiers × 4 visible cards × 11 floats each (1 point + 5 price + 5 color one-hot)
      for tier in 0..<3 {
         for position in 0..<4 {
            if tier < self.cardDecks.count && position < self.cardDecks[tier].count {
               encoded.append(contentsOf: self.cardDecks[tier][position].encoding())
            } else {
               // Zero-padding for missing cards
               encoded.append(contentsOf: Array(repeating: Float16(0), count: Card.ENCODED_SIZE))
            }
         }
      }

      // 4 current player one-hot encoding
      var currentPlayerOneHot = Array(repeating: Float16(0), count: 4)
      if self.currentPlayer < 4 {
         currentPlayerOneHot[self.currentPlayer] = 1.0
      }
      encoded.append(contentsOf: currentPlayerOneHot)

      // 1 current turn (normalized, assuming max 100 turns)
      encoded.append(Float16(self.currentTurn) / 100.0)

      precondition(encoded.count == Game.ENCODED_SIZE, "Encoded size mismatch: expected \(Game.ENCODED_SIZE), got \(encoded.count)")
      return encoded
   }

}

