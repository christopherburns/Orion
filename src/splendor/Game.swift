import Core
import Utility
import Foundation

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

   // cards/nobles are mutated only through addCard/addNobles so the cached
   // cardPower and score stay in sync — these are read on every legal-move
   // check and terminal test, so they must not require scanning the arrays.
   public private(set) var cards: [Card] = []
   public private(set) var nobles: [Noble] = []
   public private(set) var cardPower: [Int] = [0, 0, 0, 0, 0] // Indexed by GemType.rawValue
   public private(set) var score: Int = 0

   public init () {}

   public mutating func addCard (_ card: Card) {
      cards.append(card)
      cardPower[card.color.rawValue] += 1
      score += card.points
   }

   public mutating func addNobles (_ newNobles: [Noble]) {
      nobles.append(contentsOf: newNobles)
      for noble in newNobles {
         score += noble.points
      }
   }

   public func validate () -> Bool {
      precondition(self.gems.count == GemType.allCases.count, "Player gems must be indexed by GemType")
      precondition(self.reservedCards.count <= 3, "Player can only reserve up to 3 cards")
      precondition(self.cards.count <= 10, "Player can only have up to 10 cards")
      precondition(self.nobles.count <= 3, "Player can only have up to 3 nobles")
      return true
   }

   public var gemCount: Int {
      return self.gems.reduce(0, +) + self.goldGems
   }

   public func purchasePower () -> [Int] {
      var purchasePower = self.gems
      for i in 0..<purchasePower.count {
         purchasePower[i] += self.cardPower[i]
      }
      return purchasePower
   }

   public func canAfford (cost: [Int]) -> Bool {
      // Check if player can afford the given cost using gems, permanent gems from cards, and gold gems as wildcards
      assert(cost.count == GemType.allCases.count, "Cost array must match GemType count")

      var totalShortfall = 0
      for gemIndex in 0..<cost.count {
         let shortfall = cost[gemIndex] - self.gems[gemIndex] - self.cardPower[gemIndex]
         if shortfall > 0 {
            totalShortfall += shortfall
         }
      }
      // Gold gems can be used as wildcards to cover any shortfall
      return self.goldGems >= totalShortfall
   }

   public func cardBasedPurchasePower () -> [Int] {
      // Card-based purchasing power (cards only, no gems), maintained incrementally
      return self.cardPower
   }

   // Encode player state as a fixed-size array of Float16
   // Size: 5 (gems) + 1 (goldGems) + 1 (gem headroom) + 5 (card color counts) + 1 (reserved card count) + 36 (3 reserved cards × 12) + 1 (noble count) + 1 (score) = 51
   public static let ENCODED_SIZE = 51

   public func encoding () -> [Float16] {
      var encoded: [Float16] = []
      encoded.reserveCapacity(PlayerState.ENCODED_SIZE)

      // 5 gem counts (one per GemType), and a count of gold gems
      encoded.append(contentsOf: self.gems.map { Float16($0) / 10.0 })
      encoded.append(Float16(self.goldGems) / 10.0)

      // Gem headroom - how many more gems until the player will face a discard penalty
      // This surfaces a meaningful correlate to bad performance that would otherwise need
      // to be inferred from the sum of six other properties
      encoded.append(Float16(10.0 - Float(self.gemCount)) / 10.0)

      // Record the number of cards owned of each color - this is 5 more values
      encoded.append(contentsOf: self.cardPower.map { Float16($0) / 7.0 })

      // reserved card count + 3 reserved cards × 11 floats each (1 point + 5 price + 5 color one-hot)
      encoded.append(Float16(self.reservedCards.count) / 3.0)
      for i in 0..<3 {
         if i < self.reservedCards.count {
            let affordable = self.canAfford(cost: self.reservedCards[i].price)
            encoded.append(contentsOf: self.reservedCards[i].encoding(affordable: affordable))
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
      case discardGem(GemType) // Discard a gem (used when over 10 gem limit)
      case discardGoldGem // Discard a gold gem (used when over 10 gem limit)
   }

   public enum GamePhase {
      case normalAction  // Player takes their main action
      case discarding    // Player must discard gems to reach 10 gem limit
   }

   public static let GEMS_PER_PLAYER_LIMIT = 10
   public static let RESERVED_CARDS_PER_PLAYER_LIMIT = 3
   public static let VICTORY_POINTS_THRESHOLD = 15

   // Number of canonical moves:
   //    12 purchase moves (3 tiers × 4 positions)
   //    3 purchase reserved card moves (3 reserved card positions)
   //    10 take three gems moves (combinations of 3 from 5 gem types)
   //    5 take two gems moves (5 possible gem types)
   //    12 reserve moves (3 tiers × 4 positions)
   //    6 discard gem moves (6 gem types including gold)
   public static let CANONICAL_MOVE_COUNT = 12 + 3 + 10 + 5 + 12 + 6

   // Three card decks, one for each tier, top four are face up and available to be bought
   public var cardDecks: [[Card]] = [[], [], []]
   public var players: [PlayerState]
   public var supply: [Int] = [0, 0, 0, 0, 0] // Indexed by GemType.rawValue
   public var goldGemSupply: Int = 5  // Gold gems available in supply (standard Splendor has 5)
   public var nobles: [Noble] = []
   public var currentPlayer: Int = 0
   public var currentTurn: Int = 0
   public var phase: GamePhase = .normalAction

   // Canonical moves are identical for every game — shared statically so Game
   // values stay cheap to copy (MCTS copies the game on every simulation).
   private static let _allMoves: [Move] = generateAllCanonicalMoves()

   // Memoized legal move mask for the current player
   private var _currentPlayerLegalMoveMask: [Bool] = []

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

      let gemsInSupply = playerCount == 2 ? 4 : 6
      self.supply = Array(repeating: gemsInSupply, count: GemType.allCases.count)

      // Initialize nobles: select playerCount + 1 nobles randomly
      var allNobles = Noble.allNobles()
      allNobles.shuffle(using: &rng)
      self.nobles = Array(allNobles.prefix(playerCount + 1))

      guard self.validate() else { return nil }

      // Initialize memoized legal move mask
      self._currentPlayerLegalMoveMask = Game._allMoves.map { move in
         self.isMoveLegal(move, forPlayer: self.currentPlayer)
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
   private static func generateAllCanonicalMoves () -> [Move] {
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

      // Discard gem moves: 6 possible (one for each gem type including gold)
      moves.append(contentsOf: [
         Move.discardGem(.red),
         Move.discardGem(.green),
         Move.discardGem(.blue),
         Move.discardGem(.white),
         Move.discardGem(.brown),
         Move.discardGoldGem
      ])

      precondition(moves.count == Game.CANONICAL_MOVE_COUNT, "Canonical move count must be \(Game.CANONICAL_MOVE_COUNT)")
      return moves
   }

   private func isMoveLegal (_ move: Move, forPlayer playerIndex: Int) -> Bool {
      guard playerIndex < players.count else { return false }
      let player = players[playerIndex]

      switch move {
      case .purchaseCard(let tier, let position):
         // Only legal during normal action phase
         guard phase == .normalAction else { return false }
         // Check if card exists at this position and the player can afford it
         guard tier < cardDecks.count && position < cardDecks[tier].count && position < 4 && position >= 0 else { return false }
         return player.canAfford(cost: cardDecks[tier][position].price)

      case .purchaseReservedCard(let position):
         // Only legal during normal action phase
         guard phase == .normalAction else { return false }
         // Check if player has a reserved card at this position
         guard position < player.reservedCards.count else { return false }
         return player.canAfford(cost: player.reservedCards[position].price)

      case .takeThreeGems(let gems):
         // Only legal during normal action phase
         guard phase == .normalAction else { return false }
         // Always legal - player can take 0 gems if all colors depleted (effectively a pass)
         return true

      case .takeTwoGems(let gem):
         // Only legal during normal action phase
         guard phase == .normalAction else { return false }
         // Check supply has at least 4 of this gem type
         guard supply[gem.rawValue] >= 4 else { return false }
         // No gem limit check - player can exceed 10 and will discard later
         return true

      case .reserveCard(let tier, let position):
         // Only legal during normal action phase
         guard phase == .normalAction else { return false }
         // Check if card exists at this position and player has fewer than the limit
         guard tier < cardDecks.count && position < cardDecks[tier].count else { return false }
         return player.reservedCards.count < Game.RESERVED_CARDS_PER_PLAYER_LIMIT

      case .discardGem(let gemType): // Only legal during discarding phase when over limit
         return phase == .discarding && player.gemCount > Game.GEMS_PER_PLAYER_LIMIT && player.gems[gemType.rawValue] > 0

      case .discardGoldGem: // Only legal during discarding phase when over limit
         return phase == .discarding && player.gemCount > Game.GEMS_PER_PLAYER_LIMIT && player.goldGems > 0
      }
   }


   // Game Protocol Functions

   public var canonicalMoveCount: Int {
      return Game.CANONICAL_MOVE_COUNT
   }

   public func move (atIndex index: Int) -> Move {
      precondition(index >= 0 && index < canonicalMoveCount, "Move index out of bounds")
      return Game._allMoves[index]
   }

   public func legalMoveMaskForCurrentPlayer () -> [Bool] {
      return _currentPlayerLegalMoveMask
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
      // Pay for the card using permanent gem discounts from owned cards, then
      // regular gems, then gold gems (wildcards) to cover any remaining shortfall.
      var goldUsed = 0
      for (gemIndex, price) in card.price.enumerated() {
         if price > 0 {
            let permanentGems = players[playerIndex].cardPower[gemIndex]
            let needed = max(0, price - permanentGems)
            let regularPaid = min(needed, players[playerIndex].gems[gemIndex])
            let goldNeeded = needed - regularPaid

            players[playerIndex].gems[gemIndex] -= regularPaid
            supply[gemIndex] += regularPaid
            goldUsed += goldNeeded
         }
      }
      // Deduct gold gems used as wildcards and return them to supply
      players[playerIndex].goldGems -= goldUsed
      goldGemSupply += goldUsed
   }

   private mutating func awardAvailableNobles (toPlayer playerIndex: Int) {
      // Check if player can afford any noble using only card-based purchasing power
      let cardPower = players[playerIndex].cardPower

      // Helper to check if a noble is affordable
      let isAffordable = { (noble: Noble) -> Bool in
         noble.price.enumerated().allSatisfy { (gemIndex, price) in
            cardPower[gemIndex] >= price
         }
      }

      // Filter nobles to find those the player can afford, transfer to the player
      let affordableNobles = nobles.filter(isAffordable)
      players[playerIndex].addNobles(affordableNobles)
      nobles = nobles.filter { !isAffordable($0) }
   }

   public mutating func applyMove (canonicalMoveIndex: Int) {

      // Debug: uncomment to trace move application
      // print("  Apply Move: player \(currentPlayer) move \(canonicalMoveIndex), phase \(phase)")

      // This function should never be called with an invalidate game state or invalid move,
      // we can check all these conditions with asserts that crash the program if they are violated

      assert(self.validate(), "Game state is invalid")

      precondition(canonicalMoveIndex < self.canonicalMoveCount, "Canonical move index is out of bounds")
      precondition(canonicalMoveIndex >= 0, "Canonical move index is negative")
      precondition(_currentPlayerLegalMoveMask[canonicalMoveIndex], "Move is not legal")

      let playerIndex = currentPlayer
      guard playerIndex < players.count else {
         preconditionFailure("Invalid player index")
      }

      let move = Game._allMoves[canonicalMoveIndex]

      switch move {
      case .purchaseCard(let tier, let position):
         let card = cardDecks[tier][position]

         // Pay for the card
         payForCard(card: card, playerIndex: playerIndex)

         // Remove card from deck and add to player
         cardDecks[tier].remove(at: position)
         players[playerIndex].addCard(card)

         // Check and award any available nobles
         awardAvailableNobles(toPlayer: playerIndex)

      case .purchaseReservedCard(let position):
         precondition(position < players[playerIndex].reservedCards.count, "Invalid reserved card position")
         let card = players[playerIndex].reservedCards[position]

         // Pay for the card
         payForCard(card: card, playerIndex: playerIndex)

         // Move card from reserved cards to owned cards
         players[playerIndex].addCard(card)
         players[playerIndex].reservedCards.remove(at: position)

         // Check and award any available nobles
         awardAvailableNobles(toPlayer: playerIndex)

      case .takeThreeGems(let gems):
         // Take gems from supply (only from colors that have supply available)
         for gem in gems {
            if supply[gem.rawValue] >= 1 {
               supply[gem.rawValue] -= 1
               players[playerIndex].gems[gem.rawValue] += 1
            }
         }

      case .takeTwoGems(let gem):
         precondition(supply[gem.rawValue] >= 4, "Insufficient gems in supply")

         // Take 2 gems from supply (no limit check - may exceed 10)
         supply[gem.rawValue] -= 2
         players[playerIndex].gems[gem.rawValue] += 2

      case .reserveCard(let tier, let position):
         precondition(players[playerIndex].reservedCards.count < Game.RESERVED_CARDS_PER_PLAYER_LIMIT, "Too many reserved cards")

         // Remove card from deck and add to reserved cards
         let card = cardDecks[tier].remove(at: position)
         players[playerIndex].reservedCards.append(card)

         // Give gold gem only if available in supply (may exceed 10 gems total)
         if goldGemSupply > 0 {
            goldGemSupply -= 1
            players[playerIndex].goldGems += 1
         }

      case .discardGem(let gemType):
         // Discard a gem and return it to supply
         precondition(players[playerIndex].gems[gemType.rawValue] > 0, "Player has no \(gemType) gems to discard")
         players[playerIndex].gems[gemType.rawValue] -= 1
         supply[gemType.rawValue] += 1

      case .discardGoldGem:
         // Discard a gold gem and return it to supply
         precondition(players[playerIndex].goldGems > 0, "Player has no gold gems to discard")
         players[playerIndex].goldGems -= 1
         goldGemSupply += 1
      }


      // Handle phase transitions based on gem count
      if phase == .normalAction {
         // After a normal action, check if player exceeds gem limit
         if players[playerIndex].gemCount > Game.GEMS_PER_PLAYER_LIMIT {
            // Player must discard - stay on same player, enter discarding phase
            phase = .discarding
         } else {
            // Player is within limit - advance to next player
            currentPlayer = (currentPlayer + 1) % players.count
            phase = .normalAction
            currentTurn += 1  // Increment turn only when advancing to next player
         }
      }
      else if phase == .discarding {
         // After discarding, check if player is now at or below limit
         if players[playerIndex].gemCount <= Game.GEMS_PER_PLAYER_LIMIT {
            // Player is done discarding - advance to next player, return to normal phase
            currentPlayer = (currentPlayer + 1) % players.count
            phase = .normalAction
            currentTurn += 1  // Increment turn only when advancing to next player
         }
         // Otherwise stay in discarding phase with same player
      }

      // Recompute legal move mask for the current player
      _currentPlayerLegalMoveMask = Game._allMoves.map { isMoveLegal($0, forPlayer: currentPlayer) }
   }

   /// Return a copy of this game with the hidden portion of each tier's deck
   /// reshuffled — a "determinization" for search under hidden information.
   ///
   /// Cards only ever leave `cardDecks[tier]` from the visible window (positions
   /// 0..<4, via `.remove(at:)` in `applyMove`, which shifts later cards up). So
   /// at any point in the game, `cardDecks[tier][visibleCount...]` already holds
   /// exactly the correct multiset of undrawn cards for that tier — it's just in
   /// the one true (but player-invisible) order fixed by the game's seed. Search
   /// must not be allowed to see or exploit that true order — a real opponent
   /// can't — so this replaces it with a freshly sampled random order, drawn from
   /// an RNG independent of the game's own seed. The visible cards (what's
   /// actually observed on the board right now) are left untouched.
   ///
   /// Only the future is being re-randomized here, not the present: this does not
   /// change anything a player could currently see or any move's legality.
   public func determinized (seed: UInt64) -> Game {
      var result = self
      var rng = SeededRandomNumberGenerator(seed: seed)

      for tier in 0..<result.cardDecks.count {
         let visibleCount = min(4, result.cardDecks[tier].count)
         guard visibleCount < result.cardDecks[tier].count else { continue } // nothing hidden left
         var hiddenTail = Array(result.cardDecks[tier][visibleCount...])
         hiddenTail.shuffle(using: &rng)
         result.cardDecks[tier].replaceSubrange(visibleCount..., with: hiddenTail)
      }

      return result
   }

   // Encode game state as a fixed-size array of Float16
   // Size: 204 (4 players × 51, incl. gem headroom) +
   //         5 (supply gem counts) +
   //         1 (gold supply gem count) +
   //        15 (10 take-three yields + 5 take-two yields) +
   //       130 (5 nobles × 26: 1 point + 5 price + 4 players × 5 per-color deficits) +
   //       144 (3 tiers × 4 cards × 12) +
   //         1 (turn) = 500
   public static let GAME_STATE_ENCODING_SIZE = 500

   public func encoding () -> [Float16] {
      var encoded: [Float16] = []

      // 4 players × 47 floats each (rotated so current player is always at index 0)
      let n = self.players.count
      for slot in 0..<4 {
         let i = (self.currentPlayer + slot) % n
         if slot < n {
            encoded.append(contentsOf: self.players[i].encoding())
         }
         else {
            // Zero-padding for missing players
            encoded.append(contentsOf: Array(repeating: Float16(0), count: PlayerState.ENCODED_SIZE))
         }
      }

      // 5 supply gem counts (one per GemType)
      for gemType in GemType.allCases {
         encoded.append(Float16(self.supply[gemType.rawValue]) / 6.0)
      }

      // 1 gold gem supply count
      encoded.append(Float16(self.goldGemSupply) / 5.0)

      // take-N gem yield flags: 15 values
      // 10 take-three combinations (yield 0-3, normalized /3) +
      //  5 take-two by color      (yield 0-2, normalized /2)
      // Each value is "how many gems this move would actually deliver given current supply."
      // Ordering matches generateAllCanonicalMoves (lex combinations for take-three,
      // GemType.allCases order for take-two), so flag i maps 1:1 to canonical move 15+i
      // for take-three, and 25+i for take-two.
      let gemTypes = Array(GemType.allCases)
      for i in 0..<gemTypes.count {
         for j in (i+1)..<gemTypes.count {
            for k in (j+1)..<gemTypes.count {
               let yield = (self.supply[gemTypes[i].rawValue] > 0 ? 1 : 0)
                         + (self.supply[gemTypes[j].rawValue] > 0 ? 1 : 0)
                         + (self.supply[gemTypes[k].rawValue] > 0 ? 1 : 0)
               encoded.append(Float16(yield) / 3.0)
            }
         }
      }
      for gemType in gemTypes {
         let yield = min(2, self.supply[gemType.rawValue])
         encoded.append(Float16(yield) / 2.0)
      }

      // noble encodings: 130 values
      // 5 available nobles × 26 floats each (1 point + 5 price + 4 players × 5 per-color deficits)
      let NOBLE_ENCODING_SIZE = 26
      for i in 0..<5 {
         if i < self.nobles.count {
            let noble = self.nobles[i]
            encoded.append(Float16(noble.points)/3.0)
            encoded.append(contentsOf: noble.price.map { Float16($0)/4.0 })

            // For each player, record per-color deficit toward acquiring this noble.
            // Deficit = max(0, noble price - player's card count of that color); gold gems
            // don't help here because nobles require cards, not gems.
            for playerSlot in 0..<4 {
               let playerIndex = (self.currentPlayer + playerSlot) % self.players.count
               if playerSlot < n {
                  let cardBasedPurchasePower = self.players[playerIndex].cardBasedPurchasePower()
                  for colorIndex in GemType.allCases.indices {
                     let deficit = max(0, noble.price[colorIndex] - cardBasedPurchasePower[colorIndex])
                     encoded.append(Float16(deficit) / 4.0)
                  }
               }
               else { // zero-padding for missing players' deficit vector
                  encoded.append(contentsOf: Array(repeating: Float16(0), count: GemType.allCases.count))
               }
            }
         } else {
            // Zero-padding for missing nobles
            encoded.append(contentsOf: Array(repeating: Float16(0), count: NOBLE_ENCODING_SIZE))
         }
      }

      // visible card encodings: 144 values
      // 3 tiers × 4 visible cards × 12 floats each (1 point value, 5 price values, 5 color one-hot, 1 affordability flag)
      for tier in 0..<3 {
         for position in 0..<4 {
            if tier < self.cardDecks.count && position < self.cardDecks[tier].count {
               let x = self.isMoveLegal(.purchaseCard(tier: tier, position: position), forPlayer: currentPlayer)
               encoded.append(contentsOf: self.cardDecks[tier][position].encoding(affordable: x))
            } else {
               // Zero-padding for missing cards
               encoded.append(contentsOf: Array(repeating: Float16(0), count: Card.ENCODED_SIZE))
            }
         }
      }

      // 1: current turn
      encoded.append(Float16(tanh(min(1.0, Float(self.currentTurn)/100.0))))

      precondition(encoded.count == Game.GAME_STATE_ENCODING_SIZE, "Encoded size mismatch: expected \(Game.GAME_STATE_ENCODING_SIZE), got \(encoded.count)")
      return encoded
   }

}

