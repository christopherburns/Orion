
public class GamePrinter {

   // ANSI color codes
   private static let red         = "\u{001B}[31m"
   private static let green       = "\u{001B}[32m"
   private static let blue        = "\u{001B}[34m"
   private static let white       = "\u{001B}[37m"
   private static let yellow      = "\u{001B}[33m" // Using yellow for brown
   private static let reset       = "\u{001B}[0m"
   private static let bold        = "\u{001B}[1m"
   private static let brightRed   = "\u{001B}[91m"
   private static let brightYellow = "\u{001B}[93m"
   private static let grey        = "\u{001B}[90m"
   private static let dim         = "\u{001B}[2m"

   private static func gemColor (_ gem: GemType) -> String {
      switch gem {
      case .red: return red
      case .green: return green
      case .blue: return blue
      case .white: return white
      case .brown: return yellow
      }
   }

   private static func cardColor (_ gem: GemType) -> String {
      switch gem {
      case .red: return red
      case .green: return green
      case .blue: return blue
      case .white: return white
      case .brown: return yellow
      }
   }

   private static func visualWidth (_ str: String) -> Int {
      // Count visual width, ignoring ANSI escape codes
      var width = 0
      var inEscape = false
      for char in str {
         if char == "\u{001B}" {
            inEscape = true
         } else if inEscape {
            if char == "m" {
               inEscape = false
            }
         } else {
            width += 1
         }
      }
      return width
   }

   private static func formatCard (_ card: Card) -> String {
      var lines: [String] = []
      let cardColorCode = cardColor(card.color)
      let cardWidth = 14
      let maxPriceLines = 5 // Fixed number of price lines for alignment

      // Top border
      lines.append("┌" + String(repeating: "─", count: cardWidth - 2) + "┐")

      // Points line (centered)
      if card.points > 0 {
         let pointsStr = "\(bold)\(card.points) pts\(reset)"
         let padding = cardWidth - 2 - visualWidth(pointsStr)
         let leftPad = padding / 2
         let rightPad = padding - leftPad
         lines.append("│" + String(repeating: " ", count: leftPad) + pointsStr + String(repeating: " ", count: rightPad) + "│")
      } else {
         lines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")
      }

      // Empty line
      lines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")

      // Price section - show each gem type with colored squares (fixed height)
      var priceLines: [String] = []
      for (gemIndex, count) in card.price.enumerated() {
         if count > 0 {
            let gem = GemType(rawValue: gemIndex)!
            let colorCode = gemColor(gem)
            let squares = String(repeating: "■", count: count)
            let countStr = "\(count)"
            // Left side: colored squares, right side: count number
            let squaresWidth = visualWidth("\(colorCode)\(squares)\(reset)")
            let countWidth = visualWidth(countStr)
            let padding = cardWidth - 4 - squaresWidth - countWidth // -4 for: │, leading space, trailing space, │
            let priceLine = " \(colorCode)\(squares)\(reset)" + String(repeating: " ", count: max(0, padding)) + countStr
            priceLines.append("│" + priceLine + " │")
         }
      }

      // Pad price section to fixed height
      while priceLines.count < maxPriceLines {
         priceLines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")
      }

      // Take only maxPriceLines to ensure fixed height
      lines.append(contentsOf: priceLines.prefix(maxPriceLines))

      // Empty line
      lines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")

      // Color line (centered)
      let colorName = card.color.stringValue
      let colorDisplay = "\(cardColorCode)\(colorName)\(reset)"
      let padding = cardWidth - 2 - visualWidth(colorDisplay)
      let leftPad = padding / 2
      let rightPad = padding - leftPad
      lines.append("│" + String(repeating: " ", count: leftPad) + colorDisplay + String(repeating: " ", count: rightPad) + "│")

      // Bottom border
      lines.append("└" + String(repeating: "─", count: cardWidth - 2) + "┘")

      return lines.joined(separator: "\n")
   }

   public static func present (_ game: Game) {
      // Print each tier
      for tier in (0..<3).reversed() {
         print("\n\(bold)Tier \(tier + 1)\(reset)")
         print(String(repeating: "─", count: 80))

         let deck = game.cardDecks[tier]
         let visibleCards = Array(deck.prefix(4))

         if visibleCards.isEmpty {
            print("(No cards available)")
            continue
         }

         // Print cards side by side
         let cardLines = visibleCards.map { formatCard($0) }
         let maxLines = cardLines.map { $0.split(separator: "\n").count }.max() ?? 0

         for lineIndex in 0..<maxLines {
            var line = ""
            for cardStr in cardLines {
               let cardLinesArray = cardStr.split(separator: "\n")
               if lineIndex < cardLinesArray.count {
                  line += String(cardLinesArray[lineIndex])
               } else {
                  line += "       " // Empty space for shorter cards
               }
               line += "  " // Spacing between cards
            }
            print(line)
         }
      }

      // Print supply
      print("\n\(bold)Supply:\(reset)")
      for gem in GemType.allCases {
         let count = game.supply[gem, default: 0]
         let colorCode = gemColor(gem)
         print("  \(colorCode)\(gem.stringValue): \(count)\(reset)")
      }

      print()
   }

   private static func formatCardCondensed (_ card: Card) -> String {
      let cardColorCode = cardColor(card.color)
      let cardWidth = 14

      // Top border
      let topBorder = "┌" + String(repeating: "─", count: cardWidth - 2) + "┐"

      // Line 1: Points (centered)
      var line1: String
      if card.points > 0 {
         let pointsStr = "\(bold)\(card.points) pts\(reset)"
         let padding = cardWidth - 2 - visualWidth(pointsStr)
         let leftPad = padding / 2
         let rightPad = padding - leftPad
         line1 = "│" + String(repeating: " ", count: leftPad) + pointsStr + String(repeating: " ", count: rightPad) + "│"
      } else {
         line1 = "│" + String(repeating: " ", count: cardWidth - 2) + "│"
      }

      // Line 2: Color (centered)
      let colorName = card.color.stringValue
      let colorDisplay = "\(cardColorCode)\(colorName)\(reset)"
      let padding = cardWidth - 2 - visualWidth(colorDisplay)
      let leftPad = padding / 2
      let rightPad = padding - leftPad
      let line2 = "│" + String(repeating: " ", count: leftPad) + colorDisplay + String(repeating: " ", count: rightPad) + "│"

      // Bottom border
      let bottomBorder = "└" + String(repeating: "─", count: cardWidth - 2) + "┘"

      return topBorder + "\n" + line1 + "\n" + line2 + "\n" + bottomBorder
   }

   private static func formatCardsFused (_ cards: [Card]) -> [String] {
      guard !cards.isEmpty else { return [] }

      let cardWidth = 14
      var lines: [String] = []

      for (index, card) in cards.enumerated() {
         let cardColorCode = cardColor(card.color)
         let isFirst = index == 0
         let isLast = index == cards.count - 1

         // Top border (or shared border)
         if isFirst {
            lines.append("┌" + String(repeating: "─", count: cardWidth - 2) + "┐")
         } else {
            lines.append("├" + String(repeating: "─", count: cardWidth - 2) + "┤")
         }

         // Line 1: Points (centered)
         if card.points > 0 {
            let pointsStr = "\(bold)\(card.points) pts\(reset)"
            let padding = cardWidth - 2 - visualWidth(pointsStr)
            let leftPad = padding / 2
            let rightPad = padding - leftPad
            lines.append("│" + String(repeating: " ", count: leftPad) + pointsStr + String(repeating: " ", count: rightPad) + "│")
         } else {
            lines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")
         }

         // Line 2: Color (centered)
         let colorName = card.color.stringValue
         let colorDisplay = "\(cardColorCode)\(colorName)\(reset)"
         let padding = cardWidth - 2 - visualWidth(colorDisplay)
         let leftPad = padding / 2
         let rightPad = padding - leftPad
         lines.append("│" + String(repeating: " ", count: leftPad) + colorDisplay + String(repeating: " ", count: rightPad) + "│")

         // Bottom border (only for last card)
         if isLast {
            lines.append("└" + String(repeating: "─", count: cardWidth - 2) + "┘")
         }
      }

      return lines
   }

   public static func presentPlayer (_ player: PlayerState, playerIndex: Int) {
      print("\n\(bold)Player \(playerIndex)\(reset)")
      print(String(repeating: "─", count: 80))

      if player.cards.isEmpty {
         print("  No cards")
      } else {
         // Group cards by color
         var cardsByColor: [GemType: [Card]] = [:]
         for card in player.cards {
            cardsByColor[card.color, default: []].append(card)
         }

         // Get all colors in a consistent order
         let colors = GemType.allCases.filter { cardsByColor[$0] != nil }

         // Format cards for each color group as fused columns
         var colorGroups: [[String]] = []
         var maxHeight = 0

         for color in colors {
            let cards = cardsByColor[color]!
            let fusedLines = formatCardsFused(cards)
            colorGroups.append(fusedLines)
            maxHeight = max(maxHeight, fusedLines.count)
         }

         // Print cards column by column
         for lineIndex in 0..<maxHeight {
            var line = "  "
            for (groupIndex, groupLines) in colorGroups.enumerated() {
               if lineIndex < groupLines.count {
                  line += groupLines[lineIndex]
               } else {
                  line += String(repeating: " ", count: 14)
               }

               // Spacing between color groups
               if groupIndex < colorGroups.count - 1 {
                  line += "  "
               }
            }
            print(line)
         }
      }

      // Print gems
      print("\n  Gems:")
      for (gemIndex, count) in player.gems.enumerated() {
         if count > 0 {
            let gem = GemType(rawValue: gemIndex)!
            let colorCode = gemColor(gem)
            print("    \(colorCode)\(gem.stringValue): \(count)\(reset)")
         }
      }

      if player.goldGems > 0 {
         print("    \(yellow)gold: \(player.goldGems)\(reset)")
      }

      // Print reserved cards
      if !player.reservedCards.isEmpty {
         print("\n  Reserved cards:")
         // Print cards side by side using full format
         let cardLines = player.reservedCards.map { formatCard($0) }
         let maxLines = cardLines.map { $0.split(separator: "\n").count }.max() ?? 0

         for lineIndex in 0..<maxLines {
            var line = "    "
            for cardStr in cardLines {
               let cardLinesArray = cardStr.split(separator: "\n")
               if lineIndex < cardLinesArray.count {
                  line += String(cardLinesArray[lineIndex])
               } else {
                  line += String(repeating: " ", count: 14) // Empty space for shorter cards
               }
               line += "  " // Spacing between cards
            }
            print(line)
         }
      }

      print()
   }

   public static func presentMove (moveIndex: Int, game: Game) {
      let move = game.move(atIndex: moveIndex)
      let playerIndex = game.currentPlayer

      let turnPrefix = "Turn \(game.currentTurn): "

      switch move {
      case .purchaseCard(let tier, let position):
         if tier < game.cardDecks.count && position < game.cardDecks[tier].count {
            let card = game.cardDecks[tier][position]
            let colorCode = cardColor(card.color)
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) purchases \(colorCode)\(card.color.stringValue)\(reset) card (Tier \(tier + 1), Position \(position + 1)): \(card.points > 0 ? "\(card.points) pts" : "no points")")
         } else {
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) attempts to purchase card (Tier \(tier + 1), Position \(position + 1))")
         }

      case .purchaseReservedCard(let position):
         let player = game.players[playerIndex]
         if position < player.reservedCards.count {
            let card = player.reservedCards[position]
            let colorCode = cardColor(card.color)
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) purchases reserved \(colorCode)\(card.color.stringValue)\(reset) card (Position \(position + 1)): \(card.points > 0 ? "\(card.points) pts" : "no points")")
         } else {
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) attempts to purchase reserved card (Position \(position + 1))")
         }

      case .takeThreeGems(let gems):
         let gemStrings = gems.map { gem in
            let colorCode = gemColor(gem)
            return "\(colorCode)\(gem.stringValue)\(reset)"
         }
         print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) takes three gems: \(gemStrings.joined(separator: ", "))")

      case .takeTwoGems(let gem):
         let colorCode = gemColor(gem)
         print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) takes two \(colorCode)\(gem.stringValue)\(reset) gems")

      case .reserveCard(let tier, let position):
         if tier < game.cardDecks.count && position < game.cardDecks[tier].count {
            let card = game.cardDecks[tier][position]
            let colorCode = cardColor(card.color)
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) reserves \(colorCode)\(card.color.stringValue)\(reset) card (Tier \(tier + 1), Position \(position + 1)): \(card.points > 0 ? "\(card.points) pts" : "no points")")
         } else {
            print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) attempts to reserve card (Tier \(tier + 1), Position \(position + 1))")
         }

      case .discardGem(let gemType):
         let colorCode = gemColor(gemType)
         print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) discards \(colorCode)\(gemType.stringValue)\(reset) gem")

      case .discardGoldGem:
         print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) discards gold gem")

      }
   }

   /// Print a bar graph showing move probabilities
   /// - Parameters:
   ///   - probabilities: Probability distribution over all moves (must sum to ~1.0)
   ///   - game: Current game state (used to describe moves)
   ///   - topN: Show only the top N moves (default: 10)
   public static func presentMoveProbabilities (_ probabilities: [Float], game: Splendor.Game, topN: Int = 10) {
      precondition(probabilities.count == game.canonicalMoveCount, "Probabilities must match canonical move count")

      print("\n\(bold)Move Probabilities:\(reset)")

      // Create array of (index, probability, move) tuples for non-zero probabilities
      var movesWithProbs: [(index: Int, prob: Float, move: Splendor.Game.Move)] = []
      for (index, prob) in probabilities.enumerated() {
         if prob > 0.001 { // Only show moves with >0.1% probability
            movesWithProbs.append((index, prob, game.move(atIndex: index)))
         }
      }

      // Sort by probability descending
      movesWithProbs.sort { $0.prob > $1.prob }

      // Show top N moves
      let maxWidth = 50 // Maximum bar width in characters
      for (rank, item) in movesWithProbs.prefix(topN).enumerated() {
         let percentage = item.prob * 100.0
         let barWidth = Int(item.prob * Float(maxWidth))
         let bar = String(repeating: "█", count: barWidth)
         let paddedBar = bar.padding(toLength: maxWidth, withPad: " ", startingAt: 0)

         // Get move description
         let moveDesc = describeMoveShort(item.move, game: game)

         // Print: rank. [bar] XX.X% - move description
         let rankStr = String(format: "%2d", rank + 1)
         let percentStr = String(format: "%5.1f", percentage)
         print("\(rankStr). \(green)\(paddedBar)\(reset) \(percentStr)% - \(moveDesc)")
      }

      // Show total probability covered
      let totalShown = movesWithProbs.prefix(topN).reduce(0.0) { $0 + $1.prob }
      if movesWithProbs.count > topN {
         print("    ... (\(movesWithProbs.count - topN) more moves, \(String(format: "%.1f", (1.0 - totalShown) * 100.0))% probability)")
      }
   }

   // ── Interactive mode ──────────────────────────────────────────────────────────

   /// Format gem costs as colored squares, e.g. "2■(red) 1■(blue)"
   private static func costString (_ price: [Int]) -> String {
      let parts = price.enumerated().compactMap { (index, count) -> String? in
         guard count > 0 else { return nil }
         let gem = GemType(rawValue: index)!
         return "\(gemColor(gem))\(count)■\(reset)"
      }
      return parts.isEmpty ? "free" : parts.joined(separator: " ")
   }

   /// Create a verbose, human-readable description of a move including card stats and gem costs
   public static func describeMoveVerbose (_ move: Splendor.Game.Move, game: Splendor.Game) -> String {
      switch move {

      case .purchaseCard(let tier, let position):
         guard tier < game.cardDecks.count && position < game.cardDecks[tier].count else {
            return "Purchase tier-\(tier + 1) card (slot \(position + 1))"
         }
         let card = game.cardDecks[tier][position]
         let cc = cardColor(card.color)
         let pts = card.points > 0 ? "\(card.points)pt" : "0pt"
         return "Purchase tier-\(tier + 1) \(cc)\(card.color.stringValue)\(reset) card [\(pts), produces \(cc)■\(reset), costs \(costString(card.price))]"

      case .purchaseReservedCard(let position):
         let player = game.players[game.currentPlayer]
         guard position < player.reservedCards.count else {
            return "Purchase reserved card (slot \(position + 1))"
         }
         let card = player.reservedCards[position]
         let cc = cardColor(card.color)
         let pts = card.points > 0 ? "\(card.points)pt" : "0pt"
         return "Purchase reserved \(cc)\(card.color.stringValue)\(reset) card [\(pts), produces \(cc)■\(reset), costs \(costString(card.price))]"

      case .takeThreeGems(let gems):
         let available = gems.filter { game.supply[$0, default: 0] > 0 }
         let gemDesc = gems.map { "\(gemColor($0))■\(reset) \($0.stringValue)" }.joined(separator: ", ")
         if available.count < gems.count {
            return "Take 3 gems: \(gemDesc) \(dim)(some depleted)\(reset)"
         }
         return "Take 3 gems: \(gemDesc)"

      case .takeTwoGems(let gem):
         let supply = game.supply[gem, default: 0]
         return "Take 2 \(gemColor(gem))■\(reset) \(gem.stringValue) gems (supply: \(supply))"

      case .reserveCard(let tier, let position):
         guard tier < game.cardDecks.count && position < game.cardDecks[tier].count else {
            return "Reserve tier-\(tier + 1) card (slot \(position + 1))"
         }
         let card = game.cardDecks[tier][position]
         let cc = cardColor(card.color)
         let pts = card.points > 0 ? "\(card.points)pt" : "0pt"
         return "Reserve tier-\(tier + 1) \(cc)\(card.color.stringValue)\(reset) card [\(pts), produces \(cc)■\(reset), costs \(costString(card.price))] + \(brightYellow)★\(reset) gold"

      case .discardGem(let gemType):
         let player = game.players[game.currentPlayer]
         let have = player.gems[gemType.rawValue]
         return "Discard \(gemColor(gemType))■\(reset) \(gemType.stringValue) gem (you have \(have))"

      case .discardGoldGem:
         let player = game.players[game.currentPlayer]
         return "Discard \(brightYellow)★\(reset) gold gem (you have \(player.goldGems))"
      }
   }

   /// Show the chosen card in full when the CPU selects a card move
   private static func printChosenCard (_ move: Splendor.Game.Move, game: Splendor.Game) {
      let card: Card?
      let label: String
      switch move {
      case .purchaseCard(let tier, let position):
         card = (tier < game.cardDecks.count && position < game.cardDecks[tier].count) ? game.cardDecks[tier][position] : nil
         label = "Card being purchased:"
      case .purchaseReservedCard(let position):
         let player = game.players[game.currentPlayer]
         card = position < player.reservedCards.count ? player.reservedCards[position] : nil
         label = "Reserved card being purchased:"
      case .reserveCard(let tier, let position):
         card = (tier < game.cardDecks.count && position < game.cardDecks[tier].count) ? game.cardDecks[tier][position] : nil
         label = "Card being reserved:"
      default:
         return
      }
      guard let c = card else { return }
      print("\n  \(bold)\(label)\(reset)")
      let rendered = formatCard(c).split(separator: "\n", omittingEmptySubsequences: false)
      for line in rendered {
         print("  " + line)
      }
   }

   /// Display a numbered move menu with heat-colored probability bars for a CPU turn.
   /// Moves are sorted by probability (highest first). The chosen move is marked.
   public static func presentCPUMoveMenu (
      playerIndex: Int,
      legalMoveIndices: [Int],
      probabilities: [Float],
      chosenIndex: Int,
      game: Splendor.Game) {

      let BAR_WIDTH = 20
      let sorted = legalMoveIndices.sorted { probabilities[$0] > probabilities[$1] }

      print("\n\(bold)Player \(playerIndex + 1) (CPU) legal moves:\(reset)")

      for (menuNum, moveIdx) in sorted.enumerated() {
         let prob = probabilities[moveIdx]
         let barFilled = Int((prob * Float(BAR_WIDTH)).rounded())
         let bar = String(repeating: "█", count: barFilled)
            + String(repeating: "░", count: BAR_WIDTH - barFilled)

         let barColor: String
         if prob > 0.50      { barColor = brightRed    }
         else if prob > 0.20 { barColor = brightYellow }
         else if prob > 0.05 { barColor = green        }
         else                { barColor = grey         }

         let isChosen = moveIdx == chosenIndex
         let marker   = isChosen ? "  \(bold)◀ CHOSEN\(reset)" : ""
         let numStr   = String(format: "%3d.", menuNum + 1)
         let probStr  = String(format: "%5.1f%%", prob * 100.0)
         let desc     = describeMoveVerbose(game.move(atIndex: moveIdx), game: game)

         let line = "\(numStr)  \(barColor)\(bar)\(reset)  \(probStr)  \(desc)\(marker)"
         if isChosen {
            print("\(bold)\(line)\(reset)")
         } else {
            print(line)
         }
      }

      // Show the chosen card in full
      printChosenCard(game.move(atIndex: chosenIndex), game: game)
   }

   /// Display a numbered move menu for a human player to choose from.
   /// Moves are shown in canonical order (matches the board layout).
   public static func presentHumanMoveMenu (
      playerIndex: Int,
      legalMoveIndices: [Int],
      game: Splendor.Game) {

      print("\n\(bold)Player \(playerIndex + 1) (You) — choose a move:\(reset)")

      for (menuNum, moveIdx) in legalMoveIndices.enumerated() {
         let numStr = String(format: "%3d.", menuNum + 1)
         let desc   = describeMoveVerbose(game.move(atIndex: moveIdx), game: game)
         print("\(numStr)  \(desc)")
      }
   }

   /// Create a short description of a move
   private static func describeMoveShort (_ move: Splendor.Game.Move, game: Splendor.Game) -> String {
      switch move {
      case .purchaseCard(let tier, let position):
         if tier < game.cardDecks.count && position < game.cardDecks[tier].count {
            let card = game.cardDecks[tier][position]
            return "Buy \(card.color.stringValue) T\(tier+1) (\(card.points)pts)"
         }
         return "Buy T\(tier+1) P\(position+1)"

      case .purchaseReservedCard(let position):
         return "Buy reserved #\(position+1)"

      case .takeThreeGems(let gems):
         let gemNames = gems.map { $0.stringValue }.joined(separator: ", ")
         return "Take 3: \(gemNames)"

      case .takeTwoGems(let gem):
         return "Take 2: \(gem.stringValue)"

      case .reserveCard(let tier, let position):
         if tier < game.cardDecks.count && position < game.cardDecks[tier].count {
            let card = game.cardDecks[tier][position]
            return "Reserve \(card.color.stringValue) T\(tier+1) (\(card.points)pts)"
         }
         return "Reserve T\(tier+1) P\(position+1)"

      case .discardGem(let gemType):
         return "Discard \(gemType.stringValue)"

      case .discardGoldGem:
         return "Discard gold"
      }
   }
}

