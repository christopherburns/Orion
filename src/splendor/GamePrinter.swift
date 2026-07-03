
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
      let maxPriceLines = 4 // Standard Splendor cards have at most 4 colors of price

      // Top border
      lines.append("┌" + String(repeating: "─", count: cardWidth - 2) + "┐")

      // Points line (centered) — empty for 0-point cards but slot reserved for alignment
      if card.points > 0 {
         let pointsStr = "\(bold)\(card.points) pts\(reset)"
         let padding = cardWidth - 2 - visualWidth(pointsStr)
         let leftPad = padding / 2
         let rightPad = padding - leftPad
         lines.append("│" + String(repeating: " ", count: leftPad) + pointsStr + String(repeating: " ", count: rightPad) + "│")
      } else {
         lines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")
      }

      // Price section — one line per non-zero color, padded to fixed height
      var priceLines: [String] = []
      for (gemIndex, count) in card.price.enumerated() {
         if count > 0 {
            let gem = GemType(rawValue: gemIndex)!
            let colorCode = gemColor(gem)
            let squares = String(repeating: "■", count: count)
            let countStr = "\(count)"
            let squaresWidth = visualWidth("\(colorCode)\(squares)\(reset)")
            let countWidth = visualWidth(countStr)
            let padding = cardWidth - 4 - squaresWidth - countWidth
            let priceLine = " \(colorCode)\(squares)\(reset)" + String(repeating: " ", count: max(0, padding)) + countStr
            priceLines.append("│" + priceLine + " │")
         }
      }
      while priceLines.count < maxPriceLines {
         priceLines.append("│" + String(repeating: " ", count: cardWidth - 2) + "│")
      }
      lines.append(contentsOf: priceLines.prefix(maxPriceLines))

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

   /// Render a noble as a small box: 6 lines × 12 chars wide.
   /// Noble layout: ┌──┐ / points / 3 price lines (variable colors, max 3) / └──┘
   private static func formatNoble (_ noble: Noble) -> String {
      let nobleWidth = 12
      let maxPriceLines = 3 // Standard nobles have 2 or 3 colors of price
      var lines: [String] = []

      // Top border
      lines.append("┌" + String(repeating: "─", count: nobleWidth - 2) + "┐")

      // Points (centered)
      let pointsStr = "\(bold)\(noble.points) pts\(reset)"
      let pointsPad = nobleWidth - 2 - visualWidth(pointsStr)
      let pointsLeft = pointsPad / 2
      let pointsRight = pointsPad - pointsLeft
      lines.append("│" + String(repeating: " ", count: pointsLeft) + pointsStr + String(repeating: " ", count: pointsRight) + "│")

      // Price section (only non-zero colors, padded to maxPriceLines)
      var priceLines: [String] = []
      for (gemIndex, count) in noble.price.enumerated() {
         if count > 0 {
            let gem = GemType(rawValue: gemIndex)!
            let colorCode = gemColor(gem)
            let squares = String(repeating: "■", count: count)
            let squaresWidth = visualWidth("\(colorCode)\(squares)\(reset)")
            let countStr = "\(count)"
            let pad = nobleWidth - 4 - squaresWidth - visualWidth(countStr)
            priceLines.append("│ \(colorCode)\(squares)\(reset)" + String(repeating: " ", count: max(0, pad)) + "\(countStr) │")
         }
      }
      while priceLines.count < maxPriceLines {
         priceLines.append("│" + String(repeating: " ", count: nobleWidth - 2) + "│")
      }
      lines.append(contentsOf: priceLines.prefix(maxPriceLines))

      // Bottom border
      lines.append("└" + String(repeating: "─", count: nobleWidth - 2) + "┘")

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
         let count = game.supply[gem.rawValue]
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
         return "Purchase tier-\(tier + 1) \(cc)\(card.color.stringValue)\(reset) card [\(pts), costs \(costString(card.price))]"

      case .purchaseReservedCard(let position):
         let player = game.players[game.currentPlayer]
         guard position < player.reservedCards.count else {
            return "Purchase reserved card (slot \(position + 1))"
         }
         let card = player.reservedCards[position]
         let cc = cardColor(card.color)
         let pts = card.points > 0 ? "\(card.points)pt" : "0pt"
         return "Purchase reserved \(cc)\(card.color.stringValue)\(reset) card [\(pts), costs \(costString(card.price))]"

      case .takeThreeGems(let gems):
         let available = gems.filter { game.supply[$0.rawValue] > 0 }
         let gemDesc = gems.map { "\(gemColor($0))■\(reset) \($0.stringValue)" }.joined(separator: ", ")
         if available.count < gems.count {
            return "Take 3 gems: \(gemDesc) \(dim)(some depleted)\(reset)"
         }
         return "Take 3 gems: \(gemDesc)"

      case .takeTwoGems(let gem):
         let supply = game.supply[gem.rawValue]
         return "Take 2 \(gemColor(gem))■\(reset) \(gem.stringValue) gems (supply: \(supply))"

      case .reserveCard(let tier, let position):
         guard tier < game.cardDecks.count && position < game.cardDecks[tier].count else {
            return "Reserve tier-\(tier + 1) card (slot \(position + 1))"
         }
         let card = game.cardDecks[tier][position]
         let cc = cardColor(card.color)
         let pts = card.points > 0 ? "\(card.points)pt" : "0pt"
         return "Reserve tier-\(tier + 1) \(cc)\(card.color.stringValue)\(reset) card [\(pts), costs \(costString(card.price))] + \(brightYellow)★\(reset) gold"

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


   // ════════════════════════════════════════════════════════════════════════
   // Interactive 2-player layout
   //
   //   ┌─ Players ─────────────────┐  ┌─ Game Board ──────────────┐
   //   │ Opponent on top            │  │  tiers, supply, nobles    │
   //   │ ───────                    │  │                            │
   //   │ Current player on bottom   │  │                            │
   //   └────────────────────────────┘  └────────────────────────────┘
   //   ┌─ Moves (full width) ────────────────────────────────────────┐
   //   │ sorted/legal moves                                          │
   //   └─────────────────────────────────────────────────────────────┘
   //
   // Top region's height is governed by whichever side panel is taller.
   // Bottom region is full width (player width + gap + board width).
   // ════════════════════════════════════════════════════════════════════════

   private static let PLAYER_PANEL_WIDTH = 84  // fits 5 fused-card color columns (5×14 + 4×2 = 78) + 2-space indent + 4 border = 84
   private static let BOARD_PANEL_WIDTH  = 80
   private static let TOP_PANEL_GAP      = 2
   private static let MOVES_PANEL_WIDTH  = PLAYER_PANEL_WIDTH + TOP_PANEL_GAP + BOARD_PANEL_WIDTH

   /// Pad `s` (which may contain ANSI escape codes) on the right to a target visual width.
   private static func padRight (_ s: String, _ width: Int) -> String {
      let n = width - visualWidth(s)
      return n > 0 ? s + String(repeating: " ", count: n) : s
   }

   /// Wrap a list of content lines in a titled box of fixed outer width.
   /// Title sits in the top border like `┌─ Title ─────┐`.
   private static func boxed (title: String, content: [String], width: Int) -> [String] {
      var lines: [String] = []
      let titlePart = " \(bold)\(title)\(reset) "
      let titleVis = visualWidth(titlePart)
      let trailing = max(0, width - 4 - titleVis) // -4: "┌─" (2) + "─┐" (2)
      lines.append("┌─" + titlePart + String(repeating: "─", count: trailing) + "─┐")
      let innerWidth = width - 4 // -4: "│ " left, " │" right
      for c in content {
         lines.append("│ " + padRight(c, innerWidth) + " │")
      }
      lines.append("└" + String(repeating: "─", count: width - 2) + "┘")
      return lines
   }

   /// Place two panels (lists of lines) side-by-side with a small gap.
   /// Shorter panel is padded with blank lines at the bottom so both columns
   /// extend the same number of rows.
   private static func joinHorizontal (_ left: [String], _ right: [String], gap: Int = TOP_PANEL_GAP) -> [String] {
      let height = max(left.count, right.count)
      let leftWidth = left.map { visualWidth($0) }.max() ?? 0
      var out: [String] = []
      for i in 0..<height {
         let l = i < left.count  ? left[i]  : ""
         let r = i < right.count ? right[i] : ""
         out.append(padRight(l, leftWidth) + String(repeating: " ", count: gap) + r)
      }
      return out
   }

   // ── Panel builders (each returns the panel's lines including its border) ──

   /// Build the content lines for ONE player's section (no outer border).
   /// `isCurrentTurn` adds a small marker after the player number; positions are fixed
   /// (player 0 always on top, player 1 always on bottom) so the marker is the only
   /// turn-by-turn change in the player stack.
   private static func playerSection (player: PlayerState, playerIndex: Int, isCurrentTurn: Bool) -> [String] {
      var lines: [String] = []

      // Header
      let turnMarker = isCurrentTurn ? "  \(brightYellow)◀ to move\(reset)" : ""
      lines.append("\(bold)Player \(playerIndex)\(reset)\(turnMarker)   "
         + "Score: \(bold)\(player.score)\(reset)/15   "
         + "Cards: \(player.cards.count)   Gems: \(player.gemCount)/10")

      // Owned cards as per-color fused columns (the old style)
      if player.cards.isEmpty {
         lines.append("")
         lines.append("  \(dim)(no cards)\(reset)")
      } else {
         var byColor: [GemType: [Card]] = [:]
         for card in player.cards { byColor[card.color, default: []].append(card) }
         let colors = GemType.allCases.filter { byColor[$0] != nil }
         let colorGroups = colors.map { formatCardsFused(byColor[$0]!) }
         let maxHeight = colorGroups.map { $0.count }.max() ?? 0
         for i in 0..<maxHeight {
            var row = "  "
            for (idx, group) in colorGroups.enumerated() {
               row += i < group.count ? group[i] : String(repeating: " ", count: 14)
               if idx < colorGroups.count - 1 { row += "  " }
            }
            lines.append(row)
         }
      }

      // Gem inventory line (compact, only non-zero entries)
      let gemEntries = player.gems.enumerated().compactMap { (idx, n) -> String? in
         guard n > 0 else { return nil }
         let gem = GemType(rawValue: idx)!
         return "\(gemColor(gem))\(gem.stringValue):\(n)\(reset)"
      }
      var gemLine = "  Gems: " + (gemEntries.isEmpty ? "\(dim)(none)\(reset)" : gemEntries.joined(separator: "  "))
      if player.goldGems > 0 {
         gemLine += "   \(yellow)★:\(player.goldGems)\(reset)"
      }
      lines.append(gemLine)

      // Nobles claimed
      if !player.nobles.isEmpty {
         lines.append("  Nobles: \(player.nobles.count)  (+\(player.nobles.reduce(0) { $0 + $1.points }) pts)")
      }

      // Reserved cards (full card visuals)
      if !player.reservedCards.isEmpty {
         lines.append("  \(bold)Reserved:\(reset)")
         let cardLines = player.reservedCards.map { formatCard($0) }
         let cardArrays = cardLines.map { $0.split(separator: "\n").map(String.init) }
         let maxLines = cardArrays.map { $0.count }.max() ?? 0
         for i in 0..<maxLines {
            var row = "  "
            for (idx, arr) in cardArrays.enumerated() {
               row += i < arr.count ? arr[i] : String(repeating: " ", count: 14)
               if idx < cardArrays.count - 1 { row += "  " }
            }
            lines.append(row)
         }
      }

      return lines
   }

   /// Combined player-stack panel: Player 0 always on top, Player 1 always on
   /// bottom. The `◀ to move` marker on the header line is the only visual change
   /// from turn to turn — sections themselves don't swap positions.
   private static func panelPlayersStack (game: Game) -> [String] {
      let innerWidth = PLAYER_PANEL_WIDTH - 4
      let divider = String(repeating: "─", count: innerWidth)
      var content: [String] = []
      content.append(contentsOf: playerSection(player: game.players[0], playerIndex: 0,
                                               isCurrentTurn: game.currentPlayer == 0))
      content.append("")
      content.append(divider)
      content.append("")
      content.append(contentsOf: playerSection(player: game.players[1], playerIndex: 1,
                                               isCurrentTurn: game.currentPlayer == 1))
      return boxed(title: "Players", content: content, width: PLAYER_PANEL_WIDTH)
   }

   private static func panelBoard (_ game: Game) -> [String] {
      var content: [String] = []

      // Three tiers, each a row of 4 cards
      for tier in (0..<3).reversed() {
         content.append("\(bold)Tier \(tier + 1)\(reset)")
         let visible = Array(game.cardDecks[tier].prefix(4))
         if visible.isEmpty {
            content.append("  (deck empty)")
            content.append("")
            continue
         }
         let cardLines = visible.map { formatCard($0) }
         let cardLineArrays = cardLines.map { $0.split(separator: "\n").map(String.init) }
         let maxLines = cardLineArrays.map { $0.count }.max() ?? 0
         for i in 0..<maxLines {
            var row = ""
            for (idx, arr) in cardLineArrays.enumerated() {
               row += i < arr.count ? arr[i] : String(repeating: " ", count: 14)
               if idx < cardLineArrays.count - 1 { row += "  " }
            }
            content.append(row)
         }
         content.append("")
      }

      // Supply (left) and Nobles (right), side by side within the panel
      var supplyLines: [String] = ["\(bold)Supply\(reset)"]
      for gem in GemType.allCases {
         let n = game.supply[gem.rawValue]
         supplyLines.append("  \(gemColor(gem))\(gem.stringValue): \(n)\(reset)")
      }
      supplyLines.append("  \(yellow)gold: \(game.goldGemSupply)\(reset)")

      var nobleLines: [String] = ["\(bold)Nobles\(reset)"]
      if game.nobles.isEmpty {
         nobleLines.append("  (none)")
      } else {
         let nobleCardLines = game.nobles.map { formatNoble($0) }
         let arrays = nobleCardLines.map { $0.split(separator: "\n").map(String.init) }
         let maxLines = arrays.map { $0.count }.max() ?? 0
         for i in 0..<maxLines {
            var row = ""
            for (idx, arr) in arrays.enumerated() {
               row += i < arr.count ? arr[i] : String(repeating: " ", count: 12)
               if idx < arrays.count - 1 { row += " " }
            }
            nobleLines.append(row)
         }
      }

      // Side-by-side: supply gets ~18 chars, nobles get the rest
      let supplyColWidth = 18
      let height = max(supplyLines.count, nobleLines.count)
      for i in 0..<height {
         let l = i < supplyLines.count ? supplyLines[i] : ""
         let r = i < nobleLines.count  ? nobleLines[i]  : ""
         content.append(padRight(l, supplyColWidth) + r)
      }

      return boxed(title: "Game Board", content: content, width: BOARD_PANEL_WIDTH)
   }

   /// Moves panel for the CPU's turn: bars + percentages + descriptions, sorted.
   private static func panelMovesCPU (game: Game, legalMoveIndices: [Int], probabilities: [Float], chosenIndex: Int) -> [String] {
      let BAR_WIDTH = 20
      let sorted = legalMoveIndices.sorted { probabilities[$0] > probabilities[$1] }
      var content: [String] = []
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
         let marker = isChosen ? "  \(bold)◀ CHOSEN\(reset)" : ""
         let numStr = String(format: "%3d.", menuNum + 1)
         let probStr = String(format: "%5.1f%%", prob * 100.0)
         let desc = describeMoveVerbose(game.move(atIndex: moveIdx), game: game)
         var line = "\(numStr) \(barColor)\(bar)\(reset) \(probStr)  \(desc)\(marker)"
         if isChosen { line = "\(bold)\(line)\(reset)" }
         content.append(line)
      }
      return boxed(title: "CPU move probabilities", content: content, width: MOVES_PANEL_WIDTH)
   }

   /// Moves panel for the human's turn: numbered list of legal moves (canonical order).
   private static func panelMovesHuman (game: Game, legalMoveIndices: [Int]) -> [String] {
      var content: [String] = []
      for (menuNum, moveIdx) in legalMoveIndices.enumerated() {
         let numStr = String(format: "%3d.", menuNum + 1)
         let desc = describeMoveVerbose(game.move(atIndex: moveIdx), game: game)
         content.append("\(numStr)  \(desc)")
      }
      return boxed(title: "Choose a move", content: content, width: MOVES_PANEL_WIDTH)
   }

   /// Top-level interactive view for a 2-player game.
   /// Top region: combined player-stack (opponent on top, you on bottom) + game-board.
   /// Bottom region: full-width moves panel (CPU sorted+probs, or human numbered).
   public static func presentInteractive2P (
      game: Game,
      legalMoveIndices: [Int],
      probabilities: [Float]? = nil,
      chosenIndex: Int? = nil) {

      precondition(game.players.count == 2, "presentInteractive2P only supports 2-player games")

      let playersBox = panelPlayersStack(game: game)
      let boardBox   = panelBoard(game)

      let movesBox: [String]
      if let probs = probabilities, let chosen = chosenIndex {
         movesBox = panelMovesCPU(game: game, legalMoveIndices: legalMoveIndices,
                                  probabilities: probs, chosenIndex: chosen)
      } else {
         movesBox = panelMovesHuman(game: game, legalMoveIndices: legalMoveIndices)
      }

      let topRow = joinHorizontal(playersBox, boardBox)

      print("")
      print("\(bold)═══ Turn \(game.currentTurn)  ·  Player \(game.currentPlayer)'s turn  ═══\(reset)")
      for line in topRow  { print(line) }
      for line in movesBox { print(line) }
   }
}

