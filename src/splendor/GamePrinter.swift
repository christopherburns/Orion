
public class GamePrinter {

   // ANSI color codes
   private static let red = "\u{001B}[31m"
   private static let green = "\u{001B}[32m"
   private static let blue = "\u{001B}[34m"
   private static let white = "\u{001B}[37m"
   private static let yellow = "\u{001B}[33m" // Using yellow for brown
   private static let reset = "\u{001B}[0m"
   private static let bold = "\u{001B}[1m"

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
      print("\n\(bold)Player \(playerIndex + 1)\(reset)")
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

      case .pass:
         print("\(turnPrefix)\(bold)Player \(playerIndex + 1)\(reset) passes")
      }
   }

}

