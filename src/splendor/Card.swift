
public enum GemType: Int, CaseIterable {
   case red, green, blue, white, brown

   public var stringValue: String {
      return ["red", "green", "blue", "white", "brown"][rawValue]
   }
}

public struct Card {
   public var points: Int
   public var price: [Int] = [0, 0, 0, 0, 0] // Indexed by GemType.rawValue
   public var color: GemType

   // Helper initializer: price array is [red, green, blue, white, brown]
   public init (points: Int, price: [Int], color: GemType) {
      self.points = points
      self.price = price
      self.color = color
      precondition(price.count == GemType.allCases.count, "Price array must match GemType count")
   }

   // Returns all cards organized by tier: [tier1, tier2, tier3]
   public static func allCards () -> [[Card]] {
      return [tier1Cards(), tier2Cards(), tier3Cards()]
   }

   public static func tier1Cards () -> [Card] {
      // Tier 1: 40 cards, 0-5 points, cheaper costs
      return [
         // Red cards
         Card(points: 0, price: [0, 3, 0, 0, 0], color: .red),
         Card(points: 0, price: [0, 0, 2, 1, 0], color: .red),
         Card(points: 0, price: [0, 1, 1, 1, 1], color: .red),
         Card(points: 0, price: [0, 1, 2, 1, 1], color: .red),
         Card(points: 0, price: [0, 0, 0, 0, 2], color: .red),
         Card(points: 0, price: [0, 2, 2, 0, 0], color: .red),
         Card(points: 0, price: [0, 3, 1, 0, 1], color: .red),
         Card(points: 1, price: [0, 4, 0, 0, 0], color: .red),

         // Green cards
         Card(points: 0, price: [0, 0, 0, 3, 0], color: .green),
         Card(points: 0, price: [1, 0, 0, 0, 2], color: .green),
         Card(points: 0, price: [1, 0, 1, 1, 1], color: .green),
         Card(points: 0, price: [1, 0, 1, 2, 1], color: .green),
         Card(points: 0, price: [0, 0, 2, 0, 0], color: .green),
         Card(points: 0, price: [2, 0, 0, 2, 0], color: .green),
         Card(points: 0, price: [1, 0, 0, 1, 3], color: .green),
         Card(points: 1, price: [0, 0, 0, 4, 0], color: .green),

         // Blue cards
         Card(points: 0, price: [0, 0, 3, 0, 0], color: .blue),
         Card(points: 0, price: [2, 1, 0, 0, 0], color: .blue),
         Card(points: 0, price: [1, 1, 0, 1, 1], color: .blue),
         Card(points: 0, price: [1, 1, 0, 1, 2], color: .blue),
         Card(points: 0, price: [0, 2, 0, 0, 0], color: .blue),
         Card(points: 0, price: [0, 2, 0, 2, 0], color: .blue),
         Card(points: 0, price: [0, 1, 3, 1, 0], color: .blue),
         Card(points: 1, price: [0, 0, 4, 0, 0], color: .blue),

         // White cards
         Card(points: 0, price: [3, 0, 0, 0, 0], color: .white),
         Card(points: 0, price: [0, 2, 1, 0, 0], color: .white),
         Card(points: 0, price: [1, 1, 1, 0, 1], color: .white),
         Card(points: 0, price: [2, 1, 1, 0, 1], color: .white),
         Card(points: 0, price: [2, 0, 0, 0, 0], color: .white),
         Card(points: 0, price: [2, 0, 2, 0, 0], color: .white),
         Card(points: 0, price: [1, 3, 1, 0, 0], color: .white),
         Card(points: 1, price: [4, 0, 0, 0, 0], color: .white),

         // Brown cards
         Card(points: 0, price: [0, 0, 0, 0, 3], color: .brown),
         Card(points: 0, price: [1, 1, 0, 0, 1], color: .brown),
         Card(points: 0, price: [1, 1, 1, 1, 0], color: .brown),
         Card(points: 0, price: [1, 2, 1, 1, 0], color: .brown),
         Card(points: 0, price: [0, 0, 0, 2, 0], color: .brown),
         Card(points: 0, price: [0, 0, 2, 2, 0], color: .brown),
         Card(points: 0, price: [0, 1, 0, 1, 3], color: .brown),
         Card(points: 1, price: [0, 0, 0, 0, 4], color: .brown)
      ]
   }

   public static func tier2Cards () -> [Card] {
      // Tier 2: 30 cards, 1-4 points, medium costs
      return [
         // Red cards
         Card(points: 1, price: [0, 2, 2, 3, 0], color: .red),
         Card(points: 1, price: [0, 3, 2, 2, 0], color: .red),
         Card(points: 2, price: [0, 5, 0, 0, 0], color: .red),
         Card(points: 2, price: [0, 0, 1, 4, 2], color: .red),
         Card(points: 2, price: [0, 0, 0, 5, 3], color: .red),
         Card(points: 3, price: [0, 6, 0, 0, 0], color: .red),

         // Green cards
         Card(points: 1, price: [2, 0, 3, 0, 2], color: .green),
         Card(points: 1, price: [3, 0, 0, 2, 2], color: .green),
         Card(points: 2, price: [0, 0, 5, 0, 0], color: .green),
         Card(points: 2, price: [2, 0, 0, 1, 4], color: .green),
         Card(points: 2, price: [3, 0, 0, 0, 5], color: .green),
         Card(points: 3, price: [0, 0, 6, 0, 0], color: .green),

         // Blue cards
         Card(points: 1, price: [0, 2, 0, 2, 3], color: .blue),
         Card(points: 1, price: [2, 3, 0, 0, 2], color: .blue),
         Card(points: 2, price: [0, 0, 0, 5, 0], color: .blue),
         Card(points: 2, price: [4, 1, 0, 0, 2], color: .blue),
         Card(points: 2, price: [5, 3, 0, 0, 0], color: .blue),
         Card(points: 3, price: [0, 0, 0, 6, 0], color: .blue),

         // White cards
         Card(points: 1, price: [2, 0, 2, 0, 3], color: .white),
         Card(points: 1, price: [2, 2, 0, 3, 0], color: .white),
         Card(points: 2, price: [5, 0, 0, 0, 0], color: .white),
         Card(points: 2, price: [4, 2, 0, 0, 1], color: .white),
         Card(points: 2, price: [0, 5, 3, 0, 0], color: .white),
         Card(points: 3, price: [6, 0, 0, 0, 0], color: .white),

         // Brown cards
         Card(points: 1, price: [3, 2, 2, 0, 0], color: .brown),
         Card(points: 1, price: [0, 3, 2, 3, 0], color: .brown),
         Card(points: 2, price: [0, 0, 0, 0, 5], color: .brown),
         Card(points: 2, price: [1, 4, 2, 0, 0], color: .brown),
         Card(points: 2, price: [0, 0, 5, 3, 0], color: .brown),
         Card(points: 3, price: [0, 0, 0, 0, 6], color: .brown)
      ]
   }

   public static func tier3Cards () -> [Card] {
      // Tier 3: 20 cards, 3-5 points, expensive costs
      return [
         // Red cards
         Card(points: 3, price: [0, 3, 3, 5, 3], color: .red),
         Card(points: 4, price: [0, 7, 0, 0, 0], color: .red),
         Card(points: 4, price: [0, 0, 0, 7, 3], color: .red),
         Card(points: 5, price: [0, 7, 3, 0, 0], color: .red),

         // Green cards
         Card(points: 3, price: [3, 0, 3, 3, 5], color: .green),
         Card(points: 4, price: [0, 0, 7, 0, 0], color: .green),
         Card(points: 4, price: [3, 0, 0, 0, 7], color: .green),
         Card(points: 5, price: [0, 0, 7, 3, 0], color: .green),

         // Blue cards
         Card(points: 3, price: [5, 3, 0, 3, 3], color: .blue),
         Card(points: 4, price: [0, 0, 0, 7, 0], color: .blue),
         Card(points: 4, price: [7, 3, 0, 0, 0], color: .blue),
         Card(points: 5, price: [3, 0, 0, 7, 0], color: .blue),

         // White cards
         Card(points: 3, price: [3, 5, 3, 0, 3], color: .white),
         Card(points: 4, price: [7, 0, 0, 0, 0], color: .white),
         Card(points: 4, price: [0, 7, 0, 3, 0], color: .white),
         Card(points: 5, price: [0, 3, 7, 0, 0], color: .white),

         // Brown cards
         Card(points: 3, price: [3, 3, 5, 3, 0], color: .brown),
         Card(points: 4, price: [0, 0, 0, 0, 7], color: .brown),
         Card(points: 4, price: [0, 0, 3, 7, 0], color: .brown),
         Card(points: 5, price: [0, 0, 0, 3, 7], color: .brown)
      ]
   }

   // Encode card state as a fixed-size array of Float16
   // Size: 1 (points) + 5 (price) + 5 (color one-hot) = 11
   public static let ENCODED_SIZE = 11

   public func encoding () -> [Float16] {
      var encoded: [Float16] = []

      // 1 point value (normalized by 10.0 to match gem normalization)
      encoded.append(Float16(self.points) / 10.0)

      // 5 price values (normalized by 10.0)
      encoded.append(contentsOf: self.price.map { Float16($0) / 10.0 })

      // 5 color one-hot encoding
      var colorOneHot = Array(repeating: Float16(0), count: GemType.allCases.count)
      colorOneHot[self.color.rawValue] = 1.0
      encoded.append(contentsOf: colorOneHot)

      precondition(encoded.count == Card.ENCODED_SIZE, "Encoded size mismatch: expected \(Card.ENCODED_SIZE), got \(encoded.count)")
      return encoded
   }
}

