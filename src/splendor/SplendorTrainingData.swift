import Foundation
import Compression

/// A single training example collected during a game
public struct TrainingExample: Codable {
   public let turnNumber: Int
   public let playerIndex: Int
   public let state: [Float]  // 361-dimensional game state encoding
   public let policy: [Float]  // 48-dimensional policy target (probability distribution over moves)
   public let value: Float  // Value target from this player's perspective (-1, 0, or 1)

   public init (turnNumber: Int, playerIndex: Int, state: [Float], policy: [Float], value: Float) {
      self.turnNumber = turnNumber
      self.playerIndex = playerIndex
      self.state = state
      self.policy = policy
      self.value = value
   }
}

/// Data from a single self-play game
public struct GameData: Codable {
   public let gameIndex: Int
   public let seed: UInt64
   public let playerCount: Int
   public let winner: Int?  // Player index who won, or nil if tied
   public let turnCount: Int
   public let examples: [TrainingExample]
   public let moves: [(playerIndex: Int, moveIndex: Int)]  // Sequence of (player, move) for statistics

   enum CodingKeys: String, CodingKey {
      case gameIndex, seed, playerCount, winner, turnCount, examples, moves
   }

   public func encode (to encoder: Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(gameIndex, forKey: .gameIndex)
      try container.encode(seed, forKey: .seed)
      try container.encode(playerCount, forKey: .playerCount)
      try container.encode(winner, forKey: .winner)
      try container.encode(turnCount, forKey: .turnCount)
      try container.encode(examples, forKey: .examples)

      // Encode tuples as array of dicts
      let movesArray = moves.map { ["player": $0.playerIndex, "move": $0.moveIndex] }
      try container.encode(movesArray, forKey: .moves)
   }

   public init (from decoder: Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      gameIndex = try container.decode(Int.self, forKey: .gameIndex)
      seed = try container.decode(UInt64.self, forKey: .seed)
      playerCount = try container.decode(Int.self, forKey: .playerCount)
      winner = try container.decode(Int?.self, forKey: .winner)
      turnCount = try container.decode(Int.self, forKey: .turnCount)
      examples = try container.decode([TrainingExample].self, forKey: .examples)

      let movesArray = try container.decode([[String: Int]].self, forKey: .moves)
      moves = movesArray.map { (playerIndex: $0["player"]!, moveIndex: $0["move"]!) }
   }

   public init (gameIndex: Int, seed: UInt64, playerCount: Int, winner: Int?, turnCount: Int, examples: [TrainingExample], moves: [(playerIndex: Int, moveIndex: Int)]) {
      self.gameIndex = gameIndex
      self.seed = seed
      self.playerCount = playerCount
      self.winner = winner
      self.turnCount = turnCount
      self.examples = examples
      self.moves = moves
   }
}

/// Container for all training data
public struct TrainingDataset: Codable {
   public let generatedAt: String
   public let modelPath: String?
   public let temperature: Float
   public let totalGames: Int
   public let totalExamples: Int
   public let games: [GameData]

   public init (generatedAt: String, modelPath: String?, temperature: Float, totalGames: Int, totalExamples: Int, games: [GameData]) {
      self.generatedAt = generatedAt
      self.modelPath = modelPath
      self.temperature = temperature
      self.totalGames = totalGames
      self.totalExamples = totalExamples
      self.games = games
   }

   /// Load training dataset from file or directory
   public static func load (from path: String) throws -> TrainingDataset {
      let url = URL(fileURLWithPath: path)
      var fileURLs: [URL] = []

      var isDirectory: ObjCBool = false
      if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) {
         if isDirectory.boolValue {
            // Load all data files from directory
            let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            fileURLs = files.filter { $0.pathExtension == "json" || $0.pathExtension == "gz" }
         } else {
            // Single file
            fileURLs = [url]
         }
      } else {
         throw NSError(domain: "TrainingDataset", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Training data path does not exist: \(path)"])
      }

      guard !fileURLs.isEmpty else {
         throw NSError(domain: "TrainingDataset", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "No data files found in: \(path)"])
      }

      // Load and merge all datasets
      var allGames: [GameData] = []
      for fileURL in fileURLs {
         let dataset = try Self.loadSingleFile(from: fileURL)
         allGames.append(contentsOf: dataset.games)
      }

      // Create merged dataset
      let totalExamples = allGames.reduce(0) { $0 + $1.examples.count }
      return TrainingDataset(
         generatedAt: Date().iso8601,
         modelPath: nil,
         temperature: 0.0, // Not meaningful for merged dataset
         totalGames: allGames.count,
         totalExamples: totalExamples,
         games: allGames
      )
   }

   /// Save dataset to compressed JSON format
   public func save (to path: String, compress: Bool = true) throws {
      let finalPath = compress ? (path + ".gz") : (path + ".json")
      let url = URL(fileURLWithPath: finalPath)
      let dir = url.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)

      let encoder = JSONEncoder()
      let jsonData = try encoder.encode(self)

      if compress {
         let compressed = try compressData(jsonData)
         try compressed.write(to: url)
         print("File written successfully to \(finalPath) (\(jsonData.count) bytes -> \(compressed.count) bytes).")
      } else {
         try jsonData.write(to: url)
         print("File written successfully to \(finalPath) (\(jsonData.count) bytes).")
      }
   }


   /// Load a single file (compressed or uncompressed JSON)
   private static func loadSingleFile (from url: URL) throws -> TrainingDataset {
      var data = try Data(contentsOf: url)

      // Decompress if .gz extension
      if url.pathExtension == "gz" {
         data = try decompressData(data)
      }

      let decoder = JSONDecoder()
      return try decoder.decode(TrainingDataset.self, from: data)
   }
}

// MARK: - Date extension

extension Date {
   var iso8601: String {
      let formatter = ISO8601DateFormatter()
      return formatter.string(from: self)
   }
}

// MARK: - Compression helpers

fileprivate func compressData (_ data: Data) throws -> Data {
   return try data.withUnsafeBytes { bytes in
      guard let baseAddress = bytes.baseAddress else {
         throw NSError(domain: "TrainingDataset", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "Invalid data buffer"])
      }
      let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
      let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
      defer { destinationBuffer.deallocate() }

      let compressedSize = compression_encode_buffer(
         destinationBuffer, data.count,
         buffer, data.count,
         nil, COMPRESSION_LZ4)

      guard compressedSize > 0 else {
         throw NSError(domain: "TrainingDataset", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "Compression failed"])
      }

      return Data(bytes: destinationBuffer, count: compressedSize)
   }
}

fileprivate func decompressData (_ data: Data) throws -> Data {
   return try data.withUnsafeBytes { bytes in
      guard let baseAddress = bytes.baseAddress else {
         throw NSError(domain: "TrainingDataset", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "Invalid data buffer"])
      }
      let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

      // Start with a large buffer - JSON compresses well so ratio can be high
      var capacity = data.count * 20
      var destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)

      var decompressedSize = compression_decode_buffer(
         destinationBuffer, capacity,
         buffer, data.count,
         nil, COMPRESSION_LZ4)

      // If buffer was too small (returns 0), try with larger buffers
      while decompressedSize == 0 && capacity < 500_000_000 {
         destinationBuffer.deallocate()
         capacity *= 2
         destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
         decompressedSize = compression_decode_buffer(
            destinationBuffer, capacity,
            buffer, data.count,
            nil, COMPRESSION_LZ4)
      }

      guard decompressedSize > 0 else {
         destinationBuffer.deallocate()
         throw NSError(domain: "TrainingDataset", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "Decompression failed: capacity=\(capacity), inputSize=\(data.count)"])
      }

      let result = Data(bytes: destinationBuffer, count: decompressedSize)
      destinationBuffer.deallocate()
      return result
   }
}
