import Foundation
import Compression

/// A single training example collected during a game
public struct TrainingExample: Codable {
   public let turnNumber: Int
   public let playerIndex: Int
   public let state: [Float]  // 360-dimensional game state encoding
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

   // MARK: - Binary format constants

   /// Magic bytes to identify binary training data files
   private static let MAGIC: UInt32 = 0x4F52494E  // "ORIN"
   private static let FORMAT_VERSION: UInt32 = 1
   private static let STATE_DIM: Int = 357
   private static let POLICY_DIM: Int = 48
   /// Bytes per example: (360 + 48 + 1) * 4 = 1636
   private static let BYTES_PER_EXAMPLE: Int = (STATE_DIM + POLICY_DIM + 1) * MemoryLayout<Float>.size

   // MARK: - Load

   /// Load training dataset from file or directory
   public static func load (from path: String) throws -> TrainingDataset {
      let url = URL(fileURLWithPath: path)
      var fileURLs: [URL] = []

      var isDirectory: ObjCBool = false
      if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory) {
         if isDirectory.boolValue {
            let files = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            fileURLs = files.filter {
               let ext = $0.pathExtension
               return ext == "json" || ext == "gz" || ext == "lz4"
            }
         } else {
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

      let totalExamples = allGames.reduce(0) { $0 + $1.examples.count }
      return TrainingDataset(
         generatedAt: Date().iso8601,
         modelPath: nil,
         temperature: 0.0,
         totalGames: allGames.count,
         totalExamples: totalExamples,
         games: allGames
      )
   }

   // MARK: - Save (binary)

   /// Save dataset in binary format with LZ4 compression.
   ///
   /// Binary layout (before compression):
   ///   [4 bytes] magic "ORIN"
   ///   [4 bytes] format version (1)
   ///   [4 bytes] state dimensions (360)
   ///   [4 bytes] policy dimensions (48)
   ///   [4 bytes] total games (UInt32)
   ///   [4 bytes] total examples (UInt32)
   ///   [N * BYTES_PER_EXAMPLE bytes] packed examples:
   ///       [360 * Float32] state
   ///       [48 * Float32]  policy
   ///       [1 * Float32]   value
   public func save (to path: String, compress: Bool = true) throws {
      let finalPath = path + ".bin.lz4"
      let url = URL(fileURLWithPath: finalPath)
      let dir = url.deletingLastPathComponent()
      try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true, attributes: nil)

      let exampleCount = games.reduce(0) { $0 + $1.examples.count }
      let headerSize = 6 * MemoryLayout<UInt32>.size  // 24 bytes
      let bodySize = exampleCount * Self.BYTES_PER_EXAMPLE
      let totalSize = headerSize + bodySize

      var buffer = Data(count: totalSize)

      buffer.withUnsafeMutableBytes { raw in
         let ptr = raw.baseAddress!

         // Header
         ptr.storeBytes(of: Self.MAGIC, as: UInt32.self)
         ptr.storeBytes(of: Self.FORMAT_VERSION, toByteOffset: 4, as: UInt32.self)
         ptr.storeBytes(of: UInt32(Self.STATE_DIM), toByteOffset: 8, as: UInt32.self)
         ptr.storeBytes(of: UInt32(Self.POLICY_DIM), toByteOffset: 12, as: UInt32.self)
         ptr.storeBytes(of: UInt32(totalGames), toByteOffset: 16, as: UInt32.self)
         ptr.storeBytes(of: UInt32(exampleCount), toByteOffset: 20, as: UInt32.self)

         // Body — pack examples contiguously
         var offset = headerSize
         for game in games {
            for example in game.examples {
               // State: 360 floats
               for f in example.state {
                  ptr.storeBytes(of: f, toByteOffset: offset, as: Float.self)
                  offset += 4
               }
               // Policy: 48 floats
               for f in example.policy {
                  ptr.storeBytes(of: f, toByteOffset: offset, as: Float.self)
                  offset += 4
               }
               // Value: 1 float
               ptr.storeBytes(of: example.value, toByteOffset: offset, as: Float.self)
               offset += 4
            }
         }
      }

      let compressed = try compressData(buffer)
      try compressed.write(to: url)
      print("File written successfully to \(finalPath) (\(totalSize) bytes -> \(compressed.count) bytes).")
   }

   // MARK: - Load single file (format dispatch)

   private static func loadSingleFile (from url: URL) throws -> TrainingDataset {
      if url.pathExtension == "lz4" {
         return try loadBinaryFile(from: url)
      } else {
         return try loadJSONFile(from: url)
      }
   }

   /// Load legacy JSON format (.json or .gz)
   private static func loadJSONFile (from url: URL) throws -> TrainingDataset {
      var data = try Data(contentsOf: url)
      if url.pathExtension == "gz" {
         data = try decompressData(data)
      }
      let decoder = JSONDecoder()
      return try decoder.decode(TrainingDataset.self, from: data)
   }

   /// Load binary format (.bin.lz4)
   private static func loadBinaryFile (from url: URL) throws -> TrainingDataset {
      let compressed = try Data(contentsOf: url)
      let data = try decompressData(compressed)

      return try data.withUnsafeBytes { raw in
         let ptr = raw.baseAddress!
         let totalBytes = raw.count

         guard totalBytes >= 24 else {
            throw NSError(domain: "TrainingDataset", code: 6,
               userInfo: [NSLocalizedDescriptionKey: "Binary file too small: \(totalBytes) bytes"])
         }

         let magic = ptr.load(as: UInt32.self)
         guard magic == MAGIC else {
            throw NSError(domain: "TrainingDataset", code: 6,
               userInfo: [NSLocalizedDescriptionKey: "Bad magic: expected 0x\(String(MAGIC, radix: 16)), got 0x\(String(magic, radix: 16))"])
         }

         let version = ptr.load(fromByteOffset: 4, as: UInt32.self)
         guard version == FORMAT_VERSION else {
            throw NSError(domain: "TrainingDataset", code: 6,
               userInfo: [NSLocalizedDescriptionKey: "Unsupported format version: \(version)"])
         }

         let stateDim = Int(ptr.load(fromByteOffset: 8, as: UInt32.self))
         let policyDim = Int(ptr.load(fromByteOffset: 12, as: UInt32.self))
         let gameCount = Int(ptr.load(fromByteOffset: 16, as: UInt32.self))
         let exampleCount = Int(ptr.load(fromByteOffset: 20, as: UInt32.self))

         let bytesPerExample = (stateDim + policyDim + 1) * MemoryLayout<Float>.size
         let expectedSize = 24 + exampleCount * bytesPerExample
         guard totalBytes >= expectedSize else {
            throw NSError(domain: "TrainingDataset", code: 6,
               userInfo: [NSLocalizedDescriptionKey: "Binary file truncated: expected \(expectedSize) bytes, got \(totalBytes)"])
         }

         // Read all examples into a single flat game (structure doesn't matter for training)
         var examples: [TrainingExample] = []
         examples.reserveCapacity(exampleCount)

         var offset = 24
         for _ in 0..<exampleCount {
            let statePtr = (ptr + offset).assumingMemoryBound(to: Float.self)
            let state = Array(UnsafeBufferPointer(start: statePtr, count: stateDim))
            offset += stateDim * 4

            let policyPtr = (ptr + offset).assumingMemoryBound(to: Float.self)
            let policy = Array(UnsafeBufferPointer(start: policyPtr, count: policyDim))
            offset += policyDim * 4

            let value = (ptr + offset).load(as: Float.self)
            offset += 4

            examples.append(TrainingExample(
               turnNumber: 0, playerIndex: 0,
               state: state, policy: policy, value: value))
         }

         let game = GameData(
            gameIndex: 0, seed: 0, playerCount: 2, winner: nil,
            turnCount: examples.count, examples: examples, moves: [])

         return TrainingDataset(
            generatedAt: "", modelPath: nil, temperature: 0.0,
            totalGames: gameCount, totalExamples: exampleCount,
            games: [game])
      }
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
      // LZ4 can expand incompressible data; worst case is input + input/255 + 16
      let capacity = data.count + data.count / 255 + 16
      let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
      defer { destinationBuffer.deallocate() }

      let compressedSize = compression_encode_buffer(
         destinationBuffer, capacity,
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

      // Start with a large buffer - training data can be large
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
