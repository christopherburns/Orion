import Foundation
import MLX
import MLXNN
import Core

/// Model metadata for versioning and tracking
public struct ModelMetadata: Codable {
   public let version: String
   public let architectureVersion: Int
   public let createdAt: Date
   public let trainingEpochs: Int?
   public let trainingLoss: Float?
   public let description: String?
   public let checksum: String? // SHA256 checksum of weights

   public init (
      version: String, architectureVersion: Int, createdAt: Date = Date(),
      trainingEpochs: Int? = nil, trainingLoss: Float? = nil,
      description: String? = nil, checksum: String? = nil) {

      self.version = version
      self.architectureVersion = architectureVersion
      self.createdAt = createdAt
      self.trainingEpochs = trainingEpochs
      self.trainingLoss = trainingLoss
      self.description = description
      self.checksum = checksum
   }
}

/// Neural network for Splendor game playing — architecture v10.
///
/// v10 replaces the flat MLP's positional reading of the 12 visible-card
/// blocks with a SHARED CARD ENCODER: one small MLP (12 → 32 → 32) applied
/// with the same weights to every visible card slot (equivalently, a 1×1
/// convolution over the card axis / the Deep Sets construction). Its outputs
/// feed two consumers:
///
///  · pooled (mean over the 12 embeddings) into the trunk as an
///    order-invariant board summary — the value head's view of the market;
///  · un-pooled through a shared per-card readout (32 → 1) producing each
///    card's PURCHASE logit directly. Canonical purchase moves 0..<12 and the
///    encoding's card blocks share the same 4*tier+position ordering, so
///    concatenating [per-card logits, trunk logits] IS the correct scatter.
///
/// The remaining 36 moves (purchase-reserved, takes, reserves, discards) come
/// off the trunk as before. Reserved-card routing, role flags, and the e⊕h
/// context-concat are deliberately deferred — this is the minimal cut that
/// tests whether shared card reading sharpens purchase play.
public class PolicyValueNetwork: Module {

   public static let INPUT_DIMENSIONS = Game.GAME_STATE_ENCODING_SIZE  // 500
   public static let POLICY_DIMENSIONS = Game.CANONICAL_MOVE_COUNT     // 48
   public static let HIDDEN_DIMENSIONS = 512
   public static let CARD_EMBED_DIMENSIONS = 32
   public static let DEFAULT_DROPOUT: Float = 0.1

   // Encoding-layout facts the forward pass depends on, derived from the same
   // constants Game.encoding() uses so they cannot silently drift apart.
   // Card block start = 4 players × PlayerState.ENCODED_SIZE, then supply(5),
   // gold(1), take-N yields(15), nobles(130).
   static let VISIBLE_CARD_COUNT = 12                        // 3 tiers × 4 positions
   static let CARD_FEATURES = Card.ENCODED_SIZE              // 12 floats per card
   static let CARD_BLOCK_START = 4 * PlayerState.ENCODED_SIZE + 5 + 1 + 15 + 130           // 355
   static let CARD_BLOCK_END = CARD_BLOCK_START + VISIBLE_CARD_COUNT * CARD_FEATURES       // 499
   static let NON_CARD_DIMENSIONS = INPUT_DIMENSIONS - VISIBLE_CARD_COUNT * CARD_FEATURES  // 356
   static let TRUNK_INPUT_DIMENSIONS = NON_CARD_DIMENSIONS + CARD_EMBED_DIMENSIONS         // 388

   // Canonical move layout: purchase-visible moves are exactly indices 0..<12.
   static let PURCHASE_MOVE_COUNT = 12
   static let TRUNK_MOVE_COUNT = POLICY_DIMENSIONS - PURCHASE_MOVE_COUNT  // 36 (moves 12..<48)

   /// Default precision for parameters and forward pass when no override is given.
   /// bfloat16 unlocks the M-series matrix-multiply hardware for large GEMMs
   /// (~1.6× wall-clock for training) while keeping fp32's exponent range. Inference
   /// paths with small per-call batches currently run faster in fp32 — the
   /// `--precision` CLI flag on each command picks per-workload.
   public static let DEFAULT_PRECISION: DType = .bfloat16

   /// Per-instance precision used for weights and forward-pass casts.
   public let precision: DType

   // Current architecture version - increment when architecture changes
   public static let ARCHITECTURE_VERSION = 10

   // Shared card encoder (same weights applied to each visible-card slot)
   let cardEncoder1: Linear     // 12 → 32
   let cardEncoder2: Linear     // 32 → 32

   // Shared trunk layers
   let dense1: Linear           // 388 → 512
   let dense2: Linear           // 512 → 512
   let dense3: Linear           // 512 → 512

   // Dropout for regularization
   let dropout1: Dropout
   let dropout2: Dropout
   let dropout3: Dropout

   /// The dropout rate used by this network instance (stored for serialization)
   public let dropoutRate: Float

   // Policy: shared per-card purchase readout + trunk head for the other moves
   let policyCardReadout: Linear   // 32 → 1, same weights for every card slot
   let policyTrunkHead: Linear     // 512 → 36 (canonical moves 12..<48)

   // Value head (outputs win probability)
   let valueHidden: Linear      // 512 → 128
   let valueOutput: Linear      // 128 → 1

   /// Initialize network with optional seed for deterministic weight initialization
   /// - Parameters:
   ///   - seed: If provided, weights will be initialized deterministically
   ///   - dropoutRate: Dropout probability for trunk layers (default: 0.1)
   ///   - precision: dtype for weights, biases, and forward-pass casts
   public init (seed: UInt64? = nil, dropoutRate: Float = DEFAULT_DROPOUT, precision: DType = DEFAULT_PRECISION) {
      self.dropoutRate = dropoutRate
      self.precision = precision

      // Create deterministic keys if seed provided (one per weight matrix)
      let layerCount = 9
      let keys: [MLXArray]
      if let seed = seed {
         let baseKey = MLXRandom.key(seed)
         keys = MLXRandom.split(key: baseKey, into: layerCount)
      } else {
         keys = Array(repeating: MLXArray(0), count: layerCount) // Dummy keys, will use nil
      }

      func heLinear (_ keyIndex: Int, _ inDim: Int, _ outDim: Int) -> Linear {
         Linear(weight: PolicyValueNetwork.heInitialization(
                   inputDimensions: inDim, outputDimensions: outDim,
                   key: seed == nil ? nil : keys[keyIndex], precision: precision),
                bias: MLXArray.zeros([outDim]).asType(precision))
      }

      self.cardEncoder1     = heLinear(0, PolicyValueNetwork.CARD_FEATURES, PolicyValueNetwork.CARD_EMBED_DIMENSIONS)
      self.cardEncoder2     = heLinear(1, PolicyValueNetwork.CARD_EMBED_DIMENSIONS, PolicyValueNetwork.CARD_EMBED_DIMENSIONS)
      self.dense1           = heLinear(2, PolicyValueNetwork.TRUNK_INPUT_DIMENSIONS, PolicyValueNetwork.HIDDEN_DIMENSIONS)
      self.dense2           = heLinear(3, PolicyValueNetwork.HIDDEN_DIMENSIONS, PolicyValueNetwork.HIDDEN_DIMENSIONS)
      self.dense3           = heLinear(4, PolicyValueNetwork.HIDDEN_DIMENSIONS, PolicyValueNetwork.HIDDEN_DIMENSIONS)
      self.policyCardReadout = heLinear(5, PolicyValueNetwork.CARD_EMBED_DIMENSIONS, 1)
      self.policyTrunkHead  = heLinear(6, PolicyValueNetwork.HIDDEN_DIMENSIONS, PolicyValueNetwork.TRUNK_MOVE_COUNT)
      self.valueHidden      = heLinear(7, PolicyValueNetwork.HIDDEN_DIMENSIONS, 128)
      self.valueOutput      = heLinear(8, 128, 1)

      self.dropout1 = Dropout(p: dropoutRate)
      self.dropout2 = Dropout(p: dropoutRate)
      self.dropout3 = Dropout(p: dropoutRate)

      super.init()
   }

   /// Convenience initializer for non-deterministic initialization
   convenience override init () {
      self.init(seed: nil)
   }

   /// Create a deep copy of this network by applying this network's parameters
   /// to a fresh instance. Generic over the parameter set — no hardcoded layer
   /// names, so it survives future topology changes.
   public func clone () -> PolicyValueNetwork {
      let copy = PolicyValueNetwork(seed: nil, dropoutRate: dropoutRate, precision: precision)
      copy.update(parameters: ModuleParameters.unflattened(self.parameters().flattened()))
      return copy
   }

   /// Initialize a linear layer with He initialization
   /// - Parameters:
   ///   - inputDimensions: Number of input features
   ///   - outputDimensions: Number of output features
   ///   - key: Optional PRNG key for deterministic initialization
   /// - Returns: Weight matrix initialized with He normal distribution
   private static func heInitialization (inputDimensions: Int, outputDimensions: Int, key: MLXArray? = nil, precision: DType = DEFAULT_PRECISION) -> MLXArray {
      let stddev = sqrt(2.0 / Float(inputDimensions))
      let arr = MLXRandom.normal([outputDimensions, inputDimensions], key: key) * stddev
      return arr.asType(precision)
   }

   /// Forward pass through the network
   /// - Parameter x: Input tensor of shape [batchSize, INPUT_DIMENSIONS] in any float dtype; cast to `self.precision` on entry.
   /// - Returns: Tuple of (policyLogits [batchSize, 48], value [batchSize, 1] in [-1, 1]). Both are in the network's precision; callers cast as needed.
   public func execute (_ x: MLXArray) -> (policyLogits: MLXArray, value: MLXArray) {
      precondition(x.shape.count == 2, "Input must have shape [batchSize, \(PolicyValueNetwork.INPUT_DIMENSIONS)]")
      precondition(x.shape[1] == PolicyValueNetwork.INPUT_DIMENSIONS, "Input must have \(PolicyValueNetwork.INPUT_DIMENSIONS) features")

      let batch = x.shape[0]
      let xCast = x.dtype == self.precision ? x : x.asType(self.precision)

      // Shared card encoder over the 12 visible-card blocks:
      // [B, 144] → [B, 12, 12] → Linear ▶ ReLU ▶ Linear (over the last axis,
      // weights shared across the card axis) → [B, 12, 32]
      let cardBlock = xCast[0 ..< batch, PolicyValueNetwork.CARD_BLOCK_START ..< PolicyValueNetwork.CARD_BLOCK_END]
         .reshaped([batch, PolicyValueNetwork.VISIBLE_CARD_COUNT, PolicyValueNetwork.CARD_FEATURES])
      let cardEmbeddings = cardEncoder2(relu(cardEncoder1(cardBlock)))

      // Order-invariant board summary — how card information reaches the trunk
      // (and through it, the value head)
      let boardSummary = mean(cardEmbeddings, axis: 1)  // [B, 32]

      // Trunk input: every non-card float, plus the board summary
      let preCard = xCast[0 ..< batch, 0 ..< PolicyValueNetwork.CARD_BLOCK_START]
      let postCard = xCast[0 ..< batch, PolicyValueNetwork.CARD_BLOCK_END ..< PolicyValueNetwork.INPUT_DIMENSIONS]
      var h = concatenated([preCard, postCard, boardSummary], axis: 1)  // [B, 388]

      h = dropout1(relu(dense1(h)))
      h = dropout2(relu(dense2(h)))
      h = dropout3(relu(dense3(h)))

      // Policy: shared 32→1 readout over the card axis fills canonical moves
      // 0..<12; the trunk head fills moves 12..<48. Card block k and purchase
      // move k are both indexed 4*tier + position, so concatenation IS the
      // correct scatter into canonical order.
      let purchaseLogits = policyCardReadout(cardEmbeddings).reshaped([batch, PolicyValueNetwork.PURCHASE_MOVE_COUNT])
      let trunkLogits = policyTrunkHead(h)  // [B, 36]
      let policyLogits = concatenated([purchaseLogits, trunkLogits], axis: 1)  // [B, 48]

      // Value head (tanh activation for [-1, 1] range)
      let valueHiddenOut = relu(self.valueHidden(h))
      let value = tanh(valueOutput(valueHiddenOut))

      return (policyLogits, value)
   }

   /// Save model weights and metadata to disk
   /// - Parameters:
   ///   - url: Directory URL where the model will be saved
   ///   - metadata: Model metadata for versioning
   /// - Throws: Errors from file operations or serialization
   public func save (to url: URL, metadata: ModelMetadata) throws {
      // Create directory if it doesn't exist
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)

      // Get all parameters from the module and flatten them
      let parameters = self.parameters()
      let flattenedParams = parameters.flattened()

      // Convert flattened parameters to JSON-serializable format
      var weightsJSON: [String: Any] = [:]

      for (key, array) in flattenedParams {
         let shape = array.shape
         let count = shape.reduce(1, *)

         // Convert MLXArray to Float array efficiently (bulk conversion)
         let reshaped = array.reshaped([count])
         let floatArray = reshaped.asArray(Float.self)

         // Store as Data for efficient serialization
         let data = floatArray.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.size)
         }

         // Convert to base64 for JSON serialization
         weightsJSON[key] = [
            "data": data.base64EncodedString(),
            "shape": shape
         ]
      }

      let weightsURL = url.appendingPathComponent("weights.json")
      let weightsData = try JSONSerialization.data(withJSONObject: weightsJSON, options: .prettyPrinted)
      try weightsData.write(to: weightsURL)

      // Save metadata as JSON
      let metadataURL = url.appendingPathComponent("metadata.json")
      let metadataEncoder = JSONEncoder()
      metadataEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      metadataEncoder.dateEncodingStrategy = .iso8601
      var metadataData = try metadataEncoder.encode(metadata)
      metadataData.append(0x0A)  // trailing newline
      try metadataData.write(to: metadataURL)

      // Save architecture info (including dropout rate for reproducibility)
      let architectureInfo: [String: Any] = [
         "architectureVersion": PolicyValueNetwork.ARCHITECTURE_VERSION,
         "inputDimensions": PolicyValueNetwork.INPUT_DIMENSIONS,
         "policyDimensions": PolicyValueNetwork.POLICY_DIMENSIONS,
         "cardEmbedDimensions": PolicyValueNetwork.CARD_EMBED_DIMENSIONS,
         "dropoutRate": dropoutRate
      ]
      let architectureURL = url.appendingPathComponent("architecture.json")
      let architectureData = try JSONSerialization.data(withJSONObject: architectureInfo, options: .prettyPrinted)
      try architectureData.write(to: architectureURL)
   }

   /// Factory method to create a network with weights loaded from disk.
   ///
   /// Generic over the parameter set: builds a fresh network, verifies the
   /// saved keys and shapes match its parameters EXACTLY, then applies them.
   /// A mismatch (missing, extra, or misshaped parameter) fails loudly rather
   /// than silently producing a half-loaded network.
   /// - Parameters:
   ///   - url: Directory URL where the model is stored
   ///   - precision: dtype to materialize the loaded weights at
   /// - Returns: Tuple of (loaded network, metadata)
   /// - Throws: Errors from file operations or deserialization
   public static func load (from url: URL, precision: DType = DEFAULT_PRECISION) throws -> (network: PolicyValueNetwork, metadata: ModelMetadata) {
      // Load metadata
      let metadataURL = url.appendingPathComponent("metadata.json")
      let metadataData = try Data(contentsOf: metadataURL)
      let metadataDecoder = JSONDecoder()
      metadataDecoder.dateDecodingStrategy = .iso8601
      let metadata = try metadataDecoder.decode(ModelMetadata.self, from: metadataData)

      // Verify architecture version matches
      let architectureURL = url.appendingPathComponent("architecture.json")
      let architectureData = try Data(contentsOf: architectureURL)
      let architectureInfo = try JSONSerialization.jsonObject(with: architectureData) as! [String: Any]
      let savedArchitectureVersion = architectureInfo["architectureVersion"] as! Int

      guard savedArchitectureVersion == PolicyValueNetwork.ARCHITECTURE_VERSION else {
         throw NSError(domain: "PolicyValueNetwork", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Architecture version mismatch: saved \(savedArchitectureVersion), current \(PolicyValueNetwork.ARCHITECTURE_VERSION)"])
      }

      // Read dropout rate (default to 0.1 for models saved before this field existed)
      let savedDropoutRate = (architectureInfo["dropoutRate"] as? NSNumber)?.floatValue ?? DEFAULT_DROPOUT

      // Load weights
      let weightsURL = url.appendingPathComponent("weights.json")
      let weightsData = try Data(contentsOf: weightsURL)
      guard let weightsDict = try JSONSerialization.jsonObject(with: weightsData) as? [String: [String: Any]] else {
         throw NSError(domain: "PolicyValueNetwork", code: 2,
                      userInfo: [NSLocalizedDescriptionKey: "Failed to deserialize weights"])
      }

      // Decode every saved parameter
      var loaded: [(String, MLXArray)] = []
      loaded.reserveCapacity(weightsDict.count)
      for (key, info) in weightsDict {
         guard let base64Data = info["data"] as? String,
               let shape = info["shape"] as? [Int],
               let data = Data(base64Encoded: base64Data) else {
            throw NSError(domain: "PolicyValueNetwork", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Malformed weight entry for parameter: \(key)"])
         }
         let count = shape.reduce(1, *)
         let floatArray = data.withUnsafeBytes { bytes in
            Array(UnsafeBufferPointer<Float>(start: bytes.baseAddress?.assumingMemoryBound(to: Float.self),
                                             count: count))
         }
         loaded.append((key, MLXArray(floatArray).reshaped(shape).asType(precision)))
      }

      let network = PolicyValueNetwork(seed: nil, dropoutRate: savedDropoutRate, precision: precision)

      // Strict key/shape verification against the fresh network's parameter set
      let expected = Dictionary(uniqueKeysWithValues: network.parameters().flattened().map { ($0.0, $0.1.shape) })
      let loadedKeys = Set(loaded.map { $0.0 })
      let expectedKeys = Set(expected.keys)
      guard loadedKeys == expectedKeys else {
         let missing = expectedKeys.subtracting(loadedKeys).sorted()
         let extra = loadedKeys.subtracting(expectedKeys).sorted()
         throw NSError(domain: "PolicyValueNetwork", code: 5,
                      userInfo: [NSLocalizedDescriptionKey: "Weight keys do not match architecture. Missing: \(missing). Unexpected: \(extra)."])
      }
      for (key, array) in loaded {
         guard array.shape == expected[key] else {
            throw NSError(domain: "PolicyValueNetwork", code: 5,
                         userInfo: [NSLocalizedDescriptionKey: "Shape mismatch for \(key): saved \(array.shape), expected \(expected[key]!)"])
         }
      }

      network.update(parameters: ModuleParameters.unflattened(loaded))
      return (network, metadata)
   }
}


/// Neural network agent that uses the SplendorNetwork to play Splendor
public class SplendorNeuralAgent: AgentProtocol {

   private let network: PolicyValueNetwork
   private let metadata: ModelMetadata

   /// Initialize with untrained network
   /// - Parameters:
   ///   - seed: Optional seed for deterministic weight initialization
   ///   - precision: dtype for parameters and forward pass
   public init (seed: UInt64? = nil, precision: DType = PolicyValueNetwork.DEFAULT_PRECISION) {
      self.network = PolicyValueNetwork(seed: seed, precision: precision)
      self.metadata = ModelMetadata(version: "0.1.0", architectureVersion: PolicyValueNetwork.ARCHITECTURE_VERSION)
   }

   /// Initialize by loading a trained model from disk
   /// - Parameters:
   ///   - url: Directory URL where the model is stored
   ///   - precision: dtype to materialize the loaded weights at
   public init (url: URL, precision: DType = PolicyValueNetwork.DEFAULT_PRECISION) throws {
      let (network, metadata) = try PolicyValueNetwork.load(from: url, precision: precision)
      self.network = network
      self.metadata = metadata
   }

   /// Create an MCTSSearch backed by this agent's network.
   public func makeMCTSSearch (monteCarloSamples: Int, cPuct: Float, debug: Bool = false) -> MCTSSearch {
      MCTSSearch(agent: self, monteCarloSamples: monteCarloSamples, cPuct: cPuct, debug: debug)
   }

   public func prepareForInference () {
      network.train(false)
   }

   public func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      // Cast to Splendor.Game to access the encoding() method
      guard let splendorGame = game as? Splendor.Game else {
         preconditionFailure("SplendorNeuralAgent can only be used with Splendor.Game")
      }

      // Get the encoded game state, convert to Float array
      let encodedState = splendorGame.encoding().map { Float($0) }

      // Create MLX array with shape [1, INPUT_DIMENSIONS]
      let inputArray = MLXArray(encodedState).reshaped([1, PolicyValueNetwork.INPUT_DIMENSIONS])

      // Run inference
      let (policyLogits, value) = network.execute(inputArray)

      // Policy logits should have shape [1, 48]
      precondition(policyLogits.ndim == 2, "Policy logits must be 2D, got \(policyLogits.ndim)D")
      precondition(policyLogits.shape[1] == PolicyValueNetwork.POLICY_DIMENSIONS, "Policy logits must have \(PolicyValueNetwork.POLICY_DIMENSIONS) moves, got \(policyLogits.shape[1])")

      // Value should have shape [1, 1]
      precondition(value.ndim == 2, "Value must be 2D, got \(value.ndim)D")
      precondition(value.shape[1] == 1, "Value must have 1 element in second dimension, got \(value.shape[1])")

      // Convert policy logits to Swift array efficiently (single eval + device sync)
      let logitsArray = policyLogits[0]  // Extract [48] from [1, 48]
      let policyResult = logitsArray.asArray(Float.self)

      // Extract value estimate efficiently (already in [-1, 1] range from tanh activation)
      let valueArray = value.asArray(Float.self)
      let valueEstimate = valueArray[0]

      return (policyResult, valueEstimate)
   }

   public func batchPredict (games: [any GameProtocol], currentPlayerIndices: [Int]) -> [(policyLogits: [Float], valueEstimate: Float)] {
      let batchSize = games.count
      if batchSize == 0 { return [] }

      let inputDim = PolicyValueNetwork.INPUT_DIMENSIONS
      let policyDim = PolicyValueNetwork.POLICY_DIMENSIONS

      var batchFloats = [Float](repeating: 0, count: batchSize * inputDim)
      for (j, game) in games.enumerated() {
         let splendorGame = game as! Splendor.Game
         let encoding = splendorGame.encoding()
         let base = j * inputDim
         for k in 0..<inputDim {
            batchFloats[base + k] = Float(encoding[k])
         }
      }

      let inputArray = MLXArray(batchFloats).reshaped([batchSize, inputDim])
      let (policyTensor, valueTensor) = network.execute(inputArray)
      eval(policyTensor, valueTensor)

      let flatLogits = policyTensor.asArray(Float.self)
      let flatValues = valueTensor.asArray(Float.self)

      var results: [(policyLogits: [Float], valueEstimate: Float)] = []
      results.reserveCapacity(batchSize)
      for j in 0..<batchSize {
         let base = j * policyDim
         results.append((Array(flatLogits[base..<base + policyDim]), flatValues[j]))
      }
      return results
   }
}
