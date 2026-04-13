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

/// Neural network for Splendor game playing
/// Architecture:
///   Input (361) -> Dense(512) -> Dense(512) -> Dense(256) -> Policy Head (48) + Value Head (1)
public class PolicyValueNetwork: Module {

   public static let INPUT_DIMENSIONS = 357 // Matches game's state embedding size (rotated players, no current-player one-hot)
   public static let POLICY_DIMENSIONS = 48 // Matches game's move space (42 normal + 6 discard)
   public static let HIDDEN_DIMENSIONS = 512
   public static let DEFAULT_DROPOUT: Float = 0.1

   // Current architecture version - increment when architecture changes
   public static let ARCHITECTURE_VERSION = 4

   // Shared trunk layers
   let dense1: Linear
   let dense2: Linear
   let dense3: Linear

   // Dropout for regularization
   let dropout1: Dropout
   let dropout2: Dropout
   let dropout3: Dropout

   /// The dropout rate used by this network instance (stored for serialization)
   public let dropoutRate: Float

   // Policy head (outputs move logits)
   let policyHead: Linear

   // Value head (outputs win probability)
   let valueHidden: Linear
   let valueOutput: Linear

   /// Initialize network with optional seed for deterministic weight initialization
   /// - Parameter seed: If provided, weights will be initialized deterministically
   /// - Parameter dropoutRate: Dropout probability for trunk layers (default: 0.1)
   public init (seed: UInt64? = nil, dropoutRate: Float = DEFAULT_DROPOUT) {
      self.dropoutRate = dropoutRate

      // Create deterministic key if seed provided
      let keys: [MLXArray]
      if let seed = seed {
         // Create base key and split into 6 keys (one for each layer)
         let baseKey = MLXRandom.key(seed)
         keys = MLXRandom.split(key: baseKey, into: 6)
      } else {
         keys = Array(repeating: MLXArray(0), count: 6) // Dummy keys, will use nil
      }

      // Shared trunk: 361 -> 512 -> 512 -> 512
      self.dense1 = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: PolicyValueNetwork.INPUT_DIMENSIONS, outputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, key: seed == nil ? nil : keys[0]),
         bias: MLXArray.zeros([PolicyValueNetwork.HIDDEN_DIMENSIONS]))
      self.dense2 = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, outputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, key: seed == nil ? nil : keys[1]),
         bias: MLXArray.zeros([PolicyValueNetwork.HIDDEN_DIMENSIONS]))
      self.dense3 = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, outputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, key: seed == nil ? nil : keys[2]),
         bias: MLXArray.zeros([PolicyValueNetwork.HIDDEN_DIMENSIONS]))

      // Dropout layers
      self.dropout1 = Dropout(p: dropoutRate)
      self.dropout2 = Dropout(p: dropoutRate)
      self.dropout3 = Dropout(p: dropoutRate)

      // Policy head: 512 -> 48 logits
      self.policyHead = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, outputDimensions: PolicyValueNetwork.POLICY_DIMENSIONS, key: seed == nil ? nil : keys[3]),
         bias: MLXArray.zeros([PolicyValueNetwork.POLICY_DIMENSIONS]))

      // Value head: 256 -> 128 -> 1
      self.valueHidden = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: PolicyValueNetwork.HIDDEN_DIMENSIONS, outputDimensions: 128, key: seed == nil ? nil : keys[4]), bias: MLXArray.zeros([128]))
      self.valueOutput = Linear(weight: PolicyValueNetwork.heInitialization(
         inputDimensions: 128, outputDimensions: 1, key: seed == nil ? nil : keys[5]), bias: MLXArray.zeros([1]))

      super.init()
   }

   /// Private initializer that takes weights directly (for cloning and loading)
   private init (
      dense1Weight: MLXArray, dense1Bias: MLXArray,
      dense2Weight: MLXArray, dense2Bias: MLXArray,
      dense3Weight: MLXArray, dense3Bias: MLXArray,
      policyHeadWeight: MLXArray, policyHeadBias: MLXArray,
      valueHiddenWeight: MLXArray, valueHiddenBias: MLXArray,
      valueOutputWeight: MLXArray, valueOutputBias: MLXArray,
      dropoutRate: Float = DEFAULT_DROPOUT) {
      self.dropoutRate = dropoutRate
      self.dense1 = Linear(weight: dense1Weight, bias: dense1Bias)
      self.dense2 = Linear(weight: dense2Weight, bias: dense2Bias)
      self.dense3 = Linear(weight: dense3Weight, bias: dense3Bias)
      self.dropout1 = Dropout(p: dropoutRate)
      self.dropout2 = Dropout(p: dropoutRate)
      self.dropout3 = Dropout(p: dropoutRate)
      self.policyHead = Linear(weight: policyHeadWeight, bias: policyHeadBias)
      self.valueHidden = Linear(weight: valueHiddenWeight, bias: valueHiddenBias)
      self.valueOutput = Linear(weight: valueOutputWeight, bias: valueOutputBias)
      super.init()
   }

   /// Convenience initializer for non-deterministic initialization
   convenience override init () {
      self.init(seed: nil)
   }

   /// Create a deep copy of this network
   /// - Returns: A new PolicyValueNetwork with copied parameters
   public func clone () -> PolicyValueNetwork {
      // Get all parameters from this network
      let params = self.parameters()
      let flattened = params.flattened()

      // Build dictionary from flattened parameters
      var paramDict: [String: MLXArray] = [:]
      for (key, array) in flattened {
         paramDict[key] = array
      }

      // Extract weights and biases for each layer
      let dense1Weight = paramDict["dense1.weight"]!
      let dense1Bias = paramDict["dense1.bias"]!
      let dense2Weight = paramDict["dense2.weight"]!
      let dense2Bias = paramDict["dense2.bias"]!
      let dense3Weight = paramDict["dense3.weight"]!
      let dense3Bias = paramDict["dense3.bias"]!
      let policyHeadWeight = paramDict["policyHead.weight"]!
      let policyHeadBias = paramDict["policyHead.bias"]!
      let valueHiddenWeight = paramDict["valueHidden.weight"]!
      let valueHiddenBias = paramDict["valueHidden.bias"]!
      let valueOutputWeight = paramDict["valueOutput.weight"]!
      let valueOutputBias = paramDict["valueOutput.bias"]!

      // Create new network with copied weights
      return PolicyValueNetwork(
         dense1Weight: dense1Weight, dense1Bias: dense1Bias,
         dense2Weight: dense2Weight, dense2Bias: dense2Bias,
         dense3Weight: dense3Weight, dense3Bias: dense3Bias,
         policyHeadWeight: policyHeadWeight, policyHeadBias: policyHeadBias,
         valueHiddenWeight: valueHiddenWeight, valueHiddenBias: valueHiddenBias,
         valueOutputWeight: valueOutputWeight, valueOutputBias: valueOutputBias,
         dropoutRate: self.dropoutRate)
   }

   /// Initialize a linear layer with He initialization
   /// - Parameters:
   ///   - inputDimensions: Number of input features
   ///   - outputDimensions: Number of output features
   ///   - key: Optional PRNG key for deterministic initialization
   /// - Returns: Weight matrix initialized with He normal distribution
   private static func heInitialization (inputDimensions: Int, outputDimensions: Int, key: MLXArray? = nil) -> MLXArray {
      let stddev = sqrt(2.0 / Float(inputDimensions))
      return MLXRandom.normal([outputDimensions, inputDimensions], key: key) * stddev
   }

   /// Forward pass through the network
   /// - Parameter x: Input tensor of shape [batchSize, 361]
   /// - Returns: Tuple of (policy_logits, value) where policy_logits is [batchSize, 48] and value is [batchSize, 1]
   public func execute (_ x: MLXArray) -> (policyLogits: MLXArray, value: MLXArray) {
      precondition(x.shape.count == 2, "Input must have shape [batchSize, 361]")
      precondition(x.shape[1] == PolicyValueNetwork.INPUT_DIMENSIONS, "Input must have \(PolicyValueNetwork.INPUT_DIMENSIONS) features")

      // Shared trunk with ReLU activations and dropout
      var h = dropout1(relu(dense1(x)))
      h = dropout2(relu(dense2(h)))
      h = dropout3(relu(dense3(h)))

      // Policy head (raw logits, no activation)
      let policyLogits = policyHead(h)

      // Value head (tanh activation for [-1, 1] range)
      let valueHidden = relu(self.valueHidden(h))
      let value = tanh(valueOutput(valueHidden))

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
      metadataEncoder.dateEncodingStrategy = .iso8601
      let metadataData = try metadataEncoder.encode(metadata)
      try metadataData.write(to: metadataURL)

      // Save architecture info (including dropout rate for reproducibility)
      let architectureInfo: [String: Any] = [
         "architectureVersion": PolicyValueNetwork.ARCHITECTURE_VERSION,
         "inputDimensions": PolicyValueNetwork.INPUT_DIMENSIONS,
         "policyDimensions": PolicyValueNetwork.POLICY_DIMENSIONS,
         "dropoutRate": dropoutRate
      ]
      let architectureURL = url.appendingPathComponent("architecture.json")
      let architectureData = try JSONSerialization.data(withJSONObject: architectureInfo, options: .prettyPrinted)
      try architectureData.write(to: architectureURL)
   }

   /// Factory method to create a network with weights loaded from disk
   /// - Parameter url: Directory URL where the model is stored
   /// - Returns: Tuple of (loaded network, metadata)
   /// - Throws: Errors from file operations or deserialization
   public static func load (from url: URL) throws -> (network: PolicyValueNetwork, metadata: ModelMetadata) {
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

      // Helper to load an MLXArray from the weights dictionary
      func loadArray (key: String) throws -> MLXArray {
         guard let weightInfo = weightsDict[key],
               let base64Data = weightInfo["data"] as? String,
               let shape = weightInfo["shape"] as? [Int] else {
            throw NSError(domain: "PolicyValueNetwork", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Missing weight data for parameter: \(key)"])
         }

         guard let data = Data(base64Encoded: base64Data) else {
            throw NSError(domain: "PolicyValueNetwork", code: 4,
                         userInfo: [NSLocalizedDescriptionKey: "Invalid base64 data for parameter: \(key)"])
         }

         let count = shape.reduce(1, *)
         let floatArray = data.withUnsafeBytes { bytes in
            Array(UnsafeBufferPointer<Float>(start: bytes.baseAddress?.assumingMemoryBound(to: Float.self),
                                             count: count))
         }

         return MLXArray(floatArray).reshaped(shape)
      }

      // Load all weights and biases from the dictionary
      let dense1Weight = try loadArray(key: "dense1.weight")
      let dense1Bias = try loadArray(key: "dense1.bias")
      let dense2Weight = try loadArray(key: "dense2.weight")
      let dense2Bias = try loadArray(key: "dense2.bias")
      let dense3Weight = try loadArray(key: "dense3.weight")
      let dense3Bias = try loadArray(key: "dense3.bias")
      let policyHeadWeight = try loadArray(key: "policyHead.weight")
      let policyHeadBias = try loadArray(key: "policyHead.bias")
      let valueHiddenWeight = try loadArray(key: "valueHidden.weight")
      let valueHiddenBias = try loadArray(key: "valueHidden.bias")
      let valueOutputWeight = try loadArray(key: "valueOutput.weight")
      let valueOutputBias = try loadArray(key: "valueOutput.bias")

      // Create network with loaded weights using the private initializer
      let network = PolicyValueNetwork(
         dense1Weight: dense1Weight, dense1Bias: dense1Bias,
         dense2Weight: dense2Weight, dense2Bias: dense2Bias,
         dense3Weight: dense3Weight, dense3Bias: dense3Bias,
         policyHeadWeight: policyHeadWeight, policyHeadBias: policyHeadBias,
         valueHiddenWeight: valueHiddenWeight, valueHiddenBias: valueHiddenBias,
         valueOutputWeight: valueOutputWeight, valueOutputBias: valueOutputBias,
         dropoutRate: savedDropoutRate)

      return (network, metadata)
   }
}


/// Neural network agent that uses the SplendorNetwork to play Splendor
public class SplendorNeuralAgent: AgentProtocol {

   private let network: PolicyValueNetwork
   private let metadata: ModelMetadata

   /// Initialize with untrained network
   /// - Parameter seed: Optional seed for deterministic weight initialization
   public init (seed: UInt64? = nil) {
      self.network = PolicyValueNetwork(seed: seed)
      self.metadata = ModelMetadata(version: "0.1.0", architectureVersion: PolicyValueNetwork.ARCHITECTURE_VERSION)
   }

   /// Initialize by loading a trained model from disk
   /// - Parameter url: Directory URL where the model is stored
   public init (url: URL) throws {
      let (network, metadata) = try PolicyValueNetwork.load(from: url)
      self.network = network
      self.metadata = metadata
   }

   /// Get both policy and value predictions (required by AgentProtocol)
   /// - Parameters:
   ///   - game: The current game state (must be Splendor.Game)
   ///   - currentPlayerIndex: The index of the player whose turn it is
   /// - Returns: Tuple of (policyLogits, valueEstimate) where policyLogits is an
   ///   array of probabilities, one for each canonical move, and valueEstimate is a float between -1 and 1
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

      // Create MLX array with shape [1, 361]
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
}
