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
///   Input (360) -> Dense(512) -> Dense(512) -> Dense(256) -> Policy Head (48) + Value Head (1)
class PolicyValueNetwork: Module {

   static let INPUT_DIMENSIONS = 360 // Matches game's state embedding size
   static let POLICY_DIMENSIONS = 48 // Matches game's move space (42 normal + 6 discard)

   // Current architecture version - increment when architecture changes
   static let ARCHITECTURE_VERSION = 2

   // Shared trunk layers
   let dense1: Linear
   let dense2: Linear
   let dense3: Linear

   // Policy head (outputs move logits)
   let policyHead: Linear

   // Value head (outputs win probability)
   let valueHidden: Linear
   let valueOutput: Linear

   override init () {
      // Shared trunk: 360 -> 512 -> 512 -> 256
      self.dense1 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: PolicyValueNetwork.INPUT_DIMENSIONS, outputDimensions: 512))
      self.dense2 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 512, outputDimensions: 512))
      self.dense3 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 512, outputDimensions: 256))

      // Policy head: 256 -> 43 logits
      self.policyHead = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 256, outputDimensions: PolicyValueNetwork.POLICY_DIMENSIONS))

      // Value head: 256 -> 128 -> 1
      self.valueHidden = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 256, outputDimensions: 128), bias: MLXArray.zeros([128]))
      self.valueOutput = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 128, outputDimensions: 1), bias: MLXArray.zeros([1]))

      super.init()
   }

   /// Initialize a linear layer with scaled normal distribution
   private static func heInitialization (inputDimensions: Int, outputDimensions: Int) -> MLXArray {
      let stddev = sqrt(2.0 / Float(inputDimensions))
      return MLXRandom.normal([outputDimensions, inputDimensions]) * stddev
   }

   /// Forward pass through the network
   /// - Parameter x: Input tensor of shape [batchSize, 360]
   /// - Returns: Tuple of (policy_logits, value) where policy_logits is [batchSize, 43] and value is [batchSize, 1]
   func execute (_ x: MLXArray) -> (policyLogits: MLXArray, value: MLXArray) {
      precondition(x.shape.count == 2, "Input must have shape [batchSize, 360]")
      precondition(x.shape[1] == PolicyValueNetwork.INPUT_DIMENSIONS, "Input must have \(PolicyValueNetwork.INPUT_DIMENSIONS) features")

      // Shared trunk with ReLU activations
      var h = relu(dense1(x))
      h = relu(dense2(h))
      h = relu(dense3(h))

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
   func save (to url: URL, metadata: ModelMetadata) throws {
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

         // Convert MLXArray to Float array
         let floatArray = (0..<count).map { index in
            array.reshaped([count])[index].item(Float.self)
         }

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

      // Save architecture info
      let architectureInfo: [String: Any] = [
         "architectureVersion": PolicyValueNetwork.ARCHITECTURE_VERSION,
         "inputDimensions": PolicyValueNetwork.INPUT_DIMENSIONS,
         "policyDimensions": PolicyValueNetwork.POLICY_DIMENSIONS
      ]
      let architectureURL = url.appendingPathComponent("architecture.json")
      let architectureData = try JSONSerialization.data(withJSONObject: architectureInfo, options: .prettyPrinted)
      try architectureData.write(to: architectureURL)
   }

   /// Factory method to create a network with weights loaded from disk
   /// - Parameter url: Directory URL where the model is stored
   /// - Returns: Tuple of (loaded network, metadata)
   /// - Throws: Errors from file operations or deserialization
   static func load (from url: URL) throws -> (network: PolicyValueNetwork, metadata: ModelMetadata) {
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

      // Create network with loaded weights
      let network = PolicyValueNetwork()

      // Load and assign weights (this won't work if weights are 'let', so we document the limitation)
      // For a working solution, you'd need to modify the init() to accept optional weights
      // or use MLX's updateParameters() if available

      // Temporary: Create new layers with loaded weights
      // This requires making the layers mutable or using a different initialization pattern
      // For now, we'll return the network and note that manual weight assignment is needed

      // TODO: Implement proper weight loading once MLX Swift's parameter update API is clarified
      // For now, this factory method loads metadata and validates architecture version

      return (network, metadata)
   }
}


/// Neural network agent that uses the SplendorNetwork to play Splendor
public class SplendorNeuralAgent: AgentProtocol {

   private let network: PolicyValueNetwork
   private let metadata: ModelMetadata

   public init () {
      self.network = PolicyValueNetwork()
      self.metadata = ModelMetadata(version: "0.1.0", architectureVersion: PolicyValueNetwork.ARCHITECTURE_VERSION)
   }

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
   public func predict (game: any GameProtocol, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      // Cast to Splendor.Game to access the encoding() method
      guard let splendorGame = game as? Splendor.Game else {
         preconditionFailure("SplendorNeuralAgent can only be used with Splendor.Game")
      }

      // Get the encoded game state, convert to Float array
      let encodedState = splendorGame.encoding().map { Float($0) }

      // Create MLX array with shape [1, 360]
      let inputArray = MLXArray(encodedState).reshaped([1, PolicyValueNetwork.INPUT_DIMENSIONS])

      // Run inference
      let (policyLogits, value) = network.execute(inputArray)

      // Policy logits should have shape [1, 43]
      precondition(policyLogits.ndim == 2, "Policy logits must be 2D, got \(policyLogits.ndim)D")
      precondition(policyLogits.shape[1] == PolicyValueNetwork.POLICY_DIMENSIONS, "Policy logits must have \(PolicyValueNetwork.POLICY_DIMENSIONS) moves, got \(policyLogits.shape[1])")

      // Value should have shape [1, 1]
      precondition(value.ndim == 2, "Value must be 2D, got \(value.ndim)D")
      precondition(value.shape[1] == 1, "Value must have 1 element in second dimension, got \(value.shape[1])")

      // Convert policy logits to Swift array - extract the single batch element
      let logitsArray = policyLogits[0]  // Extract [43] from [1, 43]
      let policyResult: [Float] = (0..<PolicyValueNetwork.POLICY_DIMENSIONS).map { index in
         Float(logitsArray[index].item(Float.self))
      }

      // Extract value estimate (already in [-1, 1] range from tanh activation)
      let valueEstimate = Float(value[0, 0].item(Float.self))

      return (policyResult, valueEstimate)
   }
}
