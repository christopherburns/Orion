import Foundation
import MLX
import MLXNN
import Core

/// Neural network for Splendor game playing
/// Architecture:
///   Input (360) -> Dense(512) -> Dense(512) -> Dense(256) -> Policy Head (43) + Value Head (1)
class PolicyValueNetwork: Module {

   let INPUT_DIMENSIONS = 360 // Matches game's state embedding size
   let POLICY_DIMENSIONS = 43 // Matches game's move space

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
      self.dense1 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: INPUT_DIMENSIONS, outputDimensions: 512))
      self.dense2 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 512, outputDimensions: 512))
      self.dense3 = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 512, outputDimensions: 256))

      // Policy head: 256 -> 43 logits
      self.policyHead = Linear(weight: PolicyValueNetwork.heInitialization(inputDimensions: 256, outputDimensions: POLICY_DIMENSIONS))

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
      precondition(x.shape[1] == 360, "Input must have 360 features")

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
}


/// Neural network agent that uses the SplendorNetwork to play Splendor
class SplendorNeuralAgent: AgentProtocol {

   private let network: PolicyValueNetwork

   init () {
      self.network = PolicyValueNetwork()
   }

   /// Initialize with a pre-trained network (for loading weights)
   init (network: PolicyValueNetwork) {
      self.network = network
   }

   /// Calculates move preferences (logits) for the current game state
   /// - Parameters:
   ///   - game: The current game state
   ///   - currentPlayerIndex: The index of the player whose turn it is
   /// - Returns: Array of logits, one for each canonical move
   func calculateMovePreferences (game: any GameProtocol, currentPlayerIndex: Int) -> [Float] {
      // Cast to Splendor.Game to access the encoding() method
      guard let splendorGame = game as? Splendor.Game else {
         preconditionFailure("SplendorNeuralAgent can only be used with Splendor.Game")
      }

      // Get the encoded game state
      let encodedState = splendorGame.encoding()

      // Convert Float16 array to Float array for MLX
      let floatState = encodedState.map { Float($0) }

      // Create MLX array with shape [1, 360] (batch size of 1)
      let inputArray = MLXArray(floatState).reshaped([1, 360])

      // Run inference
      let (policyLogits, _) = network.execute(inputArray)

      // Convert output back to [Float]
      // policyLogits has shape [1, 43], we need to extract the single row
      let logitsArray = policyLogits.reshaped([43])

      // Convert MLXArray to Swift array
      let result: [Float] = (0..<43).map { index in
         Float(logitsArray[index].item(Float.self))
      }

      return result
   }

   /// Get both policy and value predictions (useful for training)
   /// - Parameters:
   ///   - game: The current game state
   ///   - currentPlayerIndex: The index of the player whose turn it is
   /// - Returns: Tuple of (policyLogits, valueEstimate)
   func predict (game: Splendor.Game, currentPlayerIndex: Int) -> (policyLogits: [Float], valueEstimate: Float) {
      // Get the encoded game state
      let encodedState = game.encoding()

      // Convert Float16 array to Float array for MLX
      let floatState = encodedState.map { Float($0) }

      // Create MLX array with shape [1, 360]
      let inputArray = MLXArray(floatState).reshaped([1, 360])

      // Run inference
      let (policyLogits, value) = network.execute(inputArray)

      // Convert policy logits to Swift array
      let logitsArray = policyLogits.reshaped([43])
      let policyResult: [Float] = (0..<43).map { index in
         Float(logitsArray[index].item(Float.self))
      }

      // Extract value estimate
      let valueEstimate = Float(value[0].item(Float.self))

      return (policyResult, valueEstimate)
   }
}
