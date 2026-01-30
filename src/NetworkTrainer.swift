import Swift
import Foundation
import Core
import Splendor
import Utility
import MLX
import MLXNN
import MLXOptimizers

// Note: Uses loadOrCreateNetwork from Common.swift

public struct NetworkTrainer {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Network Trainer", "i", "input", "Input training data file or directory (required)")
      opts.addOption("Network Trainer", "m", "model", "Path to input model file to continue training (default: create new untrained model)")
      opts.addOption("Network Trainer", "o", "output", "Output path for trained model (default: models/model_TIMESTAMP.mlx)")
      opts.addOption("Network Trainer", "s", "seed", "Random seed for reproducibility (default: random)")
      opts.addOption("Network Trainer", "b", "batch-size", "Batch size for training (default: 256)")
      opts.addOption("Network Trainer", "e", "epochs", "Number of training epochs (default: 10)")
      opts.addOption("Network Trainer", "lr", "learning-rate", "Learning rate (default: 0.001)")
      opts.addOption("Network Trainer", "v", "validation-split", "Fraction of data to use for validation (default: 0.1)")
      opts.addOption("Network Trainer", "save-interval", "save-interval", "Save checkpoint every N epochs (default: 5)")
      opts.addOption("Network Trainer", "opt", "optimizer", "Optimizer: adam, sgd (default: adam)")
      opts.addOption("Network Trainer", "ploss", "policy-loss-weight", "Weight for policy loss (default: 1.0)")
      opts.addOption("Network Trainer", "vloss", "value-loss-weight", "Weight for value loss (default: 1.0)")
   }

   /// Load or create a PolicyValueNetwork with metadata
   /// - Parameters:
   ///   - modelPath: Optional path to model file. If nil, creates new untrained model
   ///   - seed: Random seed for new model initialization (ignored if loading)
   /// - Returns: Tuple of (network, metadata)
   /// - Throws: Errors from file operations or model loading
   static func loadOrCreateNetwork (modelPath: String?, seed: UInt64) throws -> (network: PolicyValueNetwork, metadata: ModelMetadata) {
      if let modelPath = modelPath {
         return try PolicyValueNetwork.load(from: URL(fileURLWithPath: modelPath))
      } else {
         let network = PolicyValueNetwork(seed: seed)
         let metadata = ModelMetadata(
            version: "0.1.0",
            architectureVersion: PolicyValueNetwork.ARCHITECTURE_VERSION,
            createdAt: Date()
         )
         return (network, metadata)
      }
   }


   /// Compute policy loss (cross-entropy between predicted and target distributions)
   /// Weighted by value targets: only learn from winning moves (weight=1.0), ignore losing/tied moves (weight=0.0)
   static func policyLoss (predicted: MLXArray, target: MLXArray, valueWeights: MLXArray) -> MLXArray {
      // predicted: [batchSize, 48] logits
      // target: [batchSize, 48] probabilities
      // valueWeights: [batchSize, 1] - value targets in [-1, 1] range
      // Cross-entropy per example: -sum(target * log(softmax(predicted)))
      let logProbs = logSoftmax(predicted, axis: -1)
      let perExampleLoss = -sum(target * logProbs, axis: -1)  // [batchSize]
      // Weight by value: (value + 1) / 2 maps [-1, 1] to [0, 1]
      //    value=+1.0 → weight=1.0 (learn from winning moves)
      //    value=-1.0 → weight=0.0 (ignore losing moves)
      //    value=0.0  → weight=0.5 (half-weight for tied games)
      let weights = ((valueWeights + 1.0) / 2.0).squeezed(axis: -1)  // [batchSize]
      let weightedLoss = perExampleLoss * weights  // [batchSize]
      // Average weighted loss (mean handles zero weights correctly)
      return mean(weightedLoss)
   }

   /// Compute value loss (MSE between predicted and target values)
   static func valueLoss (predicted: MLXArray, target: MLXArray) -> MLXArray {
      // predicted: [batchSize, 1] values
      // target: [batchSize, 1] values
      let diff = predicted - target
      return mean(diff * diff)
   }

   /// Training step: forward pass, compute loss, backward pass
   static func trainingStep (
      network: PolicyValueNetwork,
      optimizer: Optimizer,
      states: MLXArray,
      policyTargets: MLXArray,
      valueTargets: MLXArray,
      policyWeight: Float,
      valueWeight: Float) -> Float {

      // Create valueAndGrad function
      let vg = valueAndGrad(model: network) { model, arrays in
         let states = arrays[0]
         let policyTargets = arrays[1]
         let valueTargets = arrays[2]
         let (p, v) = model.execute(states)
         let pl = policyLoss(predicted: p, target: policyTargets, valueWeights: valueTargets)
         let vl = valueLoss(predicted: v, target: valueTargets)
         return [policyWeight * pl + valueWeight * vl]
      }

      // Call the gradient function (computes loss and gradients)
      let (lossArray, grads) = vg(network, [states, policyTargets, valueTargets])
      let totalLoss = lossArray[0]

      // Update parameters
      optimizer.update(model: network, gradients: grads)

      // Return total loss as Float
      return Float(totalLoss.item(Float.self))
   }

   /// Create optimizer based on name
   static func createOptimizer (name: String, learningRate: Float) -> Optimizer {
      switch name.lowercased() {
      case "adam":
         return Adam(learningRate: learningRate)
      case "sgd":
         return SGD(learningRate: learningRate)
      default:
         print("Warning: Unknown optimizer '\(name)', defaulting to Adam")
         return Adam(learningRate: learningRate)
      }
   }

   public static func main () throws {
      let opts = OptionParser(help: "Train a neural network model on training data")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      // Parse command-line options
      guard let inputPath = opts.get(option: "input") as String? else {
         print("Error: --input is required")
         print("Use --help for usage information")
         return
      }

      let modelPath = opts.get(option: "model") as String?
      let outputPath = opts.get(option: "output") as String?
      let seed = opts.get(option: "seed", orElse: UInt64(42))
      let batchSize = opts.get(option: "batch-size", orElse: 256)
      let epochs = opts.get(option: "epochs", orElse: 10)
      let learningRate = opts.get(option: "learning-rate", orElse: Float(0.001))
      let validationSplit = opts.get(option: "validation-split", orElse: Float(0.1))
      let saveInterval = opts.get(option: "save-interval", orElse: 5)
      let optimizerName = opts.get(option: "optimizer", orElse: "adam")
      let policyWeight = opts.get(option: "policy-loss-weight", orElse: Float(1.0))
      let valueWeight = opts.get(option: "value-loss-weight", orElse: Float(1.0))

      print("Loading training data from: \(inputPath)")
      let dataset = try TrainingDataset.loadTrainingData(from: inputPath)
      print("Loaded \(dataset.totalGames) games with \(dataset.totalExamples) total examples")

      // Flatten all examples from all games
      var allExamples: [TrainingExample] = []
      for game in dataset.games {
         allExamples.append(contentsOf: game.examples)
      }

      // Shuffle examples
      allExamples.shuffle()

      // Split into training and validation sets
      let validationCount = Int(Float(allExamples.count) * validationSplit)
      let validationExamples = Array(allExamples.prefix(validationCount))
      let trainingExamples = Array(allExamples.suffix(allExamples.count - validationCount))

      print("Training set: \(trainingExamples.count) examples")
      print("Validation set: \(validationExamples.count) examples")

      // Initialize or load network
      if let modelPath = modelPath {
         print("Loading model from: \(modelPath)")
      } else {
         print("Creating new untrained model")
      }
      let (network, metadata) = try loadOrCreateNetwork(modelPath: modelPath, seed: seed)

      // Create optimizer
      let optimizer = createOptimizer(name: optimizerName, learningRate: learningRate)
      print("Using optimizer: \(optimizerName) with learning rate: \(learningRate)")

      // Training loop
      print("\nStarting training...")
      var bestValidationLoss = Float.infinity
      var epochLosses: [Float] = []

      for epoch in 1...epochs {
         // Shuffle training data each epoch
         var shuffledTraining = trainingExamples
         shuffledTraining.shuffle()

         var epochLoss: Float = 0.0
         var batchCount = 0

         // Process in batches
         for batchStart in stride(from: 0, to: shuffledTraining.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, shuffledTraining.count)
            let batch = Array(shuffledTraining[batchStart..<batchEnd])

            // Prepare batch data - flatten arrays and reshape
            let statesFlat = batch.flatMap { $0.state }
            let states = MLXArray(statesFlat).reshaped([batch.count, PolicyValueNetwork.INPUT_DIMENSIONS])
            let policyFlat = batch.flatMap { $0.policy }
            let policyTargets = MLXArray(policyFlat).reshaped([batch.count, PolicyValueNetwork.POLICY_DIMENSIONS])
            let valueTargets = MLXArray(batch.map { $0.value }).reshaped([batch.count, 1])

            // Training step
            let loss = trainingStep(
               network: network,
               optimizer: optimizer,
               states: states,
               policyTargets: policyTargets,
               valueTargets: valueTargets,
               policyWeight: policyWeight,
               valueWeight: valueWeight)

            epochLoss += loss
            batchCount += 1
         }

         let avgEpochLoss = epochLoss / Float(batchCount)
         epochLosses.append(avgEpochLoss)

         // Validation
         var validationLoss: Float = 0.0
         var valBatchCount = 0
         for valBatchStart in stride(from: 0, to: validationExamples.count, by: batchSize) {
            let valBatchEnd = min(valBatchStart + batchSize, validationExamples.count)
            let valBatch = Array(validationExamples[valBatchStart..<valBatchEnd])

            let valStatesFlat = valBatch.flatMap { $0.state }
            let valStates = MLXArray(valStatesFlat).reshaped([valBatch.count, PolicyValueNetwork.INPUT_DIMENSIONS])
            let valPolicyFlat = valBatch.flatMap { $0.policy }
            let valPolicyTargets = MLXArray(valPolicyFlat).reshaped([valBatch.count, PolicyValueNetwork.POLICY_DIMENSIONS])
            let valValueTargets = MLXArray(valBatch.map { $0.value }).reshaped([valBatch.count, 1])

            let (valPolicyLogits, valValuePred) = network.execute(valStates)
            let valPolLoss = policyLoss(predicted: valPolicyLogits, target: valPolicyTargets, valueWeights: valValueTargets)
            let valValLoss = valueLoss(predicted: valValuePred, target: valValueTargets)
            let valTotalLoss = policyWeight * valPolLoss + valueWeight * valValLoss

            validationLoss += Float(valTotalLoss.item(Float.self))
            valBatchCount += 1
         }

         let avgValidationLoss = validationLoss / Float(valBatchCount)

         print("Epoch \(epoch)/\(epochs): Train Loss = \(String(format: "%.6f", avgEpochLoss)), Val Loss = \(String(format: "%.6f", avgValidationLoss))")

         // Track best model
         if avgValidationLoss < bestValidationLoss {
            bestValidationLoss = avgValidationLoss
         }

         // Save checkpoint
         if epoch % saveInterval == 0 || epoch == epochs {
            let checkpointPath = outputPath ?? "models/checkpoint_epoch\(epoch).mlx"
            let checkpointURL = URL(fileURLWithPath: checkpointPath)

            // Create directory if needed
            try FileManager.default.createDirectory(at: checkpointURL.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)

            let checkpointMetadata = ModelMetadata(
               version: metadata.version,
               architectureVersion: metadata.architectureVersion,
               createdAt: metadata.createdAt,
               trainingEpochs: epoch,
               trainingLoss: avgValidationLoss,
               description: metadata.description,
               checksum: metadata.checksum
            )

            try network.save(to: checkpointURL, metadata: checkpointMetadata)
            print("Saved checkpoint to: \(checkpointPath)")
         }
      }

      // Save final model
      if let outputPath = outputPath {
         let outputURL = URL(fileURLWithPath: outputPath)
         try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)

         let finalMetadata = ModelMetadata(
            version: metadata.version,
            architectureVersion: metadata.architectureVersion,
            createdAt: metadata.createdAt,
            trainingEpochs: epochs,
            trainingLoss: bestValidationLoss,
            description: metadata.description,
            checksum: metadata.checksum
         )

         try network.save(to: outputURL, metadata: finalMetadata)
         print("\nTraining complete! Final model saved to: \(outputPath)")
         print("Best validation loss: \(String(format: "%.6f", bestValidationLoss))")
      }
   }
}

// Helper extension for Date ISO8601 formatting
extension Date {
   var iso8601: String {
      let formatter = ISO8601DateFormatter()
      formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
      return formatter.string(from: self)
   }
}
