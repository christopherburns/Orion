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
      opts.addOption("Network Trainer", "r", "learning-rate", "Learning rate (default: 0.001)")
      opts.addOption("Network Trainer", "f", "validation-split", "Fraction of data to use for validation (default: 0.1)")
      opts.addOption("Network Trainer", "O", "optimizer", "Optimizer: adam, sgd (default: adam)")
      opts.addOption("Network Trainer", "P", "policy-loss-weight", "Weight for policy loss (default: 1.0)")
      opts.addOption("Network Trainer", "V", "value-loss-weight", "Weight for value loss (default: 1.0)")
      opts.addOption("Network Trainer", "E", "early-stopping", "Stop training if validation loss doesn't improve for N epochs (default: 0 = disabled)")
      opts.addOption("Network Trainer", "w", "weight-decay", "Weight decay (L2 regularization) strength (default: 0.0)")
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
   static func policyLoss (predicted: MLXArray, target: MLXArray, valueWeights: MLXArray) -> Float {
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
      return Float(mean(weightedLoss).item(Float.self))
   }

   /// Compute value loss (MSE between predicted and target values)
   static func valueLoss (predicted: MLXArray, target: MLXArray) -> Float {
      // predicted: [batchSize, 1] values
      // target: [batchSize, 1] values
      let diff = predicted - target
      return Float(mean(diff * diff).item(Float.self))
   }

   /// Compute average validation loss on a set of examples
   /// - Parameters:
   ///   - network: The network to evaluate
   ///   - validationExamples: Array of validation examples
   ///   - batchSize: Batch size for evaluation
   /// - Returns: Tuple of (policyLoss, valueLoss) - average losses over validation set
   static func computeValidationLoss (
      network: PolicyValueNetwork,
      validationExamples: [TrainingExample],
      batchSize: Int) -> (policyLoss: Float, valueLoss: Float) {

      var totalPolicyLoss: Float = 0.0
      var totalValueLoss: Float = 0.0
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

         totalPolicyLoss += valPolLoss
         totalValueLoss += valValLoss
         valBatchCount += 1
      }

      let avgPolicyLoss = totalPolicyLoss / Float(valBatchCount)
      let avgValueLoss = totalValueLoss / Float(valBatchCount)
      return (avgPolicyLoss, avgValueLoss)
   }

   /// Training step: forward pass, compute loss, backward pass
   static func trainingStep (
      network: PolicyValueNetwork,
      optimizer: Optimizer,
      states: MLXArray,
      policyTargets: MLXArray,
      valueTargets: MLXArray,
      policyWeight: Float,
      valueWeight: Float) -> (policyLoss: Float, valueLoss: Float) {

      // Create valueAndGrad function
      let vg = valueAndGrad(model: network) { model, _ in
         // Forward pass with current model state
         let (p, v) = model.execute(states)

         // Compute losses as MLXArrays (needed for gradient computation)
         let logProbs = logSoftmax(p, axis: -1)
         let perExampleLoss = -sum(policyTargets * logProbs, axis: -1)
         let weights = ((valueTargets + 1.0) / 2.0).squeezed(axis: -1)
         let weightedLoss = perExampleLoss * weights
         let pl = mean(weightedLoss)

         let diff = v - valueTargets
         let vl = mean(diff * diff)

         return [policyWeight * pl + valueWeight * vl]
      }

      // Call the gradient function (computes loss and gradients)
      let (_, grads) = vg(network, [])

      // Update parameters
      optimizer.update(model: network, gradients: grads)

      // Forward pass again to get updated predictions for return value
      let (p, v) = network.execute(states)
      let policyLossValue = policyLoss(predicted: p, target: policyTargets, valueWeights: valueTargets)
      let valueLossValue = valueLoss(predicted: v, target: valueTargets)

      return (policyLoss: policyLossValue, valueLoss: valueLossValue)
   }

   /// Create optimizer based on name
   static func createOptimizer (name: String, learningRate: Float, weightDecay: Float) -> Optimizer {
      switch name.lowercased() {
      case "adam":
         if weightDecay > 0 {
            return AdamW(learningRate: learningRate, weightDecay: weightDecay)
         }
         return Adam(learningRate: learningRate)
      case "adamw": return AdamW(learningRate: learningRate, weightDecay: weightDecay)
      case "sgd":  return SGD(learningRate: learningRate, weightDecay: weightDecay)
      default:
         print("Warning: Unknown optimizer '\(name)', defaulting to Adam")
         return Adam(learningRate: learningRate)
      }
   }

   /// Train a model programmatically (without parsing command-line args)
   public static func trainModel (
      inputPath: String,
      modelPath: String?,
      outputPath: String?,
      seed: UInt64 = 42,
      batchSize: Int = 256,
      epochs: Int = 10,
      learningRate: Float = 0.001,
      validationSplit: Float = 0.1,
      optimizerName: String = "adam",
      policyWeight: Float = 1.0,
      valueWeight: Float = 1.0,
      weightDecay: Float = 0.0,
      earlyStoppingPatience: Int = 0) throws {

      print("Loading training data from: \(inputPath)")
      let dataset = try TrainingDataset.load(from: inputPath)
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

      let (network, metadata) = try loadOrCreateNetwork(modelPath: modelPath, seed: seed)
      let optimizer = createOptimizer(name: optimizerName, learningRate: learningRate, weightDecay: weightDecay)

      // Training loop
      print("\nStarting training...")
      var bestValidationLoss = Float.infinity
      var epochLosses: [Float] = []
      var epochsWithoutImprovement = 0
      var bestModelEpoch = 0
      var bestModelMetadata: ModelMetadata? = nil  // Metadata for best model
      var bestNetwork: PolicyValueNetwork? = nil  // Keep best model copy in memory

      for epoch in 1...epochs {
         // Shuffle training data each epoch
         var shuffledTraining = trainingExamples
         shuffledTraining.shuffle()

         var epochPolicyLoss: Float = 0.0
         var epochValueLoss: Float = 0.0
         var batchCount = 0

         // Enable training mode (activates dropout)
         network.train()

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
            let (policyLoss, valueLoss) = trainingStep(
               network: network,
               optimizer: optimizer,
               states: states,
               policyTargets: policyTargets,
               valueTargets: valueTargets,
               policyWeight: policyWeight,
               valueWeight: valueWeight)

            epochPolicyLoss += policyLoss
            epochValueLoss += valueLoss
            batchCount += 1
         }

         let avgEpochPolicyLoss = epochPolicyLoss / Float(batchCount)
         let avgEpochValueLoss = epochValueLoss / Float(batchCount)
         let avgEpochLoss = avgEpochPolicyLoss * policyWeight + avgEpochValueLoss * valueWeight
         epochLosses.append(avgEpochLoss)

         // Validation (disable dropout)
         network.train(false)
         let avgValidationLosses = computeValidationLoss(
            network: network,
            validationExamples: validationExamples,
            batchSize: batchSize)

         let validationLoss = avgValidationLosses.policyLoss * policyWeight + avgValidationLosses.valueLoss * valueWeight

         let bestTag = validationLoss < bestValidationLoss ? "[Best Yet]" : ""
         print("Epoch \(epoch)/\(epochs): Train = \(String(format: "%5.3f", avgEpochPolicyLoss)) / \(String(format: "%5.3f", avgEpochValueLoss)), Val = \(String(format: "%5.3f", avgValidationLosses.policyLoss)) / \(String(format: "%5.3f", avgValidationLosses.valueLoss)) \(bestTag)")

         // Track best model - save to temp file when found (we'll copy it at the end)
         if validationLoss < bestValidationLoss {
            bestValidationLoss = validationLoss
            epochsWithoutImprovement = 0
            bestModelEpoch = epoch
            bestNetwork = network.clone() // Clone the network to keep a true copy in memory

            // Store metadata and clone network
            bestModelMetadata = ModelMetadata(
               version: metadata.version,
               architectureVersion: metadata.architectureVersion,
               createdAt: metadata.createdAt,
               trainingEpochs: epoch,
               trainingLoss: validationLoss,
               description: metadata.description,
               checksum: metadata.checksum)
         } else {
            epochsWithoutImprovement += 1
         }

         // Early stopping
         if earlyStoppingPatience > 0 && epochsWithoutImprovement >= earlyStoppingPatience {
            print("\nEarly stopping triggered: validation loss hasn't improved for \(earlyStoppingPatience) epochs")
            print("Best validation loss: \(String(format: "%.6f", bestValidationLoss)) at epoch \(bestModelEpoch)")
            break
         }
      } // loop over epochs

      // Save best model (or final model if no improvement was found)
      if let outputPath = outputPath {
         let outputURL = URL(fileURLWithPath: outputPath)
         try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)

         if let bestMetadata = bestModelMetadata, let bestNet = bestNetwork, bestModelEpoch != epochs {
            // Save the best model (from earlier epoch)
            try bestNet.save(to: outputURL, metadata: bestMetadata)
            print("\nTraining complete!")
            print("Best model (epoch \(bestModelEpoch)) saved to: \(outputPath)")
            print("Best validation loss: \(String(format: "%.6f", bestValidationLoss)) at epoch \(bestModelEpoch)")
         } else {
            // Best model is the final model (or no improvement found)
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
            print("\nTraining complete!")
            print("Model saved to: \(outputPath)")
            print("Best validation loss: \(String(format: "%.6f", bestValidationLoss)) at epoch \(bestModelEpoch)")
         }

         if earlyStoppingPatience > 0 && epochsWithoutImprovement >= earlyStoppingPatience {
            print("Training stopped early due to no improvement.")
         }
      }
   }

   public static func main () throws {
      let opts = OptionParser(help: "Train a neural network model on training data")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

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
      let optimizerName = opts.get(option: "optimizer", orElse: "adam")
      let policyWeight = opts.get(option: "policy-loss-weight", orElse: Float(1.0))
      let valueWeight = opts.get(option: "value-loss-weight", orElse: Float(1.0))
      let earlyStoppingPatience = opts.get(option: "early-stopping", orElse: 0)
      let weightDecay = opts.get(option: "weight-decay", orElse: Float(0.0))

      // Print configuration
      print("Configuration:")
      print("  Input:            \(inputPath)")
      print("  Output:           \(outputPath ?? "(none)")")
      print("  Model:            \(modelPath ?? "(new)")")
      print("  Epochs:           \(epochs)")
      print("  Batch size:       \(batchSize)")
      print("  Learning rate:    \(learningRate)")
      print("  Weight decay:     \(weightDecay)")
      print("  Validation split: \(String(format: "%.2f", validationSplit))")
      print("  Optimizer:        \(optimizerName)")
      print("  Policy weight:    \(policyWeight)")
      print("  Value weight:     \(valueWeight)")
      print("  Early stopping:   \(earlyStoppingPatience == 0 ? "disabled" : "\(earlyStoppingPatience) epochs")")
      print("  Seed:             \(seed)")

      try trainModel(
         inputPath: inputPath,
         modelPath: modelPath,
         outputPath: outputPath,
         seed: seed,
         batchSize: batchSize,
         epochs: epochs,
         learningRate: learningRate,
         validationSplit: validationSplit,
         optimizerName: optimizerName,
         policyWeight: policyWeight,
         valueWeight: valueWeight,
         weightDecay: weightDecay,
         earlyStoppingPatience: earlyStoppingPatience
      )
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
