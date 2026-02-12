import Foundation
import Core
import Splendor
import Utility

public struct SelfPlayTrainer {

   static func registerOptions (opts: OptionParser) {
      opts.addOption("Self-Play Trainer", "i", "iterations", "Number of self-play iterations (default: 10)")
      opts.addOption("Self-Play Trainer", "g", "games-per-iter", "Number of games to generate per iteration (default: 100)")
      opts.addOption("Self-Play Trainer", "e", "epochs-per-iter", "Number of training epochs per iteration (default: 5)")
      opts.addOption("Self-Play Trainer", "m", "model", "Initial model path (optional, starts with random if not provided)")
      opts.addOption("Self-Play Trainer", "o", "output-dir", "Output directory for models and training data (default: selfplay/)")
      opts.addOption("Self-Play Trainer", "p", "player-count", "Number of players (default: 2)")
      opts.addOption("Self-Play Trainer", "t", "temperature", "Sampling temperature for move selection (default: 1.0)")
      opts.addOption("Self-Play Trainer", "b", "batch-size", "Batch size for training (default: 256)")
      opts.addOption("Self-Play Trainer", "lr", "learning-rate", "Learning rate (default: 0.001)")
      opts.addOption("Self-Play Trainer", "", "accumulate-data", "Accumulate training data across iterations (default: false, only use latest)")
   }

   public static func main () throws {
      let opts = OptionParser(help: "Self-play training loop: generate data, train model, repeat")
      self.registerOptions(opts: opts)
      opts.parse(tokens: CommandLine.arguments, failOnUnknownOption: true, ignoreHelp: false)

      let iterations = opts.get(option: "iterations", orElse: 10)
      let gamesPerIter = opts.get(option: "games-per-iter", orElse: 100)
      let epochsPerIter = opts.get(option: "epochs-per-iter", orElse: 5)
      let initialModelPath = opts.get(option: "model") as String?
      let outputDir = opts.get(option: "output-dir", orElse: "selfplay/")
      let playerCount = opts.get(option: "player-count", orElse: 2)
      let temperature = opts.get(option: "temperature", orElse: Float(1.0))
      let batchSize = opts.get(option: "batch-size", orElse: 256)
      let learningRate = opts.get(option: "learning-rate", orElse: Float(0.001))
      let accumulateData = opts.get(option: "accumulate-data", orElse: false)

      // Create output directory
      try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true, attributes: nil)
      let modelsDir = "\(outputDir)/models"
      let dataDir = "\(outputDir)/data"
      try FileManager.default.createDirectory(atPath: modelsDir, withIntermediateDirectories: true, attributes: nil)
      try FileManager.default.createDirectory(atPath: dataDir, withIntermediateDirectories: true, attributes: nil)

      // Load or create initial model
      var currentModelPath: String?
      if let initialModelPath = initialModelPath {
         currentModelPath = initialModelPath
         print("Starting with model: \(initialModelPath)")
      } else {
         print("Starting with random agent")
      }

      var allTrainingData: [TrainingDataset] = []

      for iteration in 1...iterations {
         print("\n" + String(repeating: "=", count: 80))
         print("ITERATION \(iteration)/\(iterations)")
         print(String(repeating: "=", count: 80))

         // Step 1: Generate training data using current model
         print("\n[Step 1/2] Generating \(gamesPerIter) games...")
         let agentSpec = currentModelPath ?? "random"
         let dataOutputPath = "\(dataDir)/iter\(iteration)"
         let seed = UInt64.random(in: 0..<UInt64.max)

         do {
            try DataGenerator.generateTrainingData(
               gameCount: gamesPerIter,
               playerCount: playerCount,
               agentSpec: agentSpec,
               temperature: temperature,
               seed: seed,
               maxTurns: 1000,
               outputPath: dataOutputPath
            )
         } catch {
            print("Error generating data: \(error)")
            continue
         }

         // Load the generated dataset
         let newDataset = try TrainingDataset.load(from: "\(dataOutputPath).gz")
         print("Generated \(newDataset.totalGames) games with \(newDataset.totalExamples) examples")

         // Accumulate or replace training data
         if accumulateData {
            allTrainingData.append(newDataset)
         } else {
            allTrainingData = [newDataset]
         }

         // Combine all datasets
         var allGames: [GameData] = []
         for dataset in allTrainingData {
            allGames.append(contentsOf: dataset.games)
         }
         let totalExamples = allGames.reduce(0) { $0 + $1.examples.count }
         print("Total training data: \(allGames.count) games, \(totalExamples) examples")

         // Step 2: Train model
         print("\n[Step 2/2] Training model for \(epochsPerIter) epochs...")
         let modelInputPath = currentModelPath
         let modelOutputPath = "\(modelsDir)/iter\(iteration).mlx"

         // Create temporary combined dataset file for training
         let combinedDataset = TrainingDataset(
            generatedAt: Date().iso8601,
            modelPath: currentModelPath,
            temperature: temperature,
            totalGames: allGames.count,
            totalExamples: totalExamples,
            games: allGames
         )
         let tempDataPath = "\(dataDir)/combined_iter\(iteration).gz"
         try combinedDataset.save(to: tempDataPath, compress: true)

         do {
            try NetworkTrainer.trainModel(
               inputPath: tempDataPath,
               modelPath: modelInputPath,
               outputPath: modelOutputPath,
               seed: seed,
               batchSize: batchSize,
               epochs: epochsPerIter,
               learningRate: learningRate,
               validationSplit: 0.1,
               optimizerName: "adam",
               policyWeight: 1.0,
               valueWeight: 1.0
            )

            currentModelPath = modelOutputPath
            print("Trained model saved to: \(modelOutputPath)")
         } catch {
            print("Error training model: \(error)")
            // Continue with previous model if training fails
            if currentModelPath == nil {
               print("No previous model available, stopping")
               break
            }
         }

         // Clean up temporary combined dataset file
         try? FileManager.default.removeItem(atPath: tempDataPath)
      }

      print("\n" + String(repeating: "=", count: 80))
      print("Self-play training complete!")
      print("Final model: \(currentModelPath ?? "none")")
      print("Models directory: \(modelsDir)")
      print("Data directory: \(dataDir)")
   }
}

